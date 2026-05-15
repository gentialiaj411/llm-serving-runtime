from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
import time
from pathlib import Path

import httpx


def is_pid_alive(pid: int) -> bool:
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            return False
        return any(line.strip().strip('"') and "INFO:" not in line for line in out.splitlines())

    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


async def spam_requests(base_url: str, model: str, concurrency: int, duration_s: int) -> dict:
    stop_at = time.time() + duration_s
    ok = 0
    fail = 0

    async with httpx.AsyncClient(timeout=20.0) as client:
        async def worker_loop() -> None:
            nonlocal ok, fail
            while time.time() < stop_at:
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": "chaos test prompt"}],
                    "max_tokens": 32,
                    "temperature": 0.0,
                }
                try:
                    r = await client.post(base_url, json=body)
                    if r.status_code == 200:
                        ok += 1
                    else:
                        fail += 1
                except Exception:
                    fail += 1
                await asyncio.sleep(0.01)

        await asyncio.gather(*[asyncio.create_task(worker_loop()) for _ in range(concurrency)])

    total = ok + fail
    return {
        "ok": ok,
        "fail": fail,
        "total": total,
        "success_rate": (ok / total) if total else 0.0,
    }


def kill_random_worker(worker_pids: list[int]) -> int | None:
    is_windows = os.name == "nt"
    alive = []
    for pid in worker_pids:
        if is_pid_alive(pid):
            alive.append(pid)
    if not alive:
        return None
    victim = random.choice(alive)
    if is_windows:
        subprocess.call(["taskkill", "/PID", str(victim), "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.kill(victim, 9)
        except Exception:
            return None
    return victim


async def main_async(args: argparse.Namespace) -> None:
    worker_pids = [int(x) for x in args.worker_pids.split(",") if x.strip()]
    random.seed(args.seed)

    injection_events = []

    async def injector() -> None:
        end = time.time() + args.duration_s
        while time.time() < end:
            await asyncio.sleep(args.kill_interval_s)
            victim = kill_random_worker(worker_pids)
            if victim is not None:
                injection_events.append({"ts": int(time.time() * 1000), "killed_pid": victim})

    load_task = asyncio.create_task(
        spam_requests(args.base_url, args.model, args.concurrency, args.duration_s)
    )
    inj_task = asyncio.create_task(injector())

    result = await load_task
    await inj_task

    out = {
        "run_id": args.run_id,
        "timestamp_unix_ms": int(time.time() * 1000),
        "base_url": args.base_url,
        "duration_s": args.duration_s,
        "concurrency": args.concurrency,
        "result": result,
        "injections": injection_events,
        "sla_target_success_rate": args.sla_target,
        "sla_pass": result["success_rate"] >= args.sla_target,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--worker-pids", required=True, help="Comma-separated worker process ids")
    p.add_argument("--duration-s", type=int, default=60)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--kill-interval-s", type=float, default=8.0)
    p.add_argument("--sla-target", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="bench/results/chaos-result.json")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
