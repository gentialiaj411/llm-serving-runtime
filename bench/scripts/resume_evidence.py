from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_step(name: str, cmd: list[str], required: bool) -> dict[str, object]:
    try:
        cp = subprocess.run(cmd, check=True, text=True, capture_output=True)
        return {"name": name, "status": "passed", "required": required, "command": cmd, "stdout": cp.stdout.strip()}
    except subprocess.CalledProcessError as exc:
        return {
            "name": name,
            "status": "blocked" if not required else "failed",
            "required": required,
            "command": cmd,
            "returncode": exc.returncode,
            "stderr": (exc.stderr or "").strip()[-4000:],
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--run-id-prefix", default="resume-evidence")
    parser.add_argument("--output", default="bench/results/resume-evidence-summary.json")
    parser.add_argument("--chaos-worker-pids", default="")
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--skip-chaos", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id_prefix
    steps: list[dict[str, object]] = []
    py = args.python

    steps.append(run_step("unit_tests", [py, "-m", "unittest", "discover", "-s", "tests"], required=True))
    steps.append(
        run_step(
            "phase2_integration_test",
            [py, "-m", "unittest", "discover", "-s", "tests", "-p", "test_phase2_integration.py"],
            required=True,
        )
    )
    steps.append(
        run_step(
            "kv_pressure",
            [py, "bench/scripts/kv_pressure.py", "--output", "bench/results/kv-pressure.json"],
            required=True,
        )
    )

    if not args.skip_throughput:
        steps.append(
            run_step(
                "throughput_phase2_real_inference",
                [
                    py,
                    "bench/harness/run.py",
                    "--base-url",
                    args.base_url,
                    "--system",
                    "phase2",
                    "--model",
                    args.model,
                    "--scenarios",
                    "bench/scenarios/phase2_real_inference_concurrency.yaml",
                    "--run-id",
                    f"{run_id}-phase2-real-inference",
                    "--inference-mode",
                    "real_model_inference",
                    "--determinism-check",
                    "skip",
                ],
                required=False,
            )
        )

    if not args.skip_chaos:
        if args.chaos_worker_pids.strip():
            steps.append(
                run_step(
                    "chaos_benchmark",
                    [
                        py,
                        "bench/chaos/run_chaos.py",
                        "--run-id",
                        f"{run_id}-chaos",
                        "--worker-pids",
                        args.chaos_worker_pids,
                        "--duration-s",
                        "30",
                        "--concurrency",
                        "8",
                        "--kill-interval-s",
                        "5",
                        "--max-kills",
                        "3",
                        "--require-injection",
                        "--output",
                        "bench/results/chaos-resume-evidence.json",
                    ],
                    required=False,
                )
            )
        else:
            steps.append(
                {
                    "name": "chaos_benchmark",
                    "status": "blocked",
                    "required": False,
                    "reason": "missing --chaos-worker-pids; cannot inject worker failures reproducibly",
                }
            )

    steps.append(run_step("manifest_schema_validation", [py, "bench/scripts/validate_manifests.py"], required=True))

    summary = {
        "steps": steps,
        "claims_supported": [
            "openai_compatible_runtime_path" if any(s["name"] == "phase2_integration_test" and s["status"] == "passed" for s in steps) else "TODO/VERIFY",
            "kv_modeled_reduction" if any(s["name"] == "kv_pressure" and s["status"] == "passed" for s in steps) else "TODO/VERIFY",
            "throughput_real_inference" if any(s["name"] == "throughput_phase2_real_inference" and s["status"] == "passed" for s in steps) else "TODO/VERIFY",
            "chaos_recovery" if any(s["name"] == "chaos_benchmark" and s["status"] == "passed" for s in steps) else "TODO/VERIFY",
        ],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote: {out}")
    for step in steps:
        print(f"{step['name']}: {step['status']}")

    required_failures = [s for s in steps if s["required"] and s["status"] != "passed"]
    return 1 if required_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
