"""Run paged_kv_real_gpu.py N times and emit median +/- spread statistics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0, "mean": 0.0}
    return {
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "mean": float(statistics.mean(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--model-id", default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--max-active", type=int, default=8)
    parser.add_argument("--seed", type=int, default=5070)
    parser.add_argument("--output-json", default="bench/results/paged_kv_repeat.json")
    parser.add_argument("--output-md", default="bench/results/paged_kv_repeat.md")
    parser.add_argument("--output-manifest", default="bench/results/paged_kv_repeat.manifest.json")
    args = parser.parse_args()

    run_dir = ROOT / "bench" / "results" / "paged_kv_repeat" / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    individual: list[dict] = []
    for i in range(args.runs):
        out = run_dir / f"run_{i + 1:02d}.json"
        cmd = [
            sys.executable,
            str(ROOT / "bench" / "scripts" / "paged_kv_real_gpu.py"),
            "--model-id",
            args.model_id,
            "--requests",
            str(args.requests),
            "--max-active",
            str(args.max_active),
            "--seed",
            str(args.seed),
            "--output-json",
            str(out.relative_to(ROOT)),
            "--output-md",
            str((run_dir / f"run_{i + 1:02d}.md").relative_to(ROOT)),
            "--output-csv",
            str((run_dir / f"run_{i + 1:02d}_smi.csv").relative_to(ROOT)),
        ]
        print(f"[repeat {i + 1}/{args.runs}] starting...", flush=True)
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        payload = json.loads(out.read_text(encoding="utf-8"))
        individual.append(payload)
        print(
            f"[repeat {i + 1}/{args.runs}] "
            f"cont={payload['modes']['contiguous']['output_tokens_per_sec']:.2f} "
            f"paged={payload['modes']['paged']['output_tokens_per_sec']:.2f} tok/s",
            flush=True,
        )

    cont_tps = [float(r["modes"]["contiguous"]["output_tokens_per_sec"]) for r in individual]
    paged_tps = [float(r["modes"]["paged"]["output_tokens_per_sec"]) for r in individual]
    deltas = [p - c for p, c in zip(paged_tps, cont_tps)]
    delta_pct = [100.0 * d / c if c else 0.0 for d, c in zip(deltas, cont_tps)]

    cont_smi = [float(r["modes"]["contiguous"]["peak_nvidia_smi_mb"]) for r in individual]
    paged_smi = [float(r["modes"]["paged"]["peak_nvidia_smi_mb"]) for r in individual]
    cont_torch = [float(r["modes"]["contiguous"]["peak_torch_cuda_mb"]) for r in individual]
    paged_torch = [float(r["modes"]["paged"]["peak_torch_cuda_mb"]) for r in individual]

    cont_stats = _stats(cont_tps)
    paged_stats = _stats(paged_tps)
    delta_stats = _stats(deltas)
    delta_pct_stats = _stats(delta_pct)

    median_cont = cont_stats["median"]
    median_paged = paged_stats["median"]
    parity_within_noise = abs(median_paged - median_cont) <= max(cont_stats["stdev"], paged_stats["stdev"], 0.5)

    timestamp = datetime.now(timezone.utc).isoformat()
    aggregate = {
        "timestamp_utc": timestamp,
        "measurement": "repeat_run_aggregate",
        "model_id": args.model_id,
        "workload": {
            "request_count": args.runs,
            "repeat_count": args.runs,
            "seed": args.seed,
            "max_active": args.max_active,
            "underlying_requests_per_run": args.requests,
        },
        "throughput_tokens_per_sec": {
            "contiguous": {**cont_stats, "samples": cont_tps},
            "paged": {**paged_stats, "samples": paged_tps},
            "paged_minus_contiguous": {**delta_stats, "samples": deltas},
            "paged_minus_contiguous_percent": {**delta_pct_stats, "samples": delta_pct},
        },
        "peak_nvidia_smi_mb": {
            "contiguous": _stats(cont_smi),
            "paged": _stats(paged_smi),
        },
        "peak_torch_cuda_mb": {
            "contiguous": _stats(cont_torch),
            "paged": _stats(paged_torch),
        },
        "conclusion": {
            "throughput_at_parity_within_noise": parity_within_noise,
            "median_paged_over_contiguous_ratio": median_paged / median_cont if median_cont else 0.0,
        },
        "individual_runs": [str(p.relative_to(ROOT)) for p in sorted(run_dir.glob("run_*.json"))],
    }

    out_json = ROOT / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    md_lines = [
        "# Paged KV — repeat-run throughput stability",
        "",
        f"- Runs: `{args.runs}` (same workload: {args.requests} req, max_active {args.max_active}, seed {args.seed})",
        f"- Model: `{args.model_id}`",
        f"- Aggregate UTC: `{timestamp}`",
        "",
        "## Throughput (tokens/sec)",
        "",
        "| Path | median | stdev | min | max |",
        "|------|--------|-------|-----|-----|",
        f"| Contiguous | {cont_stats['median']:.2f} | {cont_stats['stdev']:.2f} | {cont_stats['min']:.2f} | {cont_stats['max']:.2f} |",
        f"| Paged | {paged_stats['median']:.2f} | {paged_stats['stdev']:.2f} | {paged_stats['min']:.2f} | {paged_stats['max']:.2f} |",
        f"| Paged − contiguous | {delta_stats['median']:+.2f} | {delta_stats['stdev']:.2f} | {delta_stats['min']:+.2f} | {delta_stats['max']:+.2f} |",
        "",
        f"**Conclusion:** throughput at parity within noise = **{parity_within_noise}** "
        f"(median gap {delta_stats['median']:+.2f} tok/s, {delta_pct_stats['median']:+.1f}%).",
        "",
        "## Peak nvidia-smi (MB)",
        "",
        f"- Contiguous median: {statistics.median(cont_smi):.1f} (spread {min(cont_smi):.1f}–{max(cont_smi):.1f})",
        f"- Paged median: {statistics.median(paged_smi):.1f} (spread {min(paged_smi):.1f}–{max(paged_smi):.1f})",
        "",
        "Individual runs: `bench/results/paged_kv_repeat/runs/`.",
    ]
    out_md = ROOT / args.output_md
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    manifest = {
        "run_id": "paged-kv-repeat-runs",
        "timestamp_utc": timestamp,
        "system_under_test": "phase2",
        "model_id": args.model_id,
        "rows": args.runs,
        "gpu_count": 1,
    }
    out_manifest = ROOT / args.output_manifest
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote: {out_json}")
    print(f"wrote: {out_md}")
    print(f"wrote: {out_manifest}")
    print(
        f"throughput median: contiguous {cont_stats['median']:.2f}, "
        f"paged {paged_stats['median']:.2f} (+/- {paged_stats['stdev']:.2f})"
    )


if __name__ == "__main__":
    main()
