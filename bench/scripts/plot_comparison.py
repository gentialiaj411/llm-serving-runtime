from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "bench" / "results"
OUTPUT_PATH = RESULTS_DIR / "comparison.png"
ORCAFORGE_CSV = RESULTS_DIR / "phase2-rtx5070-throughput.csv"
VLLM_CSV = RESULTS_DIR / "vllm-baseline-final.csv"
ORCAFORGE_HEAD_TO_HEAD_CSV = RESULTS_DIR / "orcaforge-head-to-head-tonight.csv"
VLLM_HEAD_TO_HEAD_CSV = RESULTS_DIR / "vllm-head-to-head-tonight.csv"
ORCAFORGE_SHARED_PREFIX_CSV = RESULTS_DIR / "orcaforge-shared-prefix-tonight.csv"
VLLM_SHARED_PREFIX_ON_CSV = RESULTS_DIR / "vllm-shared-prefix-on-tonight.csv"

TARGET_ROWS = [
    ("short_short", "1", "short_short c1"),
    ("short_short", "16", "short_short c16"),
    ("short_long", "1", "short_long c1"),
    ("short_long", "16", "short_long c16"),
    ("shared_prefix", "8", "shared_prefix c8"),
]

METRICS = [
    ("tokens_per_sec_output", "Output tokens/sec"),
]


def manifest_mode(csv_path: Path) -> str | None:
    manifest_path = csv_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    mode = manifest.get("inference_mode")
    return str(mode) if mode is not None else None


def load_target_row(csv_path: Path, scenario_id: str, concurrency: str) -> dict[str, float] | None:
    if not csv_path.exists():
        return None
    if manifest_mode(csv_path) != "real_model_inference":
        return None

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("scenario_id") != scenario_id:
                continue
            if row.get("concurrency") != concurrency:
                continue
            if float(row.get("success_rate", "0") or 0) <= 0:
                return None
            values: dict[str, float] = {}
            for key, _ in METRICS:
                try:
                    values[key] = float(row[key])
                except (KeyError, TypeError, ValueError):
                    return None
            return values
    return None


def summarize() -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for scenario_id, concurrency, label in TARGET_ROWS:
        orca_csv = ORCAFORGE_SHARED_PREFIX_CSV if scenario_id == "shared_prefix" else ORCAFORGE_HEAD_TO_HEAD_CSV
        vllm_csv = VLLM_SHARED_PREFIX_ON_CSV if scenario_id == "shared_prefix" else VLLM_HEAD_TO_HEAD_CSV
        orcaforge_row = load_target_row(orca_csv, scenario_id, concurrency)
        vllm_row = load_target_row(vllm_csv, scenario_id, concurrency)
        if orcaforge_row is None or vllm_row is None:
            continue
        summary[label] = {
            "Orcaforge": orcaforge_row["tokens_per_sec_output"],
            "vLLM": vllm_row["tokens_per_sec_output"],
        }
    return summary


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = summarize()

    if not summary:
        print("Need fresh Orcaforge/vLLM head-to-head artifacts; no chart written.")
        return 0

    systems = ["Orcaforge", "vLLM"]
    labels = list(summary)
    x_positions = range(len(labels))
    width = 0.36
    offsets = {
        systems[0]: -width / 2,
        systems[1]: width / 2,
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    for system in systems:
        values = [summary[label][system] for label in labels]
        bars = ax.bar([x + offsets[system] for x in x_positions], values, width, label=system)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)

    ax.set_title("Fresh Real-Inference Head-to-Head (TinyLlama, vLLM 0.21.0)")
    ax.set_ylabel("Output tokens/sec")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=160)
    print(f"wrote: {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
