from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "bench" / "results"
OUTPUT_PATH = RESULTS_DIR / "comparison.png"

METRICS = [
    ("tokens_per_sec_output", "Output tokens/sec"),
    ("ttft_ms_p50", "TTFT p50 (ms)"),
    ("inter_token_latency_p50", "ITL p50 (ms)"),
    ("est_dollars_per_million_output_tokens", "$/M output tokens"),
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


def system_label(system_under_test: str) -> str | None:
    system = system_under_test.lower()
    if "vllm" in system:
        return "vLLM"
    if "transformers" in system:
        return "Orcaforge"
    return None


def load_rows() -> dict[str, list[dict[str, float]]]:
    by_system: dict[str, list[dict[str, float]]] = {}
    for csv_path in sorted(RESULTS_DIR.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            continue

        mode = manifest_mode(csv_path)
        if mode != "real_model_inference":
            continue
        for row in rows:
            label = system_label(row.get("system_under_test", ""))
            if label is None:
                continue

            values: dict[str, float] = {}
            for key, _ in METRICS:
                try:
                    values[key] = float(row[key])
                except (KeyError, TypeError, ValueError):
                    break
            else:
                by_system.setdefault(label, []).append(values)
    return by_system


def summarize(rows_by_system: dict[str, list[dict[str, float]]]) -> dict[str, dict[str, float]]:
    return {
        system: {key: mean(row[key] for row in rows) for key, _ in METRICS}
        for system, rows in rows_by_system.items()
        if rows
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = summarize(load_rows())
    systems = [system for system in ("Orcaforge", "vLLM") if system in summary]

    if len(systems) < 2:
        print("Need at least two real-inference systems containing 'transformers' and 'vllm'; no chart written.")
        return 0

    x_positions = range(len(METRICS))
    width = 0.36
    offsets = {
        systems[0]: -width / 2,
        systems[1]: width / 2,
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    for system in systems:
        values = [summary[system][key] for key, _ in METRICS]
        bars = ax.bar([x + offsets[system] for x in x_positions], values, width, label=system)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)

    ax.set_title("Real-Inference Smoke Comparison")
    ax.set_ylabel("Metric value")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([label for _, label in METRICS], rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=160)
    print(f"wrote: {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
