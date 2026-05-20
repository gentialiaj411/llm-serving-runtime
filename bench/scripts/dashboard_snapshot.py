from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "bench" / "results"


def _find_latest_csv(results_dir: Path) -> Path:
    csv_paths = sorted(results_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_paths:
        raise SystemExit(f"No CSV files found in {results_dir}")
    return csv_paths[0]


def _to_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _load_csv_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"CSV has no rows: {path}")

    ttft_p95_values = [_to_float(r.get("ttft_ms_p95")) for r in rows]
    ttft_p50_values = [_to_float(r.get("ttft_ms_p50")) for r in rows]
    success_values = [_to_float(r.get("success_rate")) for r in rows]
    return {
        "rows": len(rows),
        "ttft_ms_p95_max": max(ttft_p95_values),
        "ttft_ms_p50_avg": sum(ttft_p50_values) / len(ttft_p50_values),
        "success_rate_min": min(success_values),
        "scenario_ids": sorted({r["scenario_id"] for r in rows if r.get("scenario_id")}),
    }


def _fetch_runtime_metrics(url: str | None) -> dict[str, Any] | None:
    if not url:
        return None
    try:
        r = httpx.get(url, timeout=3.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="Create a compact observability dashboard snapshot.")
    p.add_argument("--csv", default="", help="Path to benchmark CSV. Defaults to latest CSV in bench/results.")
    p.add_argument("--coordinator-metrics-url", default="", help="Optional coordinator /metrics URL.")
    p.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--output-name", default="dashboard-snapshot")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else _find_latest_csv(DEFAULT_RESULTS_DIR)
    csv_summary = _load_csv_summary(csv_path)
    runtime_metrics = _fetch_runtime_metrics(args.coordinator_metrics_url or None)

    snapshot = {
        "generated_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_csv": str(csv_path),
        "csv_summary": csv_summary,
        "runtime_metrics": runtime_metrics,
        "focus_metrics": {
            "ttft_ms_p95_max": csv_summary["ttft_ms_p95_max"],
            "current_inflight_requests": (runtime_metrics or {}).get("current_inflight_requests"),
            "retry_attempts_total": (runtime_metrics or {}).get("retry_attempts_total"),
            "cancellations_total": (runtime_metrics or {}).get("cancellations_total"),
        },
    }

    json_path = output_dir / f"{args.output_name}.json"
    md_path = output_dir / f"{args.output_name}.md"
    json_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    lines = [
        "# Dashboard Snapshot",
        "",
        f"- generated_at_utc: `{snapshot['generated_at_utc']}`",
        f"- source_csv: `{csv_path}`",
        f"- ttft_ms_p95_max: `{csv_summary['ttft_ms_p95_max']:.3f}`",
        f"- success_rate_min: `{csv_summary['success_rate_min']:.6f}`",
        f"- current_inflight_requests: `{snapshot['focus_metrics']['current_inflight_requests']}`",
        f"- retry_attempts_total: `{snapshot['focus_metrics']['retry_attempts_total']}`",
        f"- cancellations_total: `{snapshot['focus_metrics']['cancellations_total']}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote: {json_path}")
    print(f"wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
