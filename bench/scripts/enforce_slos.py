from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SLO = ROOT / "bench" / "slo" / "scenario_slos.yaml"


def _to_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _load_slo(path: Path) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = payload.get("defaults") or {}
    scenarios = payload.get("scenarios") or {}
    return defaults, scenarios


def _merge_thresholds(defaults: dict[str, float], scenario_cfg: dict[str, float]) -> dict[str, float]:
    merged = dict(defaults)
    merged.update(scenario_cfg)
    return merged


def _check_row(row: dict[str, str], thresholds: dict[str, float]) -> list[str]:
    failures: list[str] = []
    sid = row.get("scenario_id", "<unknown>")

    success_rate = _to_float(row.get("success_rate"))
    timeout_rate = _to_float(row.get("timeout_rate"))
    http_error_rate = _to_float(row.get("http_error_rate"))
    ttft_ms_p95 = _to_float(row.get("ttft_ms_p95"))
    latency_ms_p95 = _to_float(row.get("latency_ms_p95"))

    if "success_rate_min" in thresholds and success_rate < thresholds["success_rate_min"]:
        failures.append(f"{sid}: success_rate {success_rate:.6f} < {thresholds['success_rate_min']:.6f}")
    if "timeout_rate_max" in thresholds and timeout_rate > thresholds["timeout_rate_max"]:
        failures.append(f"{sid}: timeout_rate {timeout_rate:.6f} > {thresholds['timeout_rate_max']:.6f}")
    if "http_error_rate_max" in thresholds and http_error_rate > thresholds["http_error_rate_max"]:
        failures.append(f"{sid}: http_error_rate {http_error_rate:.6f} > {thresholds['http_error_rate_max']:.6f}")
    if "ttft_ms_p95_max" in thresholds and ttft_ms_p95 > thresholds["ttft_ms_p95_max"]:
        failures.append(f"{sid}: ttft_ms_p95 {ttft_ms_p95:.3f} > {thresholds['ttft_ms_p95_max']:.3f}")
    if "latency_ms_p95_max" in thresholds and latency_ms_p95 > thresholds["latency_ms_p95_max"]:
        failures.append(f"{sid}: latency_ms_p95 {latency_ms_p95:.3f} > {thresholds['latency_ms_p95_max']:.3f}")
    return failures


def main() -> int:
    p = argparse.ArgumentParser(description="Enforce per-scenario performance SLOs on benchmark CSV output.")
    p.add_argument("--csv", required=True, help="Benchmark CSV path.")
    p.add_argument("--slo-file", default=str(DEFAULT_SLO), help="Scenario SLO policy file.")
    args = p.parse_args()

    csv_path = Path(args.csv)
    slo_path = Path(args.slo_file)
    defaults, scenarios = _load_slo(slo_path)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"CSV has no rows: {csv_path}")

    failures: list[str] = []
    checked = 0
    for row in rows:
        sid = row.get("scenario_id", "")
        thresholds = _merge_thresholds(defaults, scenarios.get(sid, {}))
        failures.extend(_check_row(row, thresholds))
        checked += 1

    if failures:
        print(f"SLO enforcement failed for {csv_path}:")
        for msg in failures:
            print(f"- {msg}")
        return 1

    print(f"SLO enforcement passed for {csv_path} ({checked} rows checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
