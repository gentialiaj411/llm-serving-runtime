from __future__ import annotations

import json
from pathlib import Path

from jsonschema import ValidationError, validate


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "bench" / "schemas" / "manifest.schema.json"
RESULTS_DIR = ROOT / "bench" / "results"


def main() -> int:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    manifests = sorted(RESULTS_DIR.glob("*.manifest.json"))
    if not manifests:
        raise SystemExit(f"No manifest files found in {RESULTS_DIR}")

    failures: list[str] = []
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        try:
            validate(instance=payload, schema=schema)
        except ValidationError as exc:
            failures.append(f"{path}: {exc.message}")

    if failures:
        print("Manifest validation failed:")
        for item in failures:
            print(f"- {item}")
        return 1

    print(f"Validated {len(manifests)} manifest files against {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
