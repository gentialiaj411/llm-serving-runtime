#!/usr/bin/env python3
"""Patch prometheus-fastapi-instrumentator route walking for wrapped routers.

vLLM 0.21.0 currently pulls a FastAPI/Starlette stack where the app route list can
contain wrapper objects without a `.path` attribute. Older instrumentator routing
code assumes every matched route exposes `.path`, which crashes `/health` and other
HTTP endpoints before the benchmark can start.

This patch is intentionally narrow and idempotent. It updates the installed
`prometheus_fastapi_instrumentator.routing` helper to use `getattr(..., "path", None)`
and continue recursing into nested routes when a wrapper object matches.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


OLD = """def _get_route_name(
    scope: Scope, routes: List[Route], route_name: Optional[str] = None
) -> Optional[str]:
    \"\"\"Gets route name for given scope taking mounts into account.\"\"\"

    for route in routes:
        match, child_scope = route.matches(scope)
        if match == Match.FULL:
            route_name = route.path
            child_scope = {**scope, **child_scope}
            if isinstance(route, Mount) and route.routes:
                child_route_name = _get_route_name(child_scope, route.routes, route_name)
                if child_route_name is None:
                    route_name = None
                else:
                    route_name += child_route_name
            return route_name
        elif match == Match.PARTIAL and route_name is None:
            route_name = route.path
    return None
"""

NEW = """def _get_route_name(
    scope: Scope, routes: List[Route], route_name: Optional[str] = None
) -> Optional[str]:
    \"\"\"Gets route name for given scope taking mounts into account.\"\"\"

    for route in routes:
        match, child_scope = route.matches(scope)
        route_path = getattr(route, \"path\", None)
        nested_routes = getattr(route, \"routes\", None)
        if match == Match.FULL:
            next_route_name = route_path if route_path is not None else route_name
            child_scope = {**scope, **child_scope}
            if nested_routes:
                child_route_name = _get_route_name(
                    child_scope, nested_routes, next_route_name
                )
                if child_route_name is not None:
                    return child_route_name
            return next_route_name
        elif match == Match.PARTIAL and route_name is None and route_path is not None:
            route_name = route_path
    return None
"""


def main() -> int:
    spec = importlib.util.find_spec("prometheus_fastapi_instrumentator.routing")
    if spec is None or spec.origin is None:
        print("prometheus_fastapi_instrumentator.routing not found", file=sys.stderr)
        return 1

    path = Path(spec.origin)
    text = path.read_text(encoding="utf-8")
    if NEW in text:
        print(f"already patched: {path}")
        return 0
    if OLD not in text:
        print(f"unexpected routing.py contents: {path}", file=sys.stderr)
        return 1

    path.write_text(text.replace(OLD, NEW), encoding="utf-8")
    print(f"patched: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
