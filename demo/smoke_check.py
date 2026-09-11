"""In-process offline smoke check for the demo server (no external process).

Usage: python demo/smoke_check.py [port]
Starts DemoHTTPServer on 127.0.0.1:<port> in a daemon thread, hits the
key offline endpoints, prints PASS/FAIL lines, then shuts down. Exit 0
only if every check passes. Never enables online mode.
"""
from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.server import DemoApp, DemoHTTPServer  # noqa: E402


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = pick a free port
    app = DemoApp(online=False)
    httpd = DemoHTTPServer(("127.0.0.1", port), app)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    print(f"[smoke] serving on {base} (offline mode)")

    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        print(f"[smoke] {'PASS' if ok else 'FAIL'} {name} {detail}")
        if not ok:
            failures += 1

    # 1. index HTML
    with urllib.request.urlopen(base + "/", timeout=5) as r:
        body = r.read().decode("utf-8")
        check("GET / html", r.status == 200 and "<html" in body.lower(),
              f"({len(body)} bytes)")

    # 2. /api/health offline semantics
    with urllib.request.urlopen(base + "/api/health", timeout=5) as r:
        health = json.loads(r.read().decode("utf-8"))
    check("GET /api/health", health.get("online_mode") is False
          and health.get("judge_online") is False,
          json.dumps(health, ensure_ascii=False))

    # 3. /api/data merged aggregates
    with urllib.request.urlopen(base + "/api/data", timeout=10) as r:
        data = json.loads(r.read().decode("utf-8"))
    keys = sorted(data.keys())
    ok = all(k in data for k in ("main", "c1", "ablation", "tasks_meta", "sources"))
    check("GET /api/data schema", ok, f"keys={keys}")
    tm = data.get("tasks_meta", {})
    check("tasks_meta counts",
          tm.get("n_main") == 24 and tm.get("n_probes") == 5
          and tm.get("n_total") == 29,
          f"main={tm.get('n_main')} probes={tm.get('n_probes')}")

    # 4. /api/tasks
    with urllib.request.urlopen(base + "/api/tasks", timeout=5) as r:
        tasks = json.loads(r.read().decode("utf-8"))
    main_tasks = tasks.get("tasks", []) if isinstance(tasks, dict) else tasks
    check("GET /api/tasks", len(main_tasks) == 24
          and all("id" in t and "message" in t for t in main_tasks),
          f"main tasks={len(main_tasks)} probes={len(tasks.get('probes', [])) if isinstance(tasks, dict) else '?'}")

    # 5. chart serving + traversal protection
    import pathlib
    charts = sorted(pathlib.Path("eval/charts").glob("*.svg"))
    name = charts[0].name if charts else "missing.svg"
    with urllib.request.urlopen(base + f"/api/charts/main/{name}", timeout=5) as r:
        svg = r.read().decode("utf-8")
    check(f"GET /api/charts/main/{name}",
          r.status == 200 and svg.lstrip().startswith("<?xml"))
    for bad in ("..%2f..%2fREADME.md", "..%5c..%5cREADME.md", "../../harness/run.py"):
        try:
            urllib.request.urlopen(base + f"/api/charts/main/{bad}", timeout=5)
            check(f"traversal blocked {bad!r}", False, "200 returned")
        except urllib.error.HTTPError as e:
            check(f"traversal blocked {bad!r}", e.code in (400, 403, 404),
                  f"status={e.code}")

    # 6. offline /api/decide → 503 structured JSON
    req = urllib.request.Request(
        base + "/api/decide",
        data=json.dumps({"arm": "C0", "message": "hi"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=5)
        check("POST /api/decide offline 503", False, "no error raised")
    except urllib.error.HTTPError as e:
        payload = json.loads(e.read().decode("utf-8"))
        check("POST /api/decide offline 503",
              e.code == 503 and payload.get("fallback") == "offline",
              f"status={e.code} error={payload.get('error')}")

    # 7. diagrams serving
    with urllib.request.urlopen(base + "/api/charts/diagrams/architecture.svg",
                                timeout=5) as r:
        check("GET /api/charts/diagrams/architecture.svg",
              r.status == 200)

    httpd.shutdown()
    print(f"[smoke] {'ALL PASS' if failures == 0 else f'{failures} FAILURES'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
