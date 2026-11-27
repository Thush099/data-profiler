"""Lightweight local web UI server for the data-quality profiler.

Serves the spec-sheet workbench in ``ui/index.html`` and exposes a JSON API that
runs the real ``DataQualityPipeline``. Standard library only. Start with
``python ui/server.py`` and open the printed URL.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import Config  # noqa: E402
from src.models import DataContract  # noqa: E402
from src.pipeline import DataQualityPipeline  # noqa: E402

INDEX_HTML = (REPO_ROOT / "ui" / "index.html").read_text(encoding="utf-8")

MAX_CSV_BYTES = 2_000_000


def run_profile(
    csv_text: str,
    contract_json: Optional[str] = None,
    target: Optional[str] = None,
    execute_remediation: bool = False,
) -> dict[str, Any]:
    """Profile a CSV blob and return the pipeline report as a dictionary.

    Raises ValueError with a clear message for invalid inputs, matching the
    CLI's boundary validation.
    """
    import pandas as pd

    if not isinstance(csv_text, str) or not csv_text.strip():
        raise ValueError("csv_text must be a non-empty CSV string")
    if len(csv_text.encode("utf-8")) > MAX_CSV_BYTES:
        raise ValueError(f"csv_text exceeds {MAX_CSV_BYTES} bytes")

    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:  # pandas raises several parse error types
        raise ValueError(f"csv_text could not be parsed: {exc}") from exc
    if df.empty:
        raise ValueError("csv_text must contain at least one data row")

    contract = None
    if contract_json:
        try:
            payload = json.loads(contract_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"contract_json is not valid JSON: {exc}") from exc
        contract = DataContract.from_dict(payload)

    if target is not None and target not in df.columns:
        raise ValueError(f"target column {target!r} not found in CSV")

    pipeline = DataQualityPipeline(config=Config())
    result = pipeline.run(
        df=df,
        contract=contract,
        target=target,
        execute_remediation=execute_remediation,
    )
    return result.to_dict()


MAX_REQUEST_BYTES = 4_000_000


def build_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Create (but do not start) the UI HTTP server bound to ``host:port``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/health":
                self._send_json({"status": "ok"})
            elif self.path == "/api/sample":
                sample = (REPO_ROOT / "data" / "sample_dirty.csv").read_text(encoding="utf-8")
                self._send_json({"csv_text": sample})
            else:
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)

        def do_POST(self) -> None:
            if self.path != "/api/profile":
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > MAX_REQUEST_BYTES:
                    self._send_json(
                        {"error": f"request body exceeds {MAX_REQUEST_BYTES} bytes"},
                        status=413,
                    )
                    return
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                result = run_profile(
                    csv_text=payload.get("csv_text", ""),
                    contract_json=payload.get("contract_json"),
                    target=payload.get("target"),
                    execute_remediation=bool(payload.get("execute_remediation", False)),
                )
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the data-quality spec-sheet UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8124)
    args = parser.parse_args(argv)

    server = build_server(args.host, args.port)
    print(f"Data-quality workbench running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())