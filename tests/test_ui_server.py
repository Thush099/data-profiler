"""Tests for the data-quality spec-sheet UI server (ui/server.py)."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from ui.server import build_server, run_profile

SAMPLE_CSV = "id,name,score\n1,alice,0.9\n2,bob,\n3,,1.4\n4,carol,0.2\n"


@pytest.fixture()
def server():
    srv = build_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _post(base: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base}/api/profile",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request)
        return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_profile_endpoint_runs_real_pipeline(server):
    status, payload = _post(server, {"csv_text": SAMPLE_CSV})
    assert status == 200
    profile = payload["profile"]
    assert profile["row_count"] == 4
    assert profile["column_count"] == 3
    assert len(profile["columns"]) == 3
    assert payload["remediation_plan"]["total_actions"] >= 1


def test_contract_validation_reports_violations(server):
    contract = {
        "name": "demo",
        "rules": [{"column": "score", "not_null": True}],
    }
    status, payload = _post(
        server,
        {"csv_text": SAMPLE_CSV, "contract_json": json.dumps(contract)},
    )
    assert status == 200
    report = payload["contract_validation"]
    assert report is not None
    assert report["passed"] is False
    assert report["violations"]


def test_sample_and_health_and_index_served(server):
    assert json.loads(urllib.request.urlopen(server + "/api/health").read())["status"] == "ok"
    sample = json.loads(urllib.request.urlopen(server + "/api/sample").read())
    assert "csv_text" in sample and sample["csv_text"].strip()
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert "spec sheet" in html and "/api/profile" in html


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"csv_text": ""}, "csv_text must be a non-empty CSV string"),
        ({"csv_text": 'a,b\n"unclosed'}, "could not be parsed"),
        ({"csv_text": "a,b\n"}, "at least one data row"),
        (
            {"csv_text": SAMPLE_CSV, "contract_json": '{"rules": [{"column": "score", "not_null": true, "oops": 1}]}'},
            "Unknown ColumnRule field",
        ),
        ({"csv_text": SAMPLE_CSV, "contract_json": "{oops"}, "contract_json is not valid JSON"),
        ({"csv_text": SAMPLE_CSV, "target": "missing"}, "target column 'missing' not found"),
    ],
)
def test_invalid_inputs_rejected_with_clear_message(server, payload, message):
    status, body = _post(server, payload)
    assert status == 400
    assert message in body["error"]


def test_run_profile_with_contract_dict():
    result = run_profile(SAMPLE_CSV, target="score")
    assert result["profile"]["target_column"] == "score"
    assert "remediation_plan" in result

def test_index_has_accessibility_features(server):
    """The served page ships skip-link navigation, a main landmark, and reduced-motion guard."""
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert 'class="skip-link"' in html, "missing skip link for keyboard users"
    assert '<main class="layout">' in html, "missing main landmark"
    assert 'role="alert"' in html, "error box must announce to assistive tech"
    assert "prefers-reduced-motion" in html, "missing reduced-motion guard"
