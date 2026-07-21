"""Test harness for the single-file Streamlit app.

Design rationale: app.py keeps all UI behind a __name__-guarded main() and
builds the LangGraph lazily inside session state, so importing the module
executes only imports, constants, and definitions — no Streamlit runtime,
no API key, no network.  We load it via importlib (per CLAUDE.md) so the
suite is independent of packaging, and monkeypatch the cached loaders at
the module-attribute level: replacing the attribute swaps out the whole
@st.cache_data wrapper object, so the cache is never consulted and no
clearing is needed.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Repo root on sys.path so the Phase 4 core/ package resolves from tests.
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def app():
    """Import app.py once per session as an isolated module object."""
    spec = importlib.util.spec_from_file_location("app_under_test", ROOT / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["app_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def patch_loaders(monkeypatch, app, roster_df, assignments_df):
    """Single interception point for the data loaders.

    Every test that needs fixture data goes through here — when Phase 4
    moves the loaders into core/, only this helper changes.
    """
    monkeypatch.setattr(app, "load_roster", lambda: roster_df)
    monkeypatch.setattr(app, "load_assignments", lambda: assignments_df)


def make_mock_anthropic(captured: list, payloads: list):
    """Anthropic client on an httpx.MockTransport — no test touches the
    network; LLM configuration is asserted via the captured request bodies
    (the SDK's real serialization, exactly what would go on the wire).

    payloads entries, consumed in order (last repeats):
      str        → 200 response with one text block containing the string
      list       → 200 response with that literal content-block list
      int        → that HTTP status with an API-style error body
    """
    import json as _json

    import anthropic
    import httpx

    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.content)
        captured.append(body)
        idx = min(call_count["n"], len(payloads) - 1)
        call_count["n"] += 1
        payload = payloads[idx]
        if isinstance(payload, int):
            return httpx.Response(payload, json={
                "type": "error",
                "error": {"type": "api_error", "message": "mocked failure"},
            })
        content = (
            payload if isinstance(payload, list)
            else [{"type": "text", "text": payload}]
        )
        return httpx.Response(200, json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": body.get("model", "mock"),
            "content": content,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    return anthropic.Anthropic(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def patch_llm(monkeypatch, app, captured: list, payloads: list):
    """Route all three call helpers through a mocked-transport client."""
    client = make_mock_anthropic(captured, payloads)
    monkeypatch.setattr(app, "get_anthropic_client", lambda: client)
    return client
