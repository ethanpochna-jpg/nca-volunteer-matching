"""Test harness for the app across the Phase 4 modularization.

Design rationale: Phases 0–3 wrote every test against a single-module
surface (`app.run_matching`, `app.call_likert_item`, …).  Phase 4 moves
code verbatim into core/* with the UNCHANGED suite staying green as the
acceptance criterion — so the `app` fixture now returns an AppFacade that
aggregates app.py plus every core module:

  - attribute reads resolve app.py first, then core modules;
  - attribute writes (monkeypatch.setattr) land on EVERY module that owns
    the name, so a patched symbol reaches its consumers wherever the
    split placed them, and monkeypatch's teardown restores the shared
    original through the same route.

Loader and LLM patching still flow through the two helpers below — the
single interception points promised in Phase 0.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Repo root on sys.path so the core/ package resolves from tests.
sys.path.insert(0, str(ROOT))

# Modules are aggregated in this order; app.py stays first so __file__,
# UI functions, and any still-unmoved symbol resolve there.
_CORE_MODULE_NAMES = (
    "schemas", "policy", "matching", "llm",
    "scoring", "reasoning", "records", "graph",
)


class AppFacade:
    """Aggregate attribute view over app.py + core/* (see module docstring)."""

    def __init__(self, modules):
        object.__setattr__(self, "_modules", tuple(modules))

    def __getattr__(self, name):
        for mod in object.__getattribute__(self, "_modules"):
            try:
                return getattr(mod, name)
            except AttributeError:
                continue
        raise AttributeError(name)

    def __setattr__(self, name, value):
        modules = object.__getattribute__(self, "_modules")
        owners = [m for m in modules if hasattr(m, name)]
        if not owners:
            owners = [modules[0]]
        for mod in owners:
            setattr(mod, name, value)


@pytest.fixture(scope="session")
def app():
    """Import app.py once per session; wrap it with the core aggregate."""
    spec = importlib.util.spec_from_file_location("app_under_test", ROOT / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["app_under_test"] = mod
    spec.loader.exec_module(mod)

    modules = [mod]
    for name in _CORE_MODULE_NAMES:
        try:
            modules.append(importlib.import_module(f"core.{name}"))
        except ImportError:
            continue          # module not split out yet — facade grows with Phase 4
    return AppFacade(modules)


def patch_loaders(monkeypatch, app, roster_df, assignments_df):
    """Single interception point for the data loaders."""
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
