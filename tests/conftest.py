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
