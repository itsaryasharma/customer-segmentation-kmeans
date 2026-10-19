"""Headless tests of the Streamlit app using streamlit.testing.AppTest."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _run():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def _text(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_app_renders_all_tabs():
    at = _run()
    assert len(at.tabs) == 5
    assert "Champions" in _text(at)


def test_scoring_presets():
    at = _run()
    at.button(key="preset_0").click().run()           # big recent spender
    assert not at.exception
    assert "Assigned segment" in _text(at) and "Champions" in _text(at)
    at.button(key="preset_3").click().run()           # one-off, long ago
    assert "Lapsed low-value" in _text(at)


def test_k_explorer():
    at = _run()
    at.slider(key="k_explore").set_value(6).run()
    assert not at.exception
    assert "Exploratory view only" in " ".join(c.value for c in at.caption)
