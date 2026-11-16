"""The Streamlit app, run headless with Streamlit's AppTest (no browser, no network)."""

import sqlite3
from pathlib import Path

import pytest
import streamlit as st
from conftest import FakeEmbedder, FakeProvider, add_text_item, make_segments
from streamlit.testing.v1 import AppTest

from secondbrain import analyze, db, items, llm, profile, ui
from secondbrain.config import Settings
from secondbrain.errors import FetchError
from secondbrain.ingest import pipeline
from secondbrain.models import Metadata, Transcript

APP = str(Path(__file__).parent.parent / "app" / "streamlit_app.py")
ML = "gradient descent trains neural network weights by reducing loss"
BREAD = "sourdough bread needs flour water salt and slow fermentation"


def analysis_json(topic: str, difficulty: str) -> str:
    return (
        f'{{"summary": "A summary about {topic}.", "key_points": ["First point.", "Second point."], '
        f'"topics": ["{topic}"], "skills": [], "difficulty": "{difficulty}"}}'
    )


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(ui, "FastEmbedEmbedder", lambda model, cache_dir: FakeEmbedder())
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def seeded(tmp_path: Path) -> dict[str, int]:
    """Two analyzed items in the database the app will open."""
    settings = Settings(data_dir=tmp_path / "data", link_threshold=0.0, _env_file=None)
    conn = db.connect(settings.db_path)
    embedder = FakeEmbedder()
    ml = add_text_item(conn, settings, embedder, "a" * 11, "Neural nets", ML)
    bread = add_text_item(conn, settings, embedder, "b" * 11, "Baking bread", BREAD)
    analyze.analyze_item(conn, settings, FakeProvider([analysis_json("machine learning", "intermediate")]), ml)
    analyze.analyze_item(conn, settings, FakeProvider([analysis_json("baking", "beginner")]), bread)
    conn.close()
    return {"ml": ml, "bread": bread}


def open_page(page: str, **state: object) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=30)
    app.session_state["page"] = page
    for key, value in state.items():
        app.session_state[key] = value
    app.run()
    assert not app.exception, app.exception
    return app


def text_of(app: AppTest) -> str:
    """All visible text on the page, for simple 'is it shown' checks."""
    parts = [e.value for kind in ("markdown", "caption", "info", "warning", "error", "subheader", "header") for e in getattr(app, kind)]
    return "\n".join(str(p) for p in parts)


def database(tmp_path: Path) -> sqlite3.Connection:
    return db.connect(tmp_path / "data" / "secondbrain.db")


@pytest.mark.parametrize("page", ["Add", "Library", "Item", "Search", "Graph", "Recommend", "Profile"])
def test_every_page_renders_with_an_empty_database(page: str) -> None:
    app = open_page(page)
    assert app.sidebar.radio[0].value == page
    assert "0 items saved" in "\n".join(c.value for c in app.sidebar.caption)


@pytest.mark.parametrize("page", ["Add", "Library", "Item", "Search", "Graph", "Recommend", "Profile"])
def test_every_page_renders_with_data(page: str, seeded: dict[str, int]) -> None:
    app = open_page(page)
    assert "2 items saved" in "\n".join(c.value for c in app.sidebar.caption)


def test_sidebar_switches_pages(seeded: dict[str, int]) -> None:
    app = open_page("Add")
    app.sidebar.radio[0].set_value("Library").run()
    assert not app.exception
    assert app.header[0].value == "Library"
    assert "2 of 2 items" in text_of(app)


def test_library_filter_and_open(seeded: dict[str, int]) -> None:
    app = open_page("Library")
    app.text_input(key="lib_text").set_value("bread").run()
    assert "1 of 2 items" in text_of(app)
    app.button(key="lib_open_button").click().run()
    assert not app.exception
    assert app.header[0].value == "Item"
    assert app.subheader[0].value == "Baking bread"


def test_item_page_shows_analysis_and_related(seeded: dict[str, int]) -> None:
    app = open_page("Item", current_item=seeded["ml"])
    shown = text_of(app)
    assert app.subheader[0].value == "Neural nets"
    assert "A summary about machine learning." in shown
    assert "- First point." in shown
    assert "**Difficulty:** intermediate" in shown
    assert "Baking bread — score" in shown

    app.button(key=f"related_{seeded['bread']}").click().run()
    assert not app.exception
    assert app.subheader[0].value == "Baking bread"


def test_item_progress_change_updates_the_profile(seeded: dict[str, int], tmp_path: Path) -> None:
    app = open_page("Item", current_item=seeded["bread"])
    app.radio(key=f"progress_{seeded['bread']}").set_value("done").run()
    assert not app.exception
    conn = database(tmp_path)
    assert items.get(conn, seeded["bread"])["progress_status"] == "done"
    assert [s["name"] for s in profile.skills(conn)] == ["baking"]
    conn.close()


def test_item_reanalyze_uses_the_provider(seeded: dict[str, int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = FakeProvider([analysis_json("deep learning", "advanced")])
    monkeypatch.setattr(llm, "build_provider", lambda settings: provider)
    app = open_page("Item", current_item=seeded["ml"])
    app.button(key="item_analyze").click().run()
    assert not app.exception
    assert "**Difficulty:** advanced" in text_of(app)


def test_item_analyze_without_llm_shows_the_reason(seeded: dict[str, int]) -> None:
    app = open_page("Item", current_item=seeded["ml"])
    app.button(key="item_analyze").click().run()
    assert not app.exception
    assert "ANTHROPIC_API_KEY is not set" in text_of(app)


def test_search_page(seeded: dict[str, int]) -> None:
    app = open_page("Search")
    app.text_input(key="search_query").set_value("sourdough bread fermentation")
    app.button[0].click().run()  # the form's submit button
    assert not app.exception
    shown = text_of(app)
    assert "**1. [Baking bread](https://www.youtube.com/watch?v=bbbbbbbbbbb&t=" in shown
    assert "sourdough bread needs flour" in shown

    app.button(key=f"hit_{seeded['bread']}").click().run()
    assert app.header[0].value == "Item" and app.subheader[0].value == "Baking bread"


def test_search_with_empty_query_shows_an_error(seeded: dict[str, int]) -> None:
    app = open_page("Search")
    app.button[0].click().run()
    assert "query is empty" in text_of(app)


def test_graph_page(seeded: dict[str, int]) -> None:
    app = open_page("Graph")
    assert "2 items, 1 links" in text_of(app)
    app.selectbox(key="graph_focus").set_value(seeded["ml"]).run()
    assert not app.exception
    assert "2 items, 1 links" in text_of(app)


def test_recommend_rule_based_and_llm(seeded: dict[str, int], monkeypatch: pytest.MonkeyPatch) -> None:
    app = open_page("Recommend")
    app.toggle(key="rec_llm").set_value(False)
    app.button(key="rec_go").click().run()
    assert not app.exception
    shown = text_of(app)
    assert "Rule-based order and reasons." in shown
    assert "**1. Baking bread** · beginner · 10 min" in shown

    reply = f'{{"recommendations": [{{"item_id": {seeded["ml"]}, "reason": "Start with the theory."}}]}}'
    monkeypatch.setattr(llm, "build_provider", lambda settings: FakeProvider([reply]))
    st.cache_resource.clear()
    app = open_page("Recommend")
    app.button(key="rec_go").click().run()
    shown = text_of(app)
    assert "Reasons written by the LLM." in shown and "Start with the theory." in shown


def test_recommend_without_llm_warns_and_still_answers(seeded: dict[str, int]) -> None:
    app = open_page("Recommend")
    app.button(key="rec_go").click().run()
    assert not app.exception
    shown = text_of(app)
    assert "No LLM available: ANTHROPIC_API_KEY is not set" in shown
    assert "**1. Baking bread**" in shown


def test_profile_goal_and_skill(seeded: dict[str, int], tmp_path: Path) -> None:
    app = open_page("Profile")
    app.text_input(key="goal_text").set_value("bake better bread")
    app.button[0].click().run()  # "Add goal"
    assert not app.exception
    assert "bake better bread" in text_of(app)

    app.text_input(key="skill_name").set_value("Python")
    app.selectbox(key="skill_level").set_value(2)
    add_skill = [b for b in app.button if b.label == "Set level"][0]
    add_skill.click().run()
    assert not app.exception
    conn = database(tmp_path)
    assert [g["text"] for g in profile.goals(conn)] == ["bake better bread"]
    assert [(s["name"], s["level"], s["source"]) for s in profile.skills(conn)] == [("python", 2, "manual")]
    conn.close()


def test_add_page_saves_and_analyzes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline.youtube, "fetch_metadata", lambda video_id: Metadata(title="A talk", duration_sec=600))
    monkeypatch.setattr(pipeline.youtube, "fetch_captions", lambda video_id, languages: Transcript(make_segments(40), "auto", "en"))
    monkeypatch.setattr(llm, "build_provider", lambda settings: FakeProvider([analysis_json("testing", "beginner")]))
    app = open_page("Add")
    app.text_area(key="add_urls").set_value("https://youtu.be/dQw4w9WgXcQ")
    app.button(key="add_save").click().run()
    assert not app.exception
    conn = database(tmp_path)
    item = items.get(conn, 1)
    assert (item["title"], item["ingest_status"]) == ("A talk", "analyzed")
    conn.close()


def test_add_page_shows_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(video_id: str, languages: list[str]) -> Transcript:
        raise FetchError("YouTube blocked the caption request")

    monkeypatch.setattr(pipeline.youtube, "fetch_metadata", lambda video_id: Metadata(title="A talk"))
    monkeypatch.setattr(pipeline.youtube, "fetch_captions", blocked)
    app = open_page("Add")
    app.text_area(key="add_urls").set_value("https://youtu.be/dQw4w9WgXcQ\nnot a url")
    app.button(key="add_save").click().run()
    assert not app.exception
    shown = text_of(app)
    assert "YouTube blocked the caption request" in shown
    assert "Not a valid http(s) URL" in shown


def test_add_page_needs_a_url() -> None:
    app = open_page("Add")
    app.button(key="add_save").click().run()
    assert "Paste at least one URL." in text_of(app)


def test_invalid_configuration_is_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_WORDS", "0")
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert not app.exception
    assert "Invalid configuration" in text_of(app)


def test_item_delete_needs_confirmation(seeded: dict[str, int], tmp_path: Path) -> None:
    app = open_page("Item", current_item=seeded["ml"])
    assert app.button(key="item_delete").disabled
    app.checkbox(key=f"delete_sure_{seeded['ml']}").check().run()
    app.button(key="item_delete").click().run()
    assert not app.exception
    assert app.subheader[0].value == "Baking bread"  # the remaining item is shown
    conn = database(tmp_path)
    assert [int(r["id"]) for r in items.list_all(conn)] == [seeded["bread"]]
    conn.close()
