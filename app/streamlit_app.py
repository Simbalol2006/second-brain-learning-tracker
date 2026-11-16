"""Second Brain web UI.

Run from the project folder:  uv run streamlit run app/streamlit_app.py

This file only draws widgets and calls the same functions as the CLI. All logic
lives in the `secondbrain` package. Navigation is a radio button in the sidebar;
the chosen page and the open item are kept in st.session_state.
"""

import logging
import sqlite3
from collections.abc import Callable

import graphviz
import streamlit as st
from pydantic import ValidationError

from secondbrain import analyze, db, items, library, links, profile, recommend, search, ui, vectors
from secondbrain.config import Settings
from secondbrain.embed import Embedder
from secondbrain.errors import SecondBrainError
from secondbrain.ingest import pipeline
from secondbrain.llm import LLMProvider, build_provider

PAGES = ["Add", "Library", "Item", "Search", "Graph", "Recommend", "Profile"]


# ---------- things loaded once per app session ----------


@st.cache_resource
def get_settings() -> Settings:
    return Settings()


@st.cache_resource(show_spinner="Loading the embedding model (first time takes longer)...")
def get_embedder() -> Embedder:
    return ui.load_embedder(get_settings())


@st.cache_resource
def get_provider() -> LLMProvider:
    return build_provider(get_settings())


def provider_or_warn() -> LLMProvider | None:
    """The LLM provider, or None plus a visible warning if it is not configured."""
    try:
        return get_provider()
    except SecondBrainError as exc:
        st.warning(f"No LLM available: {exc}")
        return None


def go_to(page: str, item_id: int | None = None) -> None:
    """Button callback: switch page (and open an item). Runs before the next redraw."""
    st.session_state["page"] = page
    if item_id is not None:
        st.session_state["current_item"] = item_id
        st.session_state["item_select"] = item_id  # the drop-down on the Item page


# ---------- pages ----------


def page_add(conn: sqlite3.Connection, settings: Settings) -> None:
    st.header("Add")
    text = st.text_area("YouTube or article URLs, one per line", key="add_urls", height=120)
    analyze_too = st.checkbox("Summarize and tag with the LLM", value=True, key="add_analyze")
    transcribe = st.checkbox(
        "Use local Whisper when a video has no captions (slow)", value=False, key="add_transcribe"
    )
    if not st.button("Save", type="primary", key="add_save"):
        return
    urls = [line.strip() for line in text.splitlines() if line.strip()]
    if not urls:
        st.warning("Paste at least one URL.")
        return
    for url in urls:
        _save_one(conn, settings, url, analyze_too, transcribe)


def _save_one(
    conn: sqlite3.Connection, settings: Settings, url: str, analyze_too: bool, transcribe: bool
) -> None:
    with st.status(f"Saving {url}", expanded=True) as box:
        try:
            st.write("Fetching, chunking and embedding...")
            result = pipeline.ingest_url(conn, settings, get_embedder, url, transcribe=transcribe)
            title = items.get(conn, result.item_id)["title"] or url
            if result.status == "no_transcript":
                st.warning(f"No transcript: {result.note} Open the item to transcribe it locally.")
                box.update(label=f"Saved without transcript: {title}", state="error")
                return
            if analyze_too and items.load_analysis(conn, result.item_id) is None:
                st.write("Analyzing with the LLM...")
                analyze.analyze_item(conn, settings, get_provider(), result.item_id)
            box.update(label=f"Saved: {title} (item {result.item_id})", state="complete")
        except SecondBrainError as exc:
            st.error(str(exc))
            box.update(label=f"Failed: {url}", state="error")


def page_library(conn: sqlite3.Connection, settings: Settings) -> None:
    st.header("Library")
    rows = ui.library_rows(conn)
    if not rows:
        st.info("Nothing saved yet. Go to **Add** and paste a URL.")
        return
    left, middle, right = st.columns(3)
    progress = left.multiselect("Progress", profile.PROGRESS_STATES, key="lib_progress")
    topic = middle.selectbox("Topic", [None, *ui.all_topics(rows)], key="lib_topic", format_func=lambda t: t or "all")
    text = right.text_input("Title or topic contains", key="lib_text")
    shown = ui.filter_rows(rows, progress, topic, text)
    st.caption(f"{len(shown)} of {len(rows)} items")
    st.dataframe(shown, hide_index=True, width="stretch")
    if shown:
        by_id = {row["id"]: row for row in shown}
        chosen = st.selectbox("Open an item", list(by_id), format_func=lambda i: ui.item_label(by_id[i]), key="lib_open")
        st.button("Open", key="lib_open_button", on_click=go_to, args=("Item", chosen))


def page_item(conn: sqlite3.Connection, settings: Settings) -> None:
    st.header("Item")
    rows = {int(row["id"]): row for row in items.list_all(conn)}
    if not rows:
        st.info("Nothing saved yet. Go to **Add** and paste a URL.")
        return
    # "current_item" survives page changes; the drop-down's own state does not.
    ids = list(rows)
    current_item = st.session_state.get("current_item")
    start = ids.index(current_item) if current_item in rows else 0
    item_id = st.selectbox("Item", ids, index=start, format_func=lambda i: ui.item_label(rows[i]), key="item_select")
    st.session_state["current_item"] = item_id
    item = rows[item_id]

    st.subheader(item["title"] or item["url"])
    st.link_button("Open source", item["url"])
    st.caption(
        f"{item['author'] or 'unknown author'} · {ui.format_minutes(item['duration_sec'])} · "
        f"transcript: {item['transcript_source']} · status: {item['ingest_status']}"
    )
    if item["error"]:
        (st.warning if item["ingest_status"] == "no_transcript" else st.error)(item["error"])

    current = item["progress_status"]
    chosen = st.radio(
        "Progress", profile.PROGRESS_STATES, index=profile.PROGRESS_STATES.index(current),
        horizontal=True, key=f"progress_{item_id}",
    )
    if chosen != current:
        profile.set_progress(conn, item_id, chosen)
        st.rerun()

    _item_actions(conn, settings, item_id)
    _item_analysis(conn, item_id)
    _item_related(conn, item_id)
    with st.expander("Delete this item"):
        sure = st.checkbox("Yes, delete it with its transcript, analysis and links", key=f"delete_sure_{item_id}")
        if st.button("Delete", key="item_delete", disabled=not sure):
            library.delete_item(conn, item_id)
            st.session_state.pop("current_item", None)
            st.session_state.pop("search_hits", None)  # may still list the deleted item
            st.session_state.pop("recs", None)
            st.rerun()


def _item_actions(conn: sqlite3.Connection, settings: Settings, item_id: int) -> None:
    has_transcript = items.load_segments(conn, item_id) is not None
    has_analysis = items.load_analysis(conn, item_id) is not None
    embedded = vectors.has_item_vector(conn, item_id)
    left, middle, right = st.columns(3)
    try:
        if has_transcript and left.button("Re-analyze" if has_analysis else "Analyze with the LLM", key="item_analyze"):
            with st.spinner("Analyzing..."):
                analyze.analyze_item(conn, settings, get_provider(), item_id, force=True)
            st.rerun()
        if not has_transcript and middle.button("Transcribe with Whisper (slow)", key="item_transcribe"):
            with st.spinner("Transcribing locally; this can take minutes..."):
                pipeline.process_item(conn, settings, get_embedder, item_id, transcribe=True)
            st.rerun()
        if not embedded and right.button("Retry saving", key="item_retry"):
            with st.spinner("Retrying..."):
                pipeline.process_item(conn, settings, get_embedder, item_id)
            st.rerun()
    except SecondBrainError as exc:
        st.error(str(exc))


def _item_analysis(conn: sqlite3.Connection, item_id: int) -> None:
    stored = analyze.stored_analysis(conn, item_id)
    if stored is None:
        st.info("Not analyzed yet.")
        return
    st.markdown(f"**Difficulty:** {stored.difficulty}")
    st.markdown(f"**Topics:** {', '.join(stored.topics)}")
    if stored.skills:
        st.markdown(f"**Skills:** {', '.join(stored.skills)}")
    st.markdown("**Summary**")
    st.write(stored.summary)
    st.markdown("**Key points**")
    st.markdown("\n".join(f"- {point}" for point in stored.key_points))


def _item_related(conn: sqlite3.Connection, item_id: int) -> None:
    st.markdown("**Related to this**")
    related = links.related(conn, item_id)
    if not related:
        st.caption("No related items above the link threshold.")
    for row in related:
        left, right = st.columns([5, 1])
        left.write(
            f"{row['title'] or row['url']} — score {row['score']:.2f} "
            f"(text {row['embed_sim']:.2f}, shared tags {row['tag_jaccard']:.2f})"
        )
        right.button("Open", key=f"related_{row['id']}", on_click=go_to, args=("Item", int(row["id"])))


def page_search(conn: sqlite3.Connection, settings: Settings) -> None:
    st.header("Search")
    with st.form("search_form"):
        query = st.text_input("What are you looking for?", key="search_query")
        limit = st.slider("Results", 1, 20, 5, key="search_limit")
        submitted = st.form_submit_button("Search", type="primary")
    if submitted:
        try:
            st.session_state["search_hits"] = search.search(conn, get_embedder(), query, limit=limit)
        except SecondBrainError as exc:
            st.session_state["search_hits"] = []
            st.error(str(exc))
            return
    hits = st.session_state.get("search_hits")
    if hits is None:
        return
    if not hits:
        st.info("No results.")
    for rank, hit in enumerate(hits, start=1):
        left, right = st.columns([5, 1])
        left.markdown(f"**{rank}. [{hit.title}]({hit.link})** · similarity {hit.score:.2f}")
        left.caption(hit.passage[:400] + ("..." if len(hit.passage) > 400 else ""))
        right.button("Open", key=f"hit_{hit.item_id}", on_click=go_to, args=("Item", hit.item_id))


def page_graph(conn: sqlite3.Connection, settings: Settings) -> None:
    st.header("Graph")
    rows = {int(row["id"]): row for row in items.list_all(conn)}
    left, middle, right = st.columns(3)
    max_links = left.slider("Strongest links per item", 1, 10, 5, key="graph_links")
    color_by = middle.radio("Colour by", ["progress", "difficulty"], horizontal=True, key="graph_color")
    focus = right.selectbox(
        "Focus on", [None, *rows], key="graph_focus",
        format_func=lambda i: "whole library" if i is None else ui.item_label(rows[i]),
    )
    graph = links.graph(conn, max_links)
    if focus is not None:
        graph = ui.neighbourhood(graph, focus)
    if not graph["nodes"]:
        st.info("Nothing to draw yet. Save a few items first.")
        return
    # A Source object (not a plain string) lets us pick the force-directed "neato" layout,
    # which suits a similarity graph better than the default top-down one.
    chart = graphviz.Source(ui.graph_dot(graph, color_by, focus), engine="neato")
    st.graphviz_chart(chart, width="stretch")
    st.caption(
        f"{len(graph['nodes'])} items, {len(graph['edges'])} links (threshold {settings.link_threshold}). "
        "✓ done · ▶ watching · thicker line = stronger link. Hover for full titles."
    )


def page_recommend(conn: sqlite3.Connection, settings: Settings) -> None:
    st.header("Recommend")
    use_llm = st.toggle("Let the LLM choose the order and write the reasons", value=True, key="rec_llm")
    limit = st.slider("Suggestions", 1, 10, 5, key="rec_limit")
    if st.button("What should I learn next?", type="primary", key="rec_go"):
        provider = provider_or_warn() if use_llm else None
        try:
            with st.spinner("Thinking..."):
                st.session_state["recs"] = recommend.recommend(conn, get_embedder, provider, limit=limit)
        except SecondBrainError as exc:
            st.error(str(exc))
            return
    result = st.session_state.get("recs")
    if result is None:
        return
    if result.warning:
        st.warning(result.warning)
    if not result.suggestions:
        st.info("Nothing to recommend: there are no analyzed, unfinished items.")
        return
    if not profile.goals(conn):
        st.caption("No goals set, so goal fit is ignored. Add one on the Profile page.")
    st.caption("Reasons written by the LLM." if result.by_llm else "Rule-based order and reasons.")
    for rank, suggestion in enumerate(result.suggestions, start=1):
        item = suggestion.candidate
        left, right = st.columns([5, 1])
        left.markdown(f"**{rank}. {item.title}** · {item.difficulty} · {ui.format_minutes(item.duration_sec)}")
        left.write(suggestion.reason)
        right.button("Open", key=f"rec_{item.item_id}", on_click=go_to, args=("Item", item.item_id))


def page_profile(conn: sqlite3.Connection, settings: Settings) -> None:
    st.header("Profile")
    counts = dict(conn.execute("SELECT progress_status, count(*) FROM items GROUP BY progress_status"))
    for column, state in zip(st.columns(3), profile.PROGRESS_STATES):
        column.metric(state.replace("_", " "), counts.get(state, 0))

    st.subheader("Goals")
    for goal in profile.goals(conn):
        left, right = st.columns([5, 1])
        left.write(goal["text"])
        if right.button("Remove", key=f"goal_remove_{goal['id']}"):
            profile.remove_goal(conn, int(goal["id"]))
            st.rerun()
    with st.form("goal_form", clear_on_submit=True):
        goal_text = st.text_input("New goal", key="goal_text")
        if st.form_submit_button("Add goal") and _attempt(lambda: profile.add_goal(conn, goal_text)):
            st.rerun()

    st.subheader("Skills")
    skill_rows = [
        {"skill": s["name"], "level": profile.LEVEL_NAMES[s["level"]], "set by": s["source"], "finished items": s["evidence_count"]}
        for s in profile.skills(conn)
    ]
    if skill_rows:
        st.dataframe(skill_rows, hide_index=True, width="stretch")
    else:
        st.caption("No skills yet. Mark items as done, or set a level below.")
    with st.form("skill_form", clear_on_submit=True):
        name = st.text_input("Skill or topic", key="skill_name")
        level = st.selectbox("Level", range(4), format_func=lambda n: profile.LEVEL_NAMES[n], key="skill_level")
        if st.form_submit_button("Set level") and _attempt(lambda: profile.set_skill(conn, name, level)):
            st.rerun()
    manual = [s["name"] for s in profile.skills(conn) if s["source"] == "manual"]
    if manual:
        target = st.selectbox("Remove a level you set by hand", manual, key="skill_remove")
        if st.button("Remove level", key="skill_remove_button") and _attempt(lambda: profile.remove_skill(conn, target)):
            st.rerun()


def _attempt(action: Callable[[], object]) -> bool:
    """Run an action; show its error instead of crashing. Returns True on success."""
    try:
        action()
    except SecondBrainError as exc:
        st.error(str(exc))
        return False
    return True


# ---------- entry point ----------

PAGE_FUNCTIONS: dict[str, Callable[[sqlite3.Connection, Settings], None]] = {
    "Add": page_add,
    "Library": page_library,
    "Item": page_item,
    "Search": page_search,
    "Graph": page_graph,
    "Recommend": page_recommend,
    "Profile": page_profile,
}


def main() -> None:
    st.set_page_config(page_title="Second Brain", layout="wide")
    try:
        settings = get_settings()
    except ValidationError as exc:
        st.error(f"Invalid configuration (check .env):\n\n{exc}")
        st.stop()
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        conn = db.connect(settings.db_path)
    except SecondBrainError as exc:
        st.error(str(exc))
        st.stop()
    try:
        st.sidebar.title("Second Brain")
        page = st.sidebar.radio("Page", PAGES, key="page")
        st.sidebar.caption(f"LLM: {settings.llm_provider} · {settings.llm_model_name}")
        st.sidebar.caption(f"{conn.execute('SELECT count(*) FROM items').fetchone()[0]} items saved")
        PAGE_FUNCTIONS[page](conn, settings)
    finally:
        conn.close()


main()
