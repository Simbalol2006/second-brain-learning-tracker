"""Command line interface. Run `sb --help`."""

import json
import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError

from secondbrain import analyze as analysis
from secondbrain import db, evalset, items, library, links, profile
from secondbrain import recommend as recommending
from secondbrain import search as searching
from secondbrain.config import Settings
from secondbrain.embed import Embedder, FastEmbedEmbedder
from secondbrain.errors import SecondBrainError
from secondbrain.ingest import pipeline
from secondbrain.llm import LLMProvider, build_provider

app = typer.Typer(help="Second Brain: save, search and plan your learning.", no_args_is_help=True)
profile_app = typer.Typer(help="Your goals and skill levels.", no_args_is_help=True)
app.add_typer(profile_app, name="profile")
logger = logging.getLogger(__name__)


def _open() -> tuple[Settings, sqlite3.Connection]:
    """Load settings, set up logging and open the database. Exits with a clear message on error."""
    try:
        settings = Settings()
    except ValidationError as exc:
        _fail(f"Invalid configuration (check .env):\n{exc}")
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return settings, db.connect(settings.db_path)
    except SecondBrainError as exc:
        _fail(str(exc))


def _fail(message: str) -> NoReturn:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _embedder_factory(settings: Settings) -> pipeline.EmbedderFactory:
    """Load the embedding model only when first needed, then reuse it."""
    cache: list[Embedder] = []

    def get() -> Embedder:
        if not cache:
            cache.append(FastEmbedEmbedder(settings.embedding_model, settings.model_cache_dir))
        return cache[0]

    return get


def _provider_factory(settings: Settings) -> Callable[[], LLMProvider]:
    """Build the LLM provider only when first needed, then reuse it."""
    cache: list[LLMProvider] = []

    def get() -> LLMProvider:
        if not cache:
            cache.append(build_provider(settings))
        return cache[0]

    return get


def _analyze_and_report(
    conn: sqlite3.Connection, settings: Settings, provider: LLMProvider, item_id: int, force: bool
) -> None:
    result = analysis.analyze_item(conn, settings, provider, item_id, force=force)
    usage = items.llm_usage(conn, item_id)
    typer.echo(
        f"item {item_id}: analyzed - {result.difficulty}, topics: {', '.join(result.topics)} "
        f"[all LLM calls so far: {usage['calls']}, tokens in/out: "
        f"{usage['input_tokens']}/{usage['output_tokens']}]"
    )


def _read_urls(url: str | None, file: Path | None) -> list[str]:
    urls = [url] if url else []
    if file:
        if not file.exists():
            _fail(f"File not found: {file}")
        lines = file.read_text(encoding="utf-8").splitlines()
        urls += [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    if not urls:
        _fail("Give a URL or --file with one URL per line.")
    return urls


def _describe(result: pipeline.IngestResult) -> str:
    text = f"item {result.item_id}: {result.status}"
    if not result.created:
        text += " (already saved)"
    if result.status == "no_transcript":
        text += f" - {result.note} Run: sb transcribe {result.item_id}"
    return text


@app.command()
def ingest(
    url: Annotated[str | None, typer.Argument(help="YouTube video or article URL.")] = None,
    file: Annotated[Path | None, typer.Option(help="Text file with one URL per line.")] = None,
    transcribe: Annotated[
        bool, typer.Option(help="Use local Whisper when a video has no captions.")
    ] = False,
    analyze: Annotated[
        bool, typer.Option(help="Also run the LLM analysis (summary, tags, difficulty).")
    ] = False,
) -> None:
    """Save one or more URLs: fetch, chunk and embed."""
    settings, conn = _open()
    get_embedder = _embedder_factory(settings)
    get_provider = _provider_factory(settings)
    failures = 0
    for one_url in _read_urls(url, file):
        try:
            result = pipeline.ingest_url(conn, settings, get_embedder, one_url, transcribe=transcribe)
            typer.echo(f"{one_url} -> {_describe(result)}")
            if analyze and result.status == "embedded":
                _analyze_and_report(conn, settings, get_provider(), result.item_id, force=False)
        except SecondBrainError as exc:
            failures += 1
            typer.secho(f"{one_url} -> FAILED: {exc}", fg=typer.colors.RED, err=True)
    if failures:
        raise typer.Exit(code=1)


@app.command()
def transcribe(item_id: Annotated[int, typer.Argument(help="Id of an item without transcript.")]) -> None:
    """Transcribe a saved video locally with Whisper, then chunk and embed it."""
    settings, conn = _open()
    try:
        status = pipeline.process_item(
            conn, settings, _embedder_factory(settings), item_id, transcribe=True
        )
    except SecondBrainError as exc:
        _fail(str(exc))
    typer.echo(f"item {item_id}: {status}")


@app.command(name="analyze")
def analyze_items(
    item_id: Annotated[int | None, typer.Argument(help="Item id; omit when using --all.")] = None,
    all_pending: Annotated[
        bool, typer.Option("--all", help="Analyze every item that has no analysis yet.")
    ] = False,
    force: Annotated[bool, typer.Option(help="Redo the analysis even if one exists.")] = False,
) -> None:
    """Summarize and tag items with the configured LLM."""
    settings, conn = _open()
    if (item_id is None) == (not all_pending):
        _fail("Give an item id, or --all (not both).")
    targets = items.ids_without_analysis(conn) if all_pending else [item_id]
    if not targets:
        typer.echo("Nothing to analyze: every item with a transcript already has an analysis.")
        return
    try:
        provider = build_provider(settings)
    except SecondBrainError as exc:
        _fail(str(exc))
    failures = 0
    for target in targets:
        try:
            _analyze_and_report(conn, settings, provider, int(target), force)  # type: ignore[arg-type]
        except SecondBrainError as exc:
            failures += 1
            typer.secho(f"FAILED: {exc}", fg=typer.colors.RED, err=True)
    if failures:
        raise typer.Exit(code=1)


@app.command(name="list")
def list_items(
    status: Annotated[str | None, typer.Option(help="Only show this ingest status.")] = None,
) -> None:
    """List saved items."""
    _, conn = _open()
    rows = items.list_all(conn, status)
    if not rows:
        typer.echo("No items yet. Add one with: sb ingest <url>")
        return
    typer.echo(f"{'id':>4}  {'status':<13}  {'progress':<8}  {'source':<8}  {'min':>4}  title")
    for row in rows:
        minutes = "-" if row["duration_sec"] is None else str(round(row["duration_sec"] / 60))
        title = row["title"] or row["url"]
        typer.echo(
            f"{row['id']:>4}  {row['ingest_status']:<13}  {row['progress_status']:<8}  "
            f"{row['transcript_source']:<8}  {minutes:>4}  {title}"
        )


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What you are looking for, in your own words.")],
    limit: Annotated[int, typer.Option(help="How many items to show.")] = 5,
    mode: Annotated[
        str | None,
        typer.Option(help="dense, bm25 or hybrid. Default: SEARCH_MODE from .env (hybrid)."),
    ] = None,
) -> None:
    """Search everything saved: by meaning (dense), by keywords (bm25), or both (hybrid)."""
    settings, conn = _open()
    mode = mode or settings.search_mode
    try:
        hits = searching.search(
            conn, _embedder_factory(settings)(), query, limit=limit, mode=mode,  # type: ignore[arg-type]
            rrf_k=settings.rrf_k, depth=settings.search_depth,
        )
    except SecondBrainError as exc:
        _fail(str(exc))
    if not hits:
        typer.echo("No results. Save something first with: sb ingest <url>")
        return
    for rank, hit in enumerate(hits, start=1):
        typer.echo(f"{rank}. [{searching.describe(hit, mode)}] {hit.title} (item {hit.item_id})")
        typer.echo(f"   {hit.link}")
        typer.echo(f"   {hit.passage[:240]}{'...' if len(hit.passage) > 240 else ''}")


@app.command()
def related(
    item_id: Annotated[int, typer.Argument(help="Item id from 'sb list'.")],
    limit: Annotated[int, typer.Option(help="How many related items to show.")] = 5,
) -> None:
    """Show the items linked to one item, strongest link first."""
    _, conn = _open()
    try:
        rows = links.related(conn, item_id, limit)
    except SecondBrainError as exc:
        _fail(str(exc))
    if not rows:
        typer.echo(f"Item {item_id} has no links above the threshold.")
        return
    for row in rows:
        typer.echo(
            f"[{row['score']:.2f}] {row['title'] or row['url']} (item {row['id']}) "
            f"- text similarity {row['embed_sim']:.2f}, shared tags {row['tag_jaccard']:.2f}"
        )


@app.command(name="links")
def links_command(
    rebuild: Annotated[
        bool, typer.Option(help="Recompute every link (after changing LINK_* settings).")
    ] = False,
    stats: Annotated[
        bool, typer.Option(help="Show the score distribution over all item pairs.")
    ] = False,
) -> None:
    """Maintain item links: rebuild them, or inspect scores to choose LINK_THRESHOLD."""
    settings, conn = _open()
    if not rebuild and not stats:
        _fail("Give --rebuild and/or --stats.")
    if rebuild:
        typer.echo(f"{links.rebuild(conn, settings)} links stored (threshold {settings.link_threshold}).")
    if stats:
        for key, value in links.score_stats(conn, settings).items():
            typer.echo(f"{key:<24} {value:.3f}" if isinstance(value, float) else f"{key:<24} {value}")


@app.command(name="graph")
def graph_command(
    out: Annotated[Path | None, typer.Option(help="Write JSON here instead of printing it.")] = None,
    max_links: Annotated[int, typer.Option(help="Strongest links kept per item.")] = 5,
) -> None:
    """Export the item graph (nodes and edges) as JSON."""
    _, conn = _open()
    data = json.dumps(links.graph(conn, max_links), indent=2, ensure_ascii=False)
    if out is None:
        typer.echo(data)
        return
    out.write_text(data, encoding="utf-8")
    typer.echo(f"Wrote {out}")


@app.command(name="eval-templates")
def eval_templates(
    out: Annotated[Path, typer.Option(help="Folder for the label files.")] = Path("eval"),
    force: Annotated[bool, typer.Option(help="Overwrite existing label files.")] = False,
) -> None:
    """Create the label files for the evaluation from your saved items."""
    settings, conn = _open()
    try:
        written = evalset.write_templates(conn, settings, out, force=force)
    except SecondBrainError as exc:
        _fail(str(exc))
    for path in written:
        typer.echo(f"Wrote {path}")
    typer.echo("Next: fill them in as described in eval/README.md.")


@app.command()
def delete(
    item_id: Annotated[int, typer.Argument(help="Item id from 'sb list'.")],
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask for confirmation.")] = False,
) -> None:
    """Delete an item with its transcript, analysis, vectors and links."""
    _, conn = _open()
    try:
        title = items.get(conn, item_id)["title"] or f"item {item_id}"
        if not yes:
            typer.confirm(f"Delete '{title}'? This cannot be undone", abort=True)
        library.delete_item(conn, item_id)
    except SecondBrainError as exc:
        _fail(str(exc))
    typer.echo(f"item {item_id} deleted")


@app.command()
def status(
    item_id: Annotated[int, typer.Argument(help="Item id from 'sb list'.")],
    new_status: Annotated[str, typer.Argument(help="to_watch, watching or done.")],
) -> None:
    """Set your progress on an item. Marking it done updates your skill levels."""
    _, conn = _open()
    try:
        profile.set_progress(conn, item_id, new_status)
    except SecondBrainError as exc:
        _fail(str(exc))
    typer.echo(f"item {item_id}: {new_status}")


@app.command()
def recommend(
    limit: Annotated[int, typer.Option(help="How many suggestions.")] = 5,
    no_llm: Annotated[
        bool, typer.Option("--no-llm", help="Rule-based order and reasons only; no LLM call.")
    ] = False,
) -> None:
    """Suggest what to learn next, in order, with a one-line reason each."""
    settings, conn = _open()
    provider: LLMProvider | None = None
    if not no_llm:
        try:
            provider = build_provider(settings)
        except SecondBrainError as exc:
            typer.secho(f"Warning: no LLM available ({exc}) Using rule-based reasons.", fg=typer.colors.YELLOW, err=True)
    try:
        result = recommending.recommend(conn, _embedder_factory(settings), provider, limit=limit)
    except SecondBrainError as exc:
        _fail(str(exc))
    if result.warning:
        typer.secho(f"Warning: {result.warning}", fg=typer.colors.YELLOW, err=True)
    if not result.suggestions:
        typer.echo("Nothing to recommend: no analyzed, unfinished items. Try: sb analyze --all")
        return
    if not profile.goals(conn):
        typer.echo('Tip: no goals set, so goal fit is ignored. Add one: sb profile add-goal "..."')
    for rank, suggestion in enumerate(result.suggestions, start=1):
        item = suggestion.candidate
        minutes = "?" if item.duration_sec is None else str(round(item.duration_sec / 60))
        typer.echo(f"{rank}. {item.title} (item {item.item_id}) - {item.difficulty}, {minutes} min")
        typer.echo(f"   {suggestion.reason}")


@profile_app.command("show")
def profile_show() -> None:
    """Show goals, skill levels and progress counts."""
    _, conn = _open()
    typer.echo("Goals:")
    for goal in profile.goals(conn):
        typer.echo(f"  {goal['id']:>3}  {goal['text']}")
    typer.echo("Skills:")
    for skill in profile.skills(conn):
        typer.echo(
            f"  {skill['name']:<30} {profile.LEVEL_NAMES[skill['level']]:<13} "
            f"({skill['source']}, {skill['evidence_count']} finished)"
        )
    counts = dict(conn.execute("SELECT progress_status, count(*) FROM items GROUP BY progress_status"))
    typer.echo("Progress: " + ", ".join(f"{counts.get(s, 0)} {s}" for s in profile.PROGRESS_STATES))


@profile_app.command("add-goal")
def profile_add_goal(text: Annotated[str, typer.Argument(help="What you want to learn or become.")]) -> None:
    """Add a learning goal. Recommendations favour items close to your goals."""
    _, conn = _open()
    try:
        goal_id = profile.add_goal(conn, text)
    except SecondBrainError as exc:
        _fail(str(exc))
    typer.echo(f"goal {goal_id} added")


@profile_app.command("remove-goal")
def profile_remove_goal(goal_id: Annotated[int, typer.Argument(help="Goal id from 'sb profile show'.")]) -> None:
    """Remove a goal."""
    _, conn = _open()
    try:
        profile.remove_goal(conn, goal_id)
    except SecondBrainError as exc:
        _fail(str(exc))
    typer.echo(f"goal {goal_id} removed")


@profile_app.command("set-skill")
def profile_set_skill(
    name: Annotated[str, typer.Argument(help="Skill or topic name, e.g. python.")],
    level: Annotated[int, typer.Argument(help="0 none, 1 beginner, 2 intermediate, 3 advanced.")],
) -> None:
    """Set a skill level by hand. Hand-set levels are never changed automatically."""
    _, conn = _open()
    try:
        tag = profile.set_skill(conn, name, level)
    except SecondBrainError as exc:
        _fail(str(exc))
    typer.echo(f"skill '{tag}' set to {profile.LEVEL_NAMES[level]}")


@profile_app.command("remove-skill")
def profile_remove_skill(name: Annotated[str, typer.Argument(help="Name of a hand-set skill.")]) -> None:
    """Remove a hand-set level; the skill falls back to what your finished items imply."""
    _, conn = _open()
    try:
        profile.remove_skill(conn, name)
    except SecondBrainError as exc:
        _fail(str(exc))
    typer.echo("skill removed")


@app.command()
def show(item_id: Annotated[int, typer.Argument(help="Item id from 'sb list'.")]) -> None:
    """Show one item: metadata, status, chunk count and stage timings."""
    _, conn = _open()
    try:
        row = items.get(conn, item_id)
    except SecondBrainError as exc:
        _fail(str(exc))
    for key in row.keys():
        typer.echo(f"{key:<18} {row[key]}")
    typer.echo(f"{'chunks':<18} {len(items.load_chunks(conn, item_id))}")
    stored = analysis.stored_analysis(conn, item_id)
    if stored is not None:
        typer.echo(f"{'difficulty':<18} {stored.difficulty}")
        typer.echo(f"{'topics':<18} {', '.join(stored.topics)}")
        typer.echo(f"{'skills':<18} {', '.join(stored.skills)}")
        typer.echo(f"{'summary':<18} {stored.summary}")
        for point in stored.key_points:
            typer.echo(f"{'key point':<18} {point}")
    for timing in items.load_timings(conn, item_id):
        typer.echo(f"{'timing':<18} {timing['stage']}: {timing['seconds']:.2f}s")
