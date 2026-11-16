"""Command line interface. Run `sb --help`."""

import logging
import sqlite3
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError

from secondbrain import db, items
from secondbrain.config import Settings
from secondbrain.embed import Embedder, FastEmbedEmbedder
from secondbrain.errors import SecondBrainError
from secondbrain.ingest import pipeline

app = typer.Typer(help="Second Brain: save, search and plan your learning.", no_args_is_help=True)
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
) -> None:
    """Save one or more URLs: fetch, chunk and embed."""
    settings, conn = _open()
    get_embedder = _embedder_factory(settings)
    failures = 0
    for one_url in _read_urls(url, file):
        try:
            result = pipeline.ingest_url(conn, settings, get_embedder, one_url, transcribe=transcribe)
            typer.echo(f"{one_url} -> {_describe(result)}")
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
    typer.echo(f"{'id':>4}  {'status':<13}  {'source':<8}  {'min':>4}  title")
    for row in rows:
        minutes = "-" if row["duration_sec"] is None else str(round(row["duration_sec"] / 60))
        title = row["title"] or row["url"]
        typer.echo(
            f"{row['id']:>4}  {row['ingest_status']:<13}  {row['transcript_source']:<8}  "
            f"{minutes:>4}  {title}"
        )


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
    for timing in items.load_timings(conn, item_id):
        typer.echo(f"{'timing':<18} {timing['stage']}: {timing['seconds']:.2f}s")
