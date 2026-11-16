"""Run the evaluation and write docs/EVAL.md.

    uv run python scripts/eval.py                  # everything
    uv run python scripts/eval.py --skip-summaries # no LLM judge calls
    uv run python scripts/eval.py --judge-model claude-sonnet-5-5

Needs: saved items, and the label files in eval/ (see eval/README.md). Parts whose
labels are missing are reported as "Not run"; nothing is estimated.
"""

import argparse
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from secondbrain import db, evaluation
from secondbrain.config import Settings
from secondbrain.embed import Embedder, FastEmbedEmbedder
from secondbrain.errors import SecondBrainError
from secondbrain.llm import LLMProvider, build_provider


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Second Brain evaluation.")
    parser.add_argument("--eval-dir", type=Path, default=Path("eval"), help="Folder with the label files.")
    parser.add_argument("--out", type=Path, default=Path("docs/EVAL.md"), help="Report file to write.")
    parser.add_argument("--repeats", type=int, default=5, help="Timed searches per query.")
    parser.add_argument("--judge-model", default=None, help="Model for the faithfulness judge (default: LLM_MODEL).")
    parser.add_argument("--skip-summaries", action="store_true", help="Skip the LLM judge section.")
    parser.add_argument("--rejudge", action="store_true", help="Ignore cached judge verdicts and ask again.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = Settings()
    except ValidationError as exc:
        print(f"Invalid configuration (check .env):\n{exc}", file=sys.stderr)
        return 1
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    judge_settings = settings.model_copy(update={"llm_model": args.judge_model}) if args.judge_model else settings

    def get_embedder() -> Embedder:
        return FastEmbedEmbedder(settings.embedding_model, settings.model_cache_dir)

    def get_judge() -> LLMProvider:
        return build_provider(judge_settings)

    try:
        conn = db.connect(settings.db_path)
        sections = evaluation.run(
            conn, settings, get_embedder, get_judge, args.eval_dir,
            repeats=args.repeats, skip_summaries=args.skip_summaries, rejudge=args.rejudge,
        )
        judge_name = f"{judge_settings.llm_provider} / {judge_settings.llm_model_name}"
        report = evaluation.render(sections, evaluation.environment(conn, settings, judge_name))
    except SecondBrainError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    print(f"Wrote {args.out}")
    for section in sections:
        if section.not_run:
            print(f"  NOT RUN  {section.title}: {section.not_run}")
        for name, value in section.headline.items():
            print(f"  {name}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
