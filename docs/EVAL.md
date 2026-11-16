# Evaluation results

**No evaluation has been run yet, so there are no numbers to report.**

This file is written by `scripts/eval.py`. It was not possible to produce results in the build
environment: that needs the real embedding model, a real library of saved items, an LLM for the
faithfulness judge, and labels written by the user.

To produce the results on your machine:

1. Save and analyze 30-40 items, settle `LINK_THRESHOLD`, and fill in the label files
   (steps in [`eval/README.md`](../eval/README.md)).
2. Run `uv run python scripts/eval.py`.
3. For the human spot check, fill in the `eval/spot_check.yaml` that the first run creates and
   run the script again.

The script replaces this file with the run details (date, commit, machine, configuration) and
the measured metrics: retrieval recall@5 and MRR, linking precision and recall, summary
faithfulness with judge-vs-human agreement, ingest time per stage, search latency p50/p95, and
the LLM's JSON-validity rate. Anything it cannot measure is reported as "Not run".
