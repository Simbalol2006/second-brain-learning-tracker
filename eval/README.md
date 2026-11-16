# Evaluation labels

The final phase measures the app against labels that **you** write. This folder holds them.
The files are created from your own saved items, so they do not exist until you run step 2.

## Steps

1. **Build a real library first.** Save about 30-40 items you would really keep, and analyze them:

   ```bash
   uv run sb ingest --file urls.txt --analyze
   ```

2. **Freeze the link settings, then create the templates.** Run `uv run sb links --stats`, look at
   `uv run sb related <id>` for a few items, and settle `LINK_THRESHOLD` (see the main README).
   Do this *before* labelling: the labels must not be used to pick the threshold. Then:

   ```bash
   uv run sb links --rebuild
   uv run sb eval-templates
   ```

3. **Fill in the three files** (about an hour in total).

| File | What you do | Target size |
|------|-------------|-------------|
| `retrieval.yaml` | Write search queries and list the URLs of the items that answer each one | ~25 queries |
| `links.yaml` | Set `related: true` or `false` for each pair | 30 pairs |
| `summaries.yaml` | Set `spot_check: true` on 5 items you will check by hand | 15 items |

Instructions are repeated at the top of each file.

## Rules that keep the numbers honest

- **Retrieval:** write queries the way you would search, from memory of the content. Do not
  copy titles, and do not run `sb search` first to see what comes up.
- **Links:** the files do not show the system's scores, and the pairs are in random order. Do not
  look at `sb related` while labelling.
- **Pair selection is not random.** Ten pairs are the highest-scoring ones, ten come from the
  middle of the score range and ten are random. This gives enough related pairs to measure
  recall with only 30 labels, but it means precision and recall describe this sample, not all
  pairs in the library. `docs/EVAL.md` will say so next to the numbers.
- **Do not re-run `sb eval-templates --force`** after labelling: it replaces your labels.
- If you change `LINK_THRESHOLD` after seeing the results, the eval reports both the frozen and
  the tuned numbers and marks the second as tuned on the eval set.

## Privacy

The files contain the URLs and titles of your saved items and the queries you write. They are
meant to be committed so the eval can be reproduced. If the repo is public and you would rather
not publish them, add `eval/*.yaml` to `.gitignore`.
