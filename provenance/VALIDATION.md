# Validation protocol (split v1)

One frozen split is shared by every experiment, so all numbers are comparable. Built by
`src/avito_bootcamp_2026/split.py` (deterministic; re-running it verifies the content hashes in
`data/splits/v1/manifest.json`), scored by `src/avito_bootcamp_2026/evaluate.py`.

```bash
uv run python -m avito_bootcamp_2026.split        # build or verify ("identical content hashes")
```

## Why a special split

Measured facts that make a naive split misleading (details in `ideas.md` §1):

- **The corpus is built around the benchmark queries** (corr. 0.994 between benchmark queries and
  corpus items per location, ~55–75 items per query): each benchmark query has a dense crowd of
  look-alike competitors. Validation must therefore retrieve from the **real corpus**, with queries whose
  answers are real corpus items — not from a corpus with injected answers.
- **The benchmark is a sample of distinct texts:** 2,452 unique texts, 62.5 % never seen in train, the
  seen ones typically 1–8 times. An event-level split would contain 82 % seen texts.
- 17 % of benchmark queries are wide-location searches (region / all Russia), 37 % carry a filter.
- 90 % of corpus items never appear in train, so answers must not be memorisable.

## Construction

1. A *search* = unique (text, location, filters, category, delivery) key of train. Its relevant set = the
   chosen items that are in the benchmark corpus. Eligible: 1–5 such items, delivery = 0 → 26,493 searches,
   12,168 distinct normalised texts (lower case, ё→е, alphanumeric tokens).
2. One search per text (seeded shuffle), stratified to the benchmark's joint distribution of
   **location type × text frequency × filter presence × query length**; shortfalls filled by relaxing the
   strata from the right. val and test pick alternately first; rtrain is drawn from the leftovers.
3. **Seen/unseen texts:** unseen = texts that occur in no other train search; when a cell runs short, the
   rarest seen texts (≤ 30 other rows) are *converted* (all their rows leave `hist`). This reproduces the
   benchmark's 62.5 % unseen share *and* its frequency profile of seen texts.
4. **Item-disjoint:** every train row containing a relevant item of any query set is removed from
   `hist`, and query sets never share relevant items.
5. `hist` = train − query-set searches − rows with their relevant items − converted texts.

## Result

| | bench | val | test | rtrain |
|---|---|---|---|---|
| queries | 2,452 | 2,000 | 2,000 | 5,334 |
| msk / spb / wide | 13.4 / 8.7 / 17.4 % | 13.5 / 8.7 / 17.3 % | 13.5 / 8.7 / 17.3 % | 14.7 / 8.2 / 13.1 % |
| city L / M / S | 22.6 / 29.5 / 8.3 % | 22.5 / 29.5 / 8.5 % | 22.5 / 29.5 / 8.5 % | 23.5 / 26.1 / 14.4 % |
| unseen text | 62.5 % | 62.6 % | 62.8 % | 59.3 % |
| median history rows of seen texts | 3 | 3 | 3 | 30 |
| has filter | 36.9 % | 37.4 % | 37.5 % | 67.4 % |
| 1 / 2 / 3 / 4 / 5+ words | 4.4 / 28.5 / 31.9 / 21.0 / 14.2 % | 4.3 / 29.0 / 32.1 / 21.2 / 13.4 % | 4.2 / 29.3 / 32.2 / 21.0 / 13.3 % | 9.0 / 40.7 / 32.3 / 13.9 / 4.2 % |
| category 0 | 222 | 0 | 0 | 0 |

`hist`: 454,043 of 497,673 train rows (removed: 11,977 query-set search rows, 10,357 other rows with
their relevant items, 21,296 rows of converted texts).

val/test match the benchmark within ~1 pt on every stratum. rtrain is training data: it is skewed to
head, filtered and short queries (the balanced texts were used up by val/test); reweight it with
`evaluate.bench_weights(meta)` when training if that helps on val.

## Scoring

`evaluate answer` reports Recall@50 with a bootstrap 95 % CI, a benchmark-reweighted recall (raking on
the four strata; ≈ plain recall for val/test by construction), and per-segment recall (location type,
text frequency, filter, words, #relevant, popularity of the relevant item). `evaluate pool` gives
candidate recall at depths 50…2000; `evaluate compare` gives the paired difference of two systems with a CI.

Noise: the reference baseline scores 0.859 on val and 0.839 on test — two statistically identical sets
differ by 2 pts. Compare systems on the same set with `compare`; use val+test together for absolute estimates.

## Known biases (read before trusting a number)

1. **Offline is optimistic vs the platform** (public experience: 1.5–9 pts, typically 2–4 for good
   validations). The competitor density around val queries is still lower than around benchmark queries
   (their pools were built for benchmark queries), and category-0 queries (9 % of the benchmark) are absent.
   The orchestrator calibrates `platform ≈ a·local + b` from submissions.
2. **Popular positives:** relevant items of val/test are items chosen in train that survived into the corpus
   (median 39–41 reviews vs 16 for cold corpus items). Popularity/quality features look better offline than
   they are. Check the `pos_reviews_b` segment when a gain comes from such features.
3. **Warm items:** only 4.4 % of corpus items occur in `hist` (9.6 % occur in the full train used for bench).
   History-based channels cover fewer items offline than at benchmark time; do not over-tune them offline.
4. Almost every val/test query has exactly one relevant item (1,979 / 2,000); the benchmark "usually one
   or two" — multi-answer queries are underrepresented.
5. Same (text, location) seen in history: 4.4 % of val vs 6.7 % of bench — exact-history lookups are
   slightly undervalued offline.

## Hashes (split v1)

val_queries `fa46ec92518f5756` · val_qrels `8561b4fb8417e542` · test_queries `8322ba1b3d07a1af` ·
test_qrels `e0795bad58f029fd` · rtrain_queries `06d8b4d565779375` · rtrain_qrels `832a6596569d38d9` ·
hist_row_ids `778bfea3c237f606` · bench_meta `28fb821fb1914ecf`
