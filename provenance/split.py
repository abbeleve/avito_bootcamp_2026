"""Build the shared, frozen split (protocol v1) used by every experiment.

    uv run python -m avito_bootcamp_2026.split            # build, or verify an existing split
    uv run python -m avito_bootcamp_2026.split --dry-run  # only print statistics

Query sets (all in the schema of benchmark_queries.parquet, with qrels and segment metadata):
* val    (2,000) - tune on it;
* test   (2,000) - confirm release candidates only;
* rtrain (6,000) - labelled queries for training the final ranker (features for them must come from hist).
Learning rows:
* hist.parquet - train minus every val/test/rtrain search, minus every row that contains one of their
  relevant items, minus the rows of texts converted to "unseen". Every channel, statistic or model used
  to score rtrain/val/test must be learned from hist only (full train is allowed for bench).

The procedure is deterministic: re-running it reproduces identical content hashes (manifest.json).

Why the split looks like this (numbers measured on the data, see VALIDATION.md):
* The benchmark corpus is built around the benchmark queries (dense crowds of look-alike items), so all
  query sets are taken from train *searches whose chosen item is already in the benchmark corpus*;
  retrieval then runs against the real, unchanged corpus.
* A "search" is a unique (text, location, filters, category, delivery) key. Its relevant set is the
  chosen items that are in the corpus. Aggregated head searches with > MAX_REL such items are skipped
  (the benchmark says a query usually has one or two relevant items).
* The benchmark holds 2,452 *unique* texts, so we keep one search per normalised text, and texts are
  disjoint across query sets.
* Stratified to the benchmark's joint distribution of location type x text frequency x filter presence
  x query length. "Text frequency" = other train rows with the same normalised text: 62.5 % of benchmark
  texts never occur in train, the seen ones mostly occur 1-8 times. val/test pick first; rtrain is drawn
  from the leftovers (best-effort fidelity, it is training data).
* Unseen texts come from searches whose text occurs nowhere else in train. When a cell runs short, the
  rarest seen texts (<= CONVERT_MAX other rows) are converted: all their rows leave hist.
* Item-disjoint: relevant items of the query sets never occur in hist and never repeat across query sets,
  so a model cannot memorise the answer item (90 % of the benchmark corpus is absent from train as well).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np
import polars as pl

from avito_bootcamp_2026 import paths

SEED = 20260928
N_EVAL = {"val": 2000, "test": 2000}
N_RTRAIN = {"rtrain": 6000}
ID_PREFIX = {"val": "val", "test": "tst", "rtrain": "rtr"}
MAX_REL = 5
CONVERT_MAX = 30
KEY = ["search_query", "search_location_id", "search_infm_params_text", "search_category",
       "search_is_delivery_search"]
# Stratification cells; shortfalls are filled by relaxing the strata from the right.
STRATA = ["loc_coarse", "f_bucket", "has_filter", "words_bucket"]
QUERY_COLS = ["query_id", *KEY[:2], "search_is_delivery_search", "search_infm_params_text", "search_category"]
META_COLS = ["query_id", "loc_type", "loc_coarse", "f_bucket", "has_filter", "n_words", "words_bucket", "seen_text",
             "seen_same_loc", "text_rows_in_history", "n_items_at_loc", "search_category"]

# Location ids identified from the data (ideas.md, section 1.3): the two biggest cities and the three
# "wide" search locations whose chosen items are never at the same location id.
NAMED_LOCATIONS = {637640: "msk", 653240: "spb", 107620: "wide_msk", 107621: "wide_spb", 621540: "wide_rus"}


def norm_text(col: str) -> pl.Expr:
    """Normalised query text used for uniqueness and seen/unseen checks: lower case, ё->е, alnum tokens."""
    return (pl.col(col).str.to_lowercase().str.replace_all("ё", "е")
            .str.replace_all(r"[^0-9a-zа-я]+", " ").str.strip_chars())


def freq_bucket(col: str) -> pl.Expr:
    c = pl.col(col)
    return (pl.when(c == 0).then(pl.lit("0")).when(c == 1).then(pl.lit("1")).when(c <= 3).then(pl.lit("2-3"))
            .when(c <= 8).then(pl.lit("4-8")).when(c <= 30).then(pl.lit("9-30")).otherwise(pl.lit("31+")))


def location_table(train: pl.DataFrame, corpus: pl.DataFrame) -> pl.DataFrame:
    """Per search location: how local its choices are and how many corpus items sit there -> location type."""
    local = train.group_by("search_location_id").agg(
        (pl.col("item_location_id") == pl.col("search_location_id")).mean().alias("self_rate"))
    at_loc = (corpus.group_by("item_location_id").len()
              .rename({"item_location_id": "search_location_id", "len": "n_items_at_loc"}))
    table = local.join(at_loc, on="search_location_id", how="full", coalesce=True).with_columns(
        pl.col("self_rate").fill_null(1.0), pl.col("n_items_at_loc").fill_null(0))
    loc = pl.col("search_location_id")
    loc_type = (pl.when(loc.is_in(list(NAMED_LOCATIONS))).then(loc.replace_strict(NAMED_LOCATIONS, default=None))
                .when((pl.col("self_rate") < 0.3) | (pl.col("n_items_at_loc") == 0)).then(pl.lit("wide_other"))
                .when(pl.col("n_items_at_loc") >= 2000).then(pl.lit("city_L"))
                .when(pl.col("n_items_at_loc") >= 400).then(pl.lit("city_M"))
                .otherwise(pl.lit("city_S")))
    return table.with_columns(loc_type.alias("loc_type")).with_columns(
        pl.when(pl.col("loc_type").str.starts_with("wide")).then(pl.lit("wide"))
        .otherwise(pl.col("loc_type")).alias("loc_coarse"))


def query_features(df: pl.DataFrame, locations: pl.DataFrame) -> pl.DataFrame:
    """Segment columns shared by searches and benchmark queries."""
    n_words = pl.col("norm_q").str.split(" ").list.len()
    return (df.join(locations, on="search_location_id", how="left")
            .with_columns(
                (pl.col("search_infm_params_text").str.strip_chars() != "").alias("has_filter"),
                n_words.alias("n_words"),
                pl.when(n_words >= 5).then(pl.lit("5+")).otherwise(n_words.cast(pl.String)).alias("words_bucket"),
            ))


def allocate(bench_cells: Counter, n: int) -> Counter:
    """Split n into integer per-cell targets proportional to the benchmark (largest remainder)."""
    total = sum(bench_cells.values())
    raw = {c: n * k / total for c, k in bench_cells.items()}
    out = Counter({c: int(v) for c, v in raw.items()})
    for c in sorted(raw, key=lambda c: (-(raw[c] - out[c]), c))[: n - sum(out.values())]:
        out[c] += 1
    return out


def select(cands: pl.DataFrame, bench_cells: Counter, sizes: dict[str, int],
           taken: set[int], used_items: set[str]) -> dict[str, list[tuple[int, bool]]]:
    """Stratified, item-disjoint selection -> {split: [(row, convert_to_unseen)]}; updates taken/used_items.

    Every candidate enters the pool of its own cell. Rare seen texts (1..CONVERT_MAX other rows) also
    enter the matching f_bucket="0" pool, after all naturally unseen candidates and rarest first; taking
    them from there means their text is removed from hist. Cells are filled at full depth first, then
    shortfalls are filled with the rightmost strata relaxed. Splits of one call pick alternately, so none
    gets priority on scarce cells. A candidate is skipped if one of its relevant items is already
    relevant for a selected search (keeps the query sets item-disjoint).
    """
    rows = cands.select("row", *STRATA, "rel", "f_other").to_dicts()
    entries = [(tuple(r[k] for k in STRATA), r, False) for r in rows]
    convertible = sorted((r for r in rows if 0 < r["f_other"] <= CONVERT_MAX), key=lambda r: (r["f_other"], r["row"]))
    entries += [(tuple({**r, "f_bucket": "0"}[k] for k in STRATA), r, True) for r in convertible]

    deficit = {s: allocate(bench_cells, n) for s, n in sizes.items()}
    chosen: dict[str, list[tuple[int, bool]]] = {s: [] for s in sizes}
    for depth in range(len(STRATA), -1, -1):
        pools: dict[tuple, list] = defaultdict(list)
        for cell, r, convert in entries:
            pools[cell[:depth]].append((r, convert))
        need = {s: Counter() for s in sizes}
        for s in sizes:
            for cell, d in deficit[s].items():
                need[s][cell[:depth]] += d
        for cell in sorted(pools, key=str):
            it = iter(pools[cell])
            progressed = True
            while progressed and any(need[s][cell] > 0 for s in sizes):
                progressed = False
                for s in sizes:
                    if need[s][cell] <= 0:
                        continue
                    for r, convert in it:
                        if r["row"] in taken or used_items.intersection(r["rel"]):
                            continue
                        chosen[s].append((r["row"], convert))
                        taken.add(r["row"])
                        used_items.update(r["rel"])
                        need[s][cell] -= 1
                        full = next(c for c in sorted(deficit[s], key=str) if c[:depth] == cell and deficit[s][c] > 0)
                        deficit[s][full] -= 1
                        progressed = True
                        break
    return chosen


def content_hash(df: pl.DataFrame) -> str:
    return hashlib.sha256(df.write_csv().encode()).hexdigest()[:16]


def share(df: pl.DataFrame, col: str) -> dict:
    return {str(k): round(v, 4) for k, v in
            df.group_by(col).len().with_columns(pl.col("len") / df.height).sort(col).iter_rows()}


def with_history(df: pl.DataFrame, history: pl.DataFrame) -> pl.DataFrame:
    """Attach how often the query text (and text+location) occurs in the rows a model may learn from."""
    texts = history.group_by("norm_q").agg(pl.len().alias("text_rows_in_history"))
    text_loc = history.select("norm_q", "search_location_id").unique().with_columns(pl.lit(True).alias("seen_same_loc"))
    return (df.join(texts, on="norm_q", how="left").join(text_loc, on=["norm_q", "search_location_id"], how="left")
            .with_columns(pl.col("text_rows_in_history").fill_null(0), pl.col("seen_same_loc").fill_null(False))
            .with_columns((pl.col("text_rows_in_history") > 0).alias("seen_text"),
                          freq_bucket("text_rows_in_history").alias("f_bucket")))


def build(write: bool) -> dict:
    rng = np.random.default_rng(SEED)
    train = pl.read_parquet(paths.main_root() / "train.parquet").with_row_index("train_row_id")
    corpus = paths.load_corpus(["item_id", "item_location_id", "item_rating_reviews_count"])
    train = train.with_columns(norm_text("search_query").alias("norm_q"),
                               pl.col("item_id").is_in(corpus["item_id"].implode()).alias("in_corpus"))
    locations = location_table(train, corpus)
    bench = with_history(query_features(paths.load_queries("bench").with_columns(norm_text("search_query").alias("norm_q")),
                                        locations), train)

    # ---- candidate searches: one per normalised text ------------------------------------------------
    text_rows = train.group_by("norm_q").len().rename({"len": "text_rows"})
    searches = (train.group_by(KEY).agg(
        pl.col("norm_q").first(), pl.len().alias("key_rows"),
        pl.col("item_id").filter(pl.col("in_corpus")).unique().sort().alias("rel"),
    ).with_columns(pl.col("rel").list.len().alias("n_rel")))
    eligible = (searches.filter(pl.col("n_rel").is_between(1, MAX_REL), pl.col("search_is_delivery_search") == 0,
                                pl.col("norm_q") != "")
                .join(text_rows, on="norm_q").with_columns((pl.col("text_rows") - pl.col("key_rows")).alias("f_other")))
    # joins and group_by do not keep row order -> sort by the full key before the seeded shuffle
    eligible = query_features(eligible, locations).with_columns(freq_bucket("f_other").alias("f_bucket")).sort(KEY)
    eligible = eligible[rng.permutation(eligible.height)]
    cands = eligible.unique("norm_q", keep="first", maintain_order=True).with_row_index("row")

    bench_cells = Counter(tuple(r) for r in bench.select(STRATA).iter_rows())
    taken: set[int] = set()
    used_items: set[str] = set()
    chosen = select(cands, bench_cells, N_EVAL, taken, used_items)            # evaluation sets pick first
    chosen |= select(cands, bench_cells, N_RTRAIN, taken, used_items)         # ranker training from leftovers
    picked = pl.concat([cands[[r for r, _ in rows]].with_columns(pl.lit(s).alias("split"),
                                                               pl.Series("convert", [c for _, c in rows]))
                        for s, rows in chosen.items()])

    # ---- hist = train minus query-set searches, rows with their relevant items, converted texts -----
    sel_keys = picked.select(KEY)
    rel_items = picked["rel"].explode(empty_as_null=False).unique().implode()
    remaining = (train.join(sel_keys, on=KEY, how="anti")
                 .filter(~pl.col("item_id").is_in(rel_items)).sort("train_row_id"))
    hist = remaining.filter(~pl.col("norm_q").is_in(picked.filter("convert")["norm_q"].implode()))
    n_search_rows = train.join(sel_keys, on=KEY, how="semi").height
    removed = {"query_set_search_rows": n_search_rows,
               "other_rows_with_query_set_items": train.height - n_search_rows - remaining.height,
               "converted_text_rows": remaining.height - hist.height}

    # ---- per-query metadata (same columns for every query set and bench) ----------------------------
    reviews = corpus.select("item_id", pl.col("item_rating_reviews_count").fill_null(0).alias("rv"))
    pos_rev = (picked.select(*KEY, "rel").explode("rel", empty_as_null=False)
               .join(reviews, left_on="rel", right_on="item_id").group_by(KEY).agg(pl.col("rv").min().alias("pos_reviews_min")))
    picked = with_history(picked.drop("f_bucket").join(pos_rev, on=KEY, how="left"), hist).sort("split", "row")
    picked = picked[rng.permutation(picked.height)].with_columns(      # ids carry no structure
        (pl.col("split").replace_strict(ID_PREFIX)
         + pl.int_range(pl.len()).over("split").cast(pl.String).str.zfill(13)).alias("query_id"))

    bench_schema = paths.load_queries("bench").schema
    outputs: dict[str, pl.DataFrame] = {"bench_meta": bench.select(META_COLS).sort("query_id")}
    for s in chosen:
        part = picked.filter(pl.col("split") == s).sort("query_id")
        outputs[f"{s}_queries"] = part.select(QUERY_COLS).cast(dict(bench_schema))
        outputs[f"{s}_qrels"] = (part.select("query_id", pl.col("rel").alias("item_id"))
                                 .explode("item_id", empty_as_null=False).sort("query_id", "item_id"))
        outputs[f"{s}_meta"] = part.select(*META_COLS, "n_rel", "pos_reviews_min")

    hist = hist.drop("norm_q", "in_corpus")
    warm = corpus["item_id"].is_in(hist["item_id"].unique().implode()).mean()
    manifest = {
        "protocol": paths.SPLIT_VERSION, "seed": SEED, "max_rel": MAX_REL, "convert_max": CONVERT_MAX,
        "counts": {"train_rows": train.height, "hist_rows": hist.height, "eligible_searches": eligible.height,
                   "candidate_texts": cands.height, **{s: len(r) for s, r in chosen.items()},
                   "converted_texts": int(picked["convert"].sum())},
        "removed_rows": removed,
        "corpus_items_seen_in_hist": round(float(warm), 4),
        "shares": {name: {col: share(outputs[f"{name}_meta"], col)
                          for col in ["loc_coarse", "f_bucket", "has_filter", "words_bucket", "seen_text"]}
                   for name in ["bench", *chosen]},
        "hashes": {**{k: content_hash(v) for k, v in outputs.items()},
                   "hist_row_ids": hashlib.sha256(np.sort(hist["train_row_id"].to_numpy()).astype("<i8").tobytes()).hexdigest()[:16]},
    }
    if write:
        out = paths.split_dir()
        out.mkdir(parents=True, exist_ok=True)
        hist.write_parquet(out / "hist.parquet")
        for name, df in outputs.items():
            df.write_parquet(out / f"{name}.parquet")
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print statistics, write nothing")
    ap.add_argument("--force", action="store_true", help="overwrite an existing split even if hashes differ")
    args = ap.parse_args()
    existing = paths.split_dir() / "manifest.json"
    if existing.exists() and not args.dry_run:
        old = json.loads(existing.read_text())["hashes"]
        new = build(write=False)
        if new["hashes"] == old:
            print(f"split {paths.SPLIT_VERSION} verified: identical content hashes, nothing rewritten")
            return
        if not args.force:
            raise SystemExit(f"existing split differs from a fresh build:\n{old}\n{new['hashes']}\nuse --force to overwrite")
    manifest = build(write=not args.dry_run)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
