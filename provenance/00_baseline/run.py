"""Reference baseline: one BM25 index over the concatenated item text + a learned geo prior.

    uv run python experiments/00_baseline/run.py

score(q, i) = BM25(q, i) + ALPHA * log P(item_location | search_location)

P(item_loc | search_loc) is the empirical transition frequency in the rows the model may learn from
(fit.parquet for val/test, full train for bench), with the query's own location floored at 0.5.
Writes out/{val,test,bench}_answer.csv next to this file and publishes the top-1000 candidates as the shared
channel data/shared/orchestrator/baseline_bm25_geo (a starting pool for the ranker).
It exists to (1) sanity-check the split/evaluation plumbing and (2) give every agent the same anchor.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
import snowballstemmer
from sklearn.feature_extraction.text import CountVectorizer

from avito_bootcamp_2026 import exchange, paths

ALPHA = 2.0          # geo weight, tuned once on val in the exploration (0 -> 0.42, 2 -> best)
POOL_K = 1000
OUT = Path(__file__).parent / "out"
TOKEN = re.compile(r"[a-zа-я0-9]+")
_stem = snowballstemmer.stemmer("russian")
_cache: dict[str, str] = {}


def analyze(text: str) -> list[str]:
    """Lower case, ё->е, alphanumeric tokens, Snowball stems (cached: the vocabulary is small)."""
    out = []
    for w in TOKEN.findall((text or "").lower().replace("ё", "е")):
        s = _cache.get(w)
        if s is None:
            s = _cache[w] = _stem.stemWord(w)
        out.append(s)
    return out


class BM25:
    """BM25 as sparse-matrix algebra: scores = binary(query terms) @ W^T."""

    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.cv = CountVectorizer(analyzer=analyze, dtype=np.float32)
        tf = self.cv.fit_transform(docs).tocoo()
        n_docs = tf.shape[0]
        df = np.bincount(tf.col, minlength=tf.shape[1])
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / dl.mean())
        w = sp.csr_matrix((tf.data * (k1 + 1) / denom * idf[tf.col], (tf.row, tf.col)), shape=tf.shape)
        self.wt = w.T.tocsr()

    def scores(self, queries: list[str]) -> np.ndarray:
        q = self.cv.transform(queries)
        q.data[:] = 1
        return (q @ self.wt).toarray()


def geo_log_prior(train_rows: pl.DataFrame, item_loc: np.ndarray):
    """Return f(search_loc) -> vector of log P(item_loc | search_loc) over corpus items."""
    trans = (train_rows.group_by("search_location_id", "item_location_id").len()
             .with_columns((pl.col("len") / pl.col("len").sum().over("search_location_id")).alias("p")))
    table: dict[int, dict[int, float]] = {}
    for s, i, p in trans.select("search_location_id", "item_location_id", "p").iter_rows():
        table.setdefault(s, {})[i] = p
    locs, inverse = np.unique(item_loc, return_inverse=True)
    cache: dict[int, np.ndarray] = {}

    def vec(search_loc: int) -> np.ndarray:
        if search_loc not in cache:
            probs = table.get(search_loc, {})
            v = np.array([probs.get(int(l), 1e-6) for l in locs], dtype=np.float32)
            v[locs == search_loc] = np.maximum(v[locs == search_loc], 0.5)
            cache[search_loc] = np.log(v)[inverse]
        return cache[search_loc]
    return vec


def main() -> None:
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    corpus = paths.load_corpus(["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw",
                                "item_location_id"])
    docs = (corpus["item_title_raw"].fill_null("") + " " + corpus["item_title_raw"].fill_null("") + " "
            + corpus["item_infm_params_text"].fill_null("").str.slice(0, 600) + " "
            + corpus["item_description_raw"].fill_null("").str.slice(0, 1500)).to_list()
    bm25 = BM25(docs)
    ids = np.array(corpus["item_id"].to_list())
    item_loc = corpus["item_location_id"].to_numpy()
    print(f"index built in {time.time() - t0:.0f}s")

    for split in ("rtrain", "val", "test", "bench"):
        queries = paths.load_queries(split)
        geo = geo_log_prior(paths.load_train_rows(split, ["search_location_id", "item_location_id"]), item_loc)
        qids, texts, locs = queries["query_id"].to_list(), queries["search_query"].to_list(), queries["search_location_id"].to_list()
        answers, pool_rows = [], []
        for start in range(0, len(qids), 250):
            scores = bm25.scores(texts[start:start + 250])
            for j, row in enumerate(scores):
                qi = start + j
                s = row + ALPHA * geo(locs[qi])
                top = np.argpartition(-s, POOL_K)[:POOL_K]
                top = top[np.lexsort((ids[top], -s[top]))]          # score desc, item_id as tie-break
                answers.append(" ".join(ids[top[:50]]))
                pool_rows.append(pl.DataFrame({"query_id": qids[qi], "item_id": ids[top],
                                               "score": s[top], "rank": np.arange(1, POOL_K + 1, dtype=np.int32)}))
        pl.DataFrame({"query_id": qids, "answer": answers}).write_csv(OUT / f"{split}_answer.csv")
        exchange.save_channel("orchestrator", "baseline_bm25_geo", split, pl.concat(pool_rows))
        print(f"{split}: {len(qids)} queries done, {time.time() - t0:.0f}s")
    exchange.write_meta("orchestrator", "baseline_bm25_geo", kind="channel", fit_only=True, k=POOL_K,
                        description=f"BM25 over title x2 + params[:600] + desc[:1500] + {ALPHA} * log geo prior",
                        command="uv run python experiments/00_baseline/run.py")


if __name__ == "__main__":
    main()
