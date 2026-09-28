"""Shared helpers for agent_4 experiments: thread limits, text analysis, BM25, cached score matrices.

Import this module FIRST in every agent_4 script (it sets the thread-count environment variables before
numpy / polars are imported).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import re
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
import snowballstemmer
from sklearn.feature_extraction.text import CountVectorizer

from avito_bootcamp_2026 import paths

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"          # heavy, git-ignored intermediates
OUT = HERE / "out"                # answer CSVs scored by the harness
ART.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
SPLITS = ("rtrain", "val", "test", "bench")

TOKEN = re.compile(r"[a-zа-я0-9]+")
_stem = snowballstemmer.stemmer("russian")
_cache: dict[str, str] = {}


def tokens(text: str | None) -> list[str]:
    return TOKEN.findall((text or "").lower().replace("ё", "е"))


def analyze(text: str | None) -> list[str]:
    """Lower case, ё->е, alphanumeric tokens, Snowball stems (same analyzer as the reference baseline)."""
    out = []
    for w in tokens(text):
        s = _cache.get(w)
        if s is None:
            s = _cache[w] = _stem.stemWord(w)
        out.append(s)
    return out


class BM25:
    """BM25 as sparse-matrix algebra: scores = binary(query terms) @ W^T (as in experiments/00_baseline)."""

    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75, analyzer=analyze):
        self.cv = CountVectorizer(analyzer=analyzer, dtype=np.float32)
        tf = self.cv.fit_transform(docs).tocoo()
        n_docs = tf.shape[0]
        df = np.bincount(tf.col, minlength=tf.shape[1])
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / max(dl.mean(), 1e-9))
        w = sp.csr_matrix((tf.data * (k1 + 1) / denom * self.idf[tf.col], (tf.row, tf.col)), shape=tf.shape)
        self.wt = w.T.tocsr()

    def sparse_scores(self, queries: list[str]) -> sp.csr_matrix:
        q = self.cv.transform(queries)
        q.data[:] = 1
        return (q @ self.wt).tocsr()


def f32(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Float64)


def corpus_docs(corpus: pl.DataFrame) -> list[str]:
    """Baseline document: title x2 + params[:600] + description[:1500]."""
    return (corpus["item_title_raw"].fill_null("") + " " + corpus["item_title_raw"].fill_null("") + " "
            + corpus["item_infm_params_text"].fill_null("").str.slice(0, 600) + " "
            + corpus["item_description_raw"].fill_null("").str.slice(0, 1500)).to_list()


def baseline_bm25_scores(split: str) -> sp.csr_matrix:
    """Sparse BM25 (queries x corpus) of the baseline text model, cached in artifacts/."""
    path = ART / f"bm25_base_{split}.npz"
    if path.exists():
        return sp.load_npz(path).tocsr()
    t0 = time.time()
    corpus = paths.load_corpus(["item_title_raw", "item_infm_params_text", "item_description_raw"])
    bm = BM25(corpus_docs(corpus))
    del corpus
    for s in SPLITS:
        p = ART / f"bm25_base_{s}.npz"
        if not p.exists():
            m = bm.sparse_scores(paths.load_queries(s)["search_query"].to_list()).astype(np.float32)
            sp.save_npz(p, m)
            print(f"bm25 {s}: nnz={m.nnz:,} ({time.time() - t0:.0f}s)")
    return sp.load_npz(path).tocsr()


def top_k(scores: np.ndarray, ids: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k scores, ordered by score desc then item_id (deterministic tie-break)."""
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.lexsort((ids[top], -scores[top]))]


def write_answer(path: Path, qids: list[str], answers: list[list[str]]) -> None:
    assert all(len(a) == 50 and len(set(a)) == 50 for a in answers), "need exactly 50 unique ids per query"
    pl.DataFrame({"query_id": qids, "answer": [" ".join(a) for a in answers]}).write_csv(path)


def recall_at(split: str, qids: list[str], answers: list[list[str]], k: int = 50) -> np.ndarray:
    """Per-query recall (rtrain/val/test) for quick in-process tuning; official numbers go through evaluate."""
    rel = paths.load_qrels(split).group_by("query_id").agg(pl.col("item_id"))
    rel = dict(rel.iter_rows())
    out = np.zeros(len(qids))
    for j, (q, a) in enumerate(zip(qids, answers)):
        r = rel.get(q, [])
        if r:
            top = set(a[:k])
            out[j] = sum(i in top for i in r) / len(r)
    return out
