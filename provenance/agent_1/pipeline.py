"""agent_1 infrastructure: analyzed-field caches, BM25 weight matrices, geo prior, fast exact evaluation.

The corpus fields are tokenised once per analyzer and stored as integer ids (flat int32 array + offsets) in
experiments/agent_1/artifacts/analyzed_<analyzer>/. A BM25 matrix of any field combination is then pure
numpy/scipy (seconds), which makes a small grid over fields / k1 / b / weights cheap.

Evaluation shortcut used for tuning: for each query the rank of each relevant item under a scoring function
is (#items with a strictly higher score) + 1, computed on the full corpus (exact, no candidate pool).
"""

from __future__ import annotations

import json
import sys
import time
from array import array
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
import lexical as lx  # noqa: E402
from avito_bootcamp_2026 import evaluate, paths  # noqa: E402

HERE = Path(__file__).parent
ART = HERE / "artifacts"
OUT = HERE / "out"
PARAMS_TABLE = paths.shared_dir("agent_1") / "items_params" / "items.parquet"

# desc is split into char ranges so that desc[:1500], desc[:4000] and the full text are sums of parts
FIELDS = ("title", "core", "desc_a", "desc_b", "desc_c", "other", "addr")
FIELD_PARTS = {"title": ("title",), "core": ("core",), "desc1500": ("desc_a",), "desc4000": ("desc_a", "desc_b"),
               "descall": ("desc_a", "desc_b", "desc_c"), "other": ("other",), "addr": ("addr",)}


def log(msg: str, t0: float | None = None) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" + (f" ({time.time() - t0:.0f}s)" if t0 else ""), flush=True)


# ----------------------------------------------------------------------------------------------------
# Corpus fields and analyzed caches
# ----------------------------------------------------------------------------------------------------

def corpus_field_texts() -> dict[str, list[str]]:
    corpus = paths.load_corpus(["item_id", "item_title_raw", "item_description_raw"])
    prm = pl.read_parquet(PARAMS_TABLE, columns=["item_id", "core_text", "other_params_text", "place"])
    assert corpus["item_id"].equals(prm["item_id"]), "items_params must be in corpus order"
    desc = corpus["item_description_raw"].fill_null("")
    return {
        "title": corpus["item_title_raw"].fill_null("").to_list(),
        "core": prm["core_text"].to_list(),
        "desc_a": desc.str.slice(0, 1500).to_list(),
        "desc_b": desc.str.slice(1500, 2500).to_list(),
        "desc_c": desc.str.slice(4000).to_list(),
        "other": prm["other_params_text"].to_list(),
        "addr": prm["place"].to_list(),
    }


class Analyzed:
    """Integer-coded analyzed corpus fields for one analyzer."""

    def __init__(self, analyzer: str):
        self.analyzer = analyzer
        self.dir = ART / f"analyzed_{analyzer}"
        self.fn = lx.ANALYZERS[analyzer]
        if not (self.dir / "vocab.json").exists():
            self._build()
        self.vocab: dict[str, int] = {t: i for i, t in enumerate(json.loads((self.dir / "vocab.json").read_text()))}
        self.n_docs = int(np.load(self.dir / "title_off.npy").shape[0] - 1)

    def _build(self) -> None:
        t0 = time.time()
        self.dir.mkdir(parents=True, exist_ok=True)
        texts = corpus_field_texts()
        vocab: dict[str, int] = {}
        for f in FIELDS:
            ids, off = array("i"), array("q", [0])
            for t in texts[f]:
                for w in self.fn(t):
                    i = vocab.get(w)
                    if i is None:
                        i = vocab[w] = len(vocab)
                    ids.append(i)
                off.append(len(ids))
            np.save(self.dir / f"{f}_ids.npy", np.asarray(ids, dtype=np.int32))
            np.save(self.dir / f"{f}_off.npy", np.asarray(off, dtype=np.int64))
            log(f"{self.analyzer}: field {f} analyzed, {len(ids)} tokens, vocab {len(vocab)}", t0)
        (self.dir / "vocab.json").write_text(json.dumps(list(vocab), ensure_ascii=False))

    def tf(self, parts: tuple[str, ...]) -> sp.csr_matrix:
        """Document-term count matrix of the concatenation of the given field parts."""
        mats = []
        for f in parts:
            ids = np.load(self.dir / f"{f}_ids.npy")
            off = np.load(self.dir / f"{f}_off.npy")
            rows = np.repeat(np.arange(self.n_docs, dtype=np.int32), np.diff(off).astype(np.int64))
            m = sp.csr_matrix((np.ones(len(ids), dtype=np.float32), (rows, ids)), shape=(self.n_docs, len(self.vocab)))
            m.sum_duplicates()
            mats.append(m)
        out = mats[0]
        for m in mats[1:]:
            out = out + m
        return out.tocsr()

    def query_matrix(self, texts: list[str]) -> sp.csr_matrix:
        """Binary query-term matrix (unknown terms are dropped, repeated terms count once)."""
        rows, cols = [], []
        for r, t in enumerate(texts):
            for c in {self.vocab[w] for w in self.fn(t) if w in self.vocab}:
                rows.append(r)
                cols.append(c)
        return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(texts), len(self.vocab)))


def bm25_weights(tf: sp.csr_matrix, k1: float, b: float) -> sp.csr_matrix:
    """Return W^T (terms x docs) with BM25 per-(doc, term) weights; scores = Q_binary @ W^T."""
    tf = tf.tocoo()
    n_docs = tf.shape[0]
    df = np.bincount(tf.col, minlength=tf.shape[1])
    idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    dl = np.bincount(tf.row, weights=tf.data, minlength=n_docs).astype(np.float32)
    avg = max(float(dl.mean()), 1e-6)
    denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avg)
    w = sp.csr_matrix(((tf.data * (k1 + 1) / denom * idf[tf.col]).astype(np.float32), (tf.row, tf.col)),
                      shape=tf.shape)
    return w.T.tocsr()


# ----------------------------------------------------------------------------------------------------
# Geo prior (copied from experiments/00_baseline/run.py; to be replaced by agent_4's geo table)
# ----------------------------------------------------------------------------------------------------

class GeoPrior:
    """log P(item_loc | search_loc) from the transition counts in the allowed train rows; own location
    floored at 0.5, unseen transitions at 1e-6 (baseline definition)."""

    def __init__(self, split: str, item_loc: np.ndarray):
        rows = paths.load_train_rows(split, ["search_location_id", "item_location_id"])
        trans = (rows.group_by("search_location_id", "item_location_id").len()
                 .with_columns((pl.col("len") / pl.col("len").sum().over("search_location_id")).alias("p")))
        self.table: dict[int, dict[int, float]] = {}
        for s, i, p in trans.select("search_location_id", "item_location_id", "p").iter_rows():
            self.table.setdefault(s, {})[i] = p
        self.locs, self.inverse = np.unique(item_loc, return_inverse=True)
        self.cache: dict[int, np.ndarray] = {}

    def __call__(self, search_loc: int) -> np.ndarray:
        v = self.cache.get(search_loc)
        if v is None:
            probs = self.table.get(search_loc, {})
            p = np.array([probs.get(int(l), 1e-6) for l in self.locs], dtype=np.float32)
            own = self.locs == search_loc
            p[own] = np.maximum(p[own], 0.5)
            v = np.log(p)[self.inverse].astype(np.float32)
            if len(self.cache) > 400:
                self.cache.clear()
            self.cache[search_loc] = v
        return v


# ----------------------------------------------------------------------------------------------------
# Queries, labels, evaluation
# ----------------------------------------------------------------------------------------------------

class Split:
    """Queries of a split, relevant item indices (if labelled) and benchmark weights."""

    def __init__(self, split: str, item_index: dict[str, int]):
        self.name = split
        q = paths.load_queries(split)
        self.qids = q["query_id"].to_list()
        self.texts = q["search_query"].fill_null("").to_list()
        self.filters = q["search_infm_params_text"].fill_null("").to_list()
        self.locs = q["search_location_id"].to_list()
        self.cats = q["search_category"].to_list()
        self.rel: list[np.ndarray] | None = None
        self.w = np.ones(len(self.qids))
        if split != "bench":
            qrels = paths.load_qrels(split).group_by("query_id").agg(pl.col("item_id"))
            m = {qid: np.array([item_index[i] for i in items], dtype=np.int64) for qid, items in qrels.iter_rows()}
            self.rel = [m.get(qid, np.zeros(0, dtype=np.int64)) for qid in self.qids]
            meta = paths.load_meta(split)
            meta = pl.DataFrame({"query_id": self.qids}).join(meta, on="query_id", how="left")
            self.w = evaluate.bench_weights(meta)

    def __len__(self) -> int:
        return len(self.qids)


def ranks_of_relevant(scores: np.ndarray, rel: list[np.ndarray]) -> list[np.ndarray]:
    """For each row, expected rank (1 = best) of every relevant item under random tie-breaking:
    1 + #items scoring strictly higher + #other items with an equal score / 2. Ties are frequent (items with
    no text match share the same geo score), and the final top-k breaks them by item_id, i.e. arbitrarily."""
    out = []
    for row, r in zip(scores, rel):
        if len(r) == 0:
            out.append(np.zeros(0, dtype=np.float64))
            continue
        s = row[r]
        higher = (row[None, :] > s[:, None]).sum(axis=1)
        equal = (row[None, :] == s[:, None]).sum(axis=1) - 1
        out.append(1 + higher + equal / 2)
    return out


def recall_from_ranks(ranks: list[np.ndarray], k: int, w: np.ndarray | None = None) -> float:
    per_q = np.array([(r <= k).mean() if len(r) else 0.0 for r in ranks])
    if w is None:
        return float(per_q.mean())
    return float((per_q * w).sum() / w.sum())


def topk(scores: np.ndarray, k: int, ids: np.ndarray) -> list[np.ndarray]:
    """Top-k corpus indices per row, ordered by score desc then item_id (deterministic)."""
    out = []
    for row in scores:
        idx = np.argpartition(-row, k)[:k]
        idx = idx[np.lexsort((ids[idx], -row[idx]))]
        out.append(idx)
    return out


def write_answer(split: Split, top: list[np.ndarray], ids: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ans = [" ".join(ids[t[:50]]) for t in top]
    assert all(len(set(a.split(" "))) == 50 for a in ans), "answers must contain 50 unique ids"
    pl.DataFrame({"query_id": split.qids, "answer": ans}).write_csv(path)


def channel_frame(split: Split, top: list[np.ndarray], scores_top: list[np.ndarray], ids: np.ndarray) -> pl.DataFrame:
    frames = []
    for qid, t, s in zip(split.qids, top, scores_top):
        frames.append(pl.DataFrame({"query_id": [qid] * len(t), "item_id": ids[t], "score": s.astype(np.float32),
                                    "rank": np.arange(1, len(t) + 1, dtype=np.int32)}))
    return pl.concat(frames)


# ----------------------------------------------------------------------------------------------------
# BM25F: term frequencies are combined across fields (weighted, per-field length-normalised) *before* the
# k1 saturation, so a term that appears in several fields is not rewarded several times.
# ----------------------------------------------------------------------------------------------------

class BM25F:
    def __init__(self, an: "Analyzed", fields: tuple[str, ...]):
        self.an = an
        self.tf = {f: an.tf(FIELD_PARTS[f]) for f in fields}
        self.dl = {f: np.asarray(m.sum(axis=1)).ravel().astype(np.float32) for f, m in self.tf.items()}
        self.avg = {f: max(float(v.mean()), 1e-6) for f, v in self.dl.items()}

    def weights(self, fw: dict[str, float], fb: dict[str, float], k1: float) -> sp.csr_matrix:
        """W^T (terms x docs). fw = field weights, fb = per-field length normalisation b."""
        T = None
        for f, w in fw.items():
            if not w:
                continue
            norm = 1 - fb[f] + fb[f] * self.dl[f] / self.avg[f]
            m = sp.diags((w / norm).astype(np.float32)) @ self.tf[f]
            T = m if T is None else T + m
        T = T.tocsr()
        T.sum_duplicates()
        n_docs = T.shape[0]
        df = np.bincount(T.indices, minlength=T.shape[1])
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        data = T.data * (k1 + 1) / (T.data + k1) * idf[T.indices]
        W = sp.csr_matrix((data.astype(np.float32), T.indices, T.indptr), shape=T.shape)
        return W.T.tocsr()
