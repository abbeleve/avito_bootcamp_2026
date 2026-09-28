"""Artifact exchange between parallel agents.

Layout: data/shared/<owner>/<artifact>/ with a ``meta.json`` describing how it was produced. An agent
writes only under its own ``<owner>`` folder and may read everybody's artifacts. Three kinds:

* channel    - top-K candidates per query: {rtrain,val,test,bench}.parquet with columns
               query_id (str), item_id (str), score (f32, higher = better), rank (i32, 1 = best), K <= 2000.
               A channel file is also a valid input for ``evaluate pool``.
* embeddings - items.npy (float16, row-aligned with ``paths.load_corpus()``) and {rtrain,val,test,bench}.npy
               (row-aligned with ``paths.load_queries(split)``), L2-normalised -> cosine = dot product.
* table      - any other parquet (per-query, per-item or per-pair features); document the key in meta.json.

The ``fit_only`` flag in meta.json is mandatory: it asserts that the rtrain/val/test artifacts were
produced from hist.parquet only (``paths.load_train_rows``); bench artifacts may use the full train.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import polars as pl

from avito_bootcamp_2026 import paths

SPLITS = ("rtrain", "val", "test", "bench")
CHANNEL_SCHEMA = {"query_id": pl.String, "item_id": pl.String, "score": pl.Float32, "rank": pl.Int32}
MAX_K = 2000


def artifact_dir(owner: str, name: str) -> Path:
    d = paths.shared_dir(owner) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_meta(owner: str, name: str, kind: str, description: str, fit_only: bool, **extra) -> None:
    """Record what an artifact is and how to rebuild it (command, git revision)."""
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        rev = "?"
    meta = {"owner": owner, "name": name, "kind": kind, "description": description, "fit_only": fit_only,
            "git": rev or "no-commit", **extra}
    (artifact_dir(owner, name) / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def save_channel(owner: str, name: str, split: str, df: pl.DataFrame) -> Path:
    """Validate and store a candidate channel for one split."""
    df = df.select(list(CHANNEL_SCHEMA)).cast(CHANNEL_SCHEMA)
    queries = set(paths.load_queries(split)["query_id"])
    assert set(df["query_id"].unique()) <= queries, "channel contains query_ids that are not in this split"
    assert df["item_id"].is_in(paths.load_corpus(["item_id"])["item_id"].implode()).all(), "unknown item_ids"
    per_q = df.group_by("query_id").agg(pl.len().alias("n"), pl.col("rank").min().alias("r0"),
                                        pl.col("item_id").n_unique().alias("u"))
    assert per_q["n"].max() <= MAX_K, f"more than {MAX_K} candidates for a query"
    assert (per_q["n"] == per_q["u"]).all(), "duplicated item_ids within a query"
    assert (per_q["r0"] == 1).all(), "ranks must start at 1 for every query"
    path = artifact_dir(owner, name) / f"{split}.parquet"
    df.sort("query_id", "rank").write_parquet(path)
    return path


def load_channel(owner: str, name: str, split: str) -> pl.DataFrame:
    return pl.read_parquet(paths.shared_dir(owner) / name / f"{split}.parquet")


def save_embeddings(owner: str, name: str, which: str, emb: np.ndarray) -> Path:
    """which = 'items' (corpus order) or a split name (query order of that split). Stored as float16."""
    expected = paths.load_corpus(["item_id"]).height if which == "items" else paths.load_queries(which).height
    assert emb.shape[0] == expected, f"{which}: expected {expected} rows, got {emb.shape[0]}"
    norms = np.linalg.norm(emb.astype(np.float32), axis=1)
    assert np.allclose(norms, 1, atol=1e-2), "embeddings must be L2-normalised"
    path = artifact_dir(owner, name) / f"{which}.npy"
    np.save(path, emb.astype(np.float16))
    return path


def load_embeddings(owner: str, name: str, which: str) -> np.ndarray:
    return np.load(paths.shared_dir(owner) / name / f"{which}.npy", mmap_mode="r")


def list_artifacts() -> pl.DataFrame:
    """Everything published so far (owner, name, kind, fit_only, description)."""
    rows = []
    for meta in sorted(paths.shared_dir().glob("*/*/meta.json")):
        m = json.loads(meta.read_text())
        rows.append({k: m.get(k) for k in ("owner", "name", "kind", "fit_only", "description")})
    return pl.DataFrame(rows) if rows else pl.DataFrame()


if __name__ == "__main__":
    with pl.Config(tbl_rows=100, fmt_str_lengths=80, tbl_width_chars=200):
        print(list_artifacts())
