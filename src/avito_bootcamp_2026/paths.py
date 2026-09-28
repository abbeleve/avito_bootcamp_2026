"""Locations of the shared data and canonical loaders.

Every experiment (and every parallel agent, possibly working in its own git worktree) must load data
through this module, so that all of them read the *same* frozen split and write to the *same*
results log.

The data root is the main checkout of the repository (where the three raw parquet files live). It is
resolved in this order:
1. the ``AVITO_ROOT`` environment variable;
2. the parent of git's *common* directory - the same for the main checkout and all its worktrees;
3. the first parent directory of this file that contains ``benchmark_items.parquet``.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Literal

import polars as pl

SPLIT_VERSION = "v1"
Split = Literal["rtrain", "val", "test", "bench"]
RAW_FILES = ("train.parquet", "benchmark_queries.parquet", "benchmark_items.parquet")


@lru_cache(maxsize=1)
def main_root() -> Path:
    """Directory that holds the raw parquet files (shared by all worktrees)."""
    env = os.environ.get("AVITO_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve().parent
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=here, capture_output=True, text=True, check=True,
        ).stdout.strip()
        root = Path(common).parent
        if (root / RAW_FILES[2]).exists():
            return root
    except (OSError, subprocess.CalledProcessError):
        pass
    for parent in (here, *here.parents):
        if (parent / RAW_FILES[2]).exists():
            return parent
    raise FileNotFoundError("Cannot find the raw data; set AVITO_ROOT to the directory with the parquet files")


def data_dir() -> Path:
    """Shared, git-ignored directory for derived data (splits, results log, exchanged candidates)."""
    return main_root() / "data"


def split_dir(version: str = SPLIT_VERSION) -> Path:
    return data_dir() / "splits" / version


def results_log() -> Path:
    """Append-only JSONL log of every evaluation made through ``avito_bootcamp_2026.evaluate``."""
    return data_dir() / "results.jsonl"


def gpu_lock() -> Path:
    """Lock file: wrap every CUDA job in ``flock <this path> ...`` so parallel agents take turns on the GPU."""
    return data_dir() / "gpu.lock"


def shared_dir(owner: str | None = None) -> Path:
    """Exchange area data/shared/<owner>/: each agent writes only its own folder and may read all others."""
    base = data_dir() / "shared"
    return base / owner if owner else base


# ----------------------------------------------------------------------------------------------------
# Loaders. Use these instead of reading parquet files directly: they enforce the split protocol.
# ----------------------------------------------------------------------------------------------------

def load_corpus(columns: list[str] | None = None) -> pl.DataFrame:
    """The benchmark corpus (189,212 items). It is the retrieval corpus for val, test *and* bench."""
    return pl.read_parquet(main_root() / "benchmark_items.parquet", columns=columns)


def load_queries(split: Split) -> pl.DataFrame:
    """Queries of a split in exactly the schema of ``benchmark_queries.parquet``."""
    if split == "bench":
        return pl.read_parquet(main_root() / "benchmark_queries.parquet")
    return pl.read_parquet(split_dir() / f"{split}_queries.parquet")


def load_train_rows(for_split: Split, columns: list[str] | None = None) -> pl.DataFrame:
    """Rows a model is allowed to learn from when it predicts ``for_split``.

    * rtrain / val / test -> ``hist.parquet``: train minus every rtrain/val/test search, every row that
      contains one of their relevant items and every row of a text converted to "unseen";
    * bench               -> the full ``train.parquet``.
    """
    if for_split == "bench":
        return pl.read_parquet(main_root() / "train.parquet", columns=columns)
    if for_split not in ("rtrain", "val", "test"):
        raise ValueError(f"unknown split {for_split!r}")
    if columns is not None and "train_row_id" not in columns:
        columns = [*columns, "train_row_id"]
    return pl.read_parquet(split_dir() / "hist.parquet", columns=columns)


def load_qrels(split: Literal["rtrain", "val", "test"]) -> pl.DataFrame:
    """Relevant (query_id, item_id) pairs. rtrain qrels are ranker training labels; val/test are for scoring only."""
    return pl.read_parquet(split_dir() / f"{split}_qrels.parquet")


def load_meta(split: Split) -> pl.DataFrame:
    """Per-query segment labels (location type, filters, seen text, ...) for error analysis."""
    return pl.read_parquet(split_dir() / f"{split}_meta.parquet")
