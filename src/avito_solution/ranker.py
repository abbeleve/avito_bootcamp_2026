"""LightGBM ranker training with deterministic negative sampling and optional raking."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(v, "3")

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from avito_bootcamp_2026 import evaluate, paths


def _features(frame: pl.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ("query_id", "item_id", "label")]


def fit_model(features_path: Path, model_path: Path, objective: str, sampling: str,
              rake: bool, seed: int, exclude: set[str] | None = None) -> None:
    frame = pl.read_parquet(features_path)
    positives = int(frame["label"].sum())
    print(f"rtrain pool: {frame.height:,} candidates, {positives} positives", flush=True)
    if sampling == "top_random":
        rng = np.random.default_rng(seed)
        # Baseline supplies 1000 candidates per query. Keep its top 300 and 100 of the rest.
        rank = frame["c0_rank"].to_numpy()
        keep = (rank <= 300) | (rng.random(frame.height) < 100 / 700) | (frame["label"].to_numpy() == 1)
        frame = frame.filter(pl.Series(keep))
    frame = frame.sort("query_id")
    cols = [c for c in _features(frame) if c not in (exclude or set())]
    labels = frame["label"].to_numpy().astype(np.float32)
    qids = frame["query_id"].to_numpy()
    qmeta = paths.load_meta("rtrain").select("query_id")
    query_weights = dict(zip(qmeta["query_id"].to_list(), evaluate.bench_weights(paths.load_meta("rtrain")))) if rake else {}
    weights = np.array([query_weights.get(q, 1.0) for q in qids], np.float32)
    # Equal total positive weight per query; positives should estimate marginal relevance.
    qrel_counts = dict(paths.load_qrels("rtrain").group_by("query_id").len().iter_rows())
    weights *= np.where(labels > 0, np.array([1 / qrel_counts.get(q, 1) for q in qids], np.float32) * 60, 1)
    X = frame.select(cols).to_numpy().astype(np.float32)
    X = np.nan_to_num(X, nan=-999, posinf=999, neginf=-999)
    ds_args = {"data": X, "label": labels, "weight": weights, "feature_name": cols,
               "free_raw_data": True}
    if objective == "lambdarank":
        _, counts = np.unique(qids, return_counts=True)
        ds_args["group"] = counts.tolist()
    data = lgb.Dataset(**ds_args)
    params = {"objective": objective, "verbosity": -1, "num_threads": 3, "seed": seed,
              "learning_rate": 0.055, "num_leaves": 31, "min_data_in_leaf": 90,
              "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1,
              "max_bin": 127, "lambda_l2": 5.0}
    if objective == "lambdarank":
        params.update(metric="ndcg", ndcg_eval_at=[50], lambdarank_truncation_level=100)
    else:
        params["metric"] = "binary_logloss"
    print(f"training {objective}/{sampling}/rake={rake}/seed={seed} on {len(labels):,} pairs", flush=True)
    model = lgb.train(params, data, num_boost_round=200)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    print(f"saved {model_path}", flush=True)
    for name, value in sorted(zip(cols, model.feature_importance(importance_type="gain")), key=lambda x: -x[1])[:15]:
        print(f"  {name}: {value:.0f}", flush=True)


def predict_model(model_path: Path, frame: pl.DataFrame) -> np.ndarray:
    model = lgb.Booster(model_file=str(model_path))
    X = frame.select(model.feature_name()).to_numpy().astype(np.float32)
    X = np.nan_to_num(X, nan=-999, posinf=999, neginf=-999)
    return model.predict(X, num_threads=3).astype(np.float32)
