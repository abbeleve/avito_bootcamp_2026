"""Build a channel union, extract features, train a ranker and publish top-50 answers.

    .venv/bin/python experiments/agent_3/pipeline.py --mode all

Re-running the command discovers newly published channels automatically. Artifacts and cached
features are kept under this agent's folders; no labels from val/test enter training.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(v, "3")

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from avito_bootcamp_2026 import exchange, paths
from features import item_data, make_features
from train_ranker import fit_model, predict_model

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
OUT = ROOT / "out"
SPLITS = ("rtrain", "val", "test", "bench")


def channels() -> list[tuple[str, str, Path]]:
    found = [("orchestrator", "baseline_bm25_geo", paths.shared_dir("orchestrator") / "baseline_bm25_geo")]
    for meta in sorted(paths.shared_dir().glob("*/*/meta.json")):
        cfg = json.loads(meta.read_text())
        if cfg.get("kind") != "channel" or cfg.get("owner") == "agent_3":
            continue
        candidate = (cfg["owner"], cfg["name"], meta.parent)
        if candidate not in found and all((meta.parent / f"{s}.parquet").exists() for s in SPLITS):
            found.append(candidate)
    return found


def channel_columns(channels_: list[tuple[str, str, Path]]) -> list[str]:
    return [f"c{j}_{part}" for j in range(len(channels_)) for part in ("score", "rank")]


def build_pool(split: str, channels_: list[tuple[str, str, Path]]) -> pl.DataFrame:
    parts = []
    for j, (owner, name, _) in enumerate(channels_):
        ch = exchange.load_channel(owner, name, split)
        quota = 1000 if j == 0 else min(350, max(100, 1500 // max(1, len(channels_) - 1)))
        parts.append(ch.filter(pl.col("rank") <= quota).select("query_id", "item_id"))
    keys = pl.concat(parts).unique(subset=["query_id", "item_id"]).sort("query_id", "item_id")
    for j, (owner, name, _) in enumerate(channels_):
        ch = exchange.load_channel(owner, name, split).select(
            "query_id", "item_id", pl.col("score").alias(f"c{j}_score"), pl.col("rank").alias(f"c{j}_rank"))
        keys = keys.join(ch, on=["query_id", "item_id"], how="left")
    # Bound the pool to 2000 while retaining all baseline candidates; RRF decides overflow.
    rank_cols = [f"c{j}_rank" for j in range(len(channels_))]
    keys = keys.with_columns(pl.sum_horizontal([(1 / (60 + pl.col(x).fill_null(2001))) for x in rank_cols]).alias("rrf"))
    keys = (keys.sort("query_id", "rrf", "item_id", descending=[False, True, False])
            .with_columns((pl.int_range(pl.len()).over("query_id") + 1).alias("pool_rank"))
            .filter(pl.col("pool_rank") <= 2000).drop("rrf", "pool_rank"))
    print(f"{split}: {keys.height:,} union candidates from {len(channels_)} channels", flush=True)
    return keys


def signature(channels_: list[tuple[str, str, Path]]) -> str:
    sources = [p / "rtrain.parquet" for _, _, p in channels_]
    sources += [paths.shared_dir("agent_1") / "items_params" / "items.parquet",
                paths.shared_dir("agent_1") / "query_filters" / "rtrain.parquet",
                paths.shared_dir("agent_2") / "e5_base_zs_geo_emb" / "items.npy",
                paths.shared_dir("agent_2") / "user_bge_m3_zs_geo_emb" / "items.npy",
                paths.shared_dir("agent_1") / "query_microcat" / "rtrain.parquet",
                paths.shared_dir("agent_4") / "geo" / "rtrain.parquet"]
    s = "|".join(f"{p}:{p.stat().st_mtime_ns if p.exists() else 0}" for p in sources)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def build_features(channels_: list[tuple[str, str, Path]], selected: list[str], refresh: bool) -> dict[str, Path]:
    ART.mkdir(exist_ok=True)
    sig = signature(channels_)
    result = {split: ART / f"features_{sig}_{split}.parquet" for split in selected}
    todo = [s for s in selected if refresh or not result[s].exists()]
    if todo:
        lookup, items = item_data()
        for split in todo:
            pool = build_pool(split, channels_)
            frame = make_features(split, pool, lookup, items, channel_columns(channels_))
            frame.write_parquet(result[split], compression="zstd")
            print(f"{split}: cached {frame.height:,} features in {result[split]}", flush=True)
            del frame, pool
    return result


def write_outputs(split: str, features_path: Path, model_paths: list[Path], tag: str, publish: bool) -> None:
    frame = pl.read_parquet(features_path)
    feature_cols = [c for c in frame.columns if c not in ("query_id", "item_id", "label")]
    scores = np.zeros(frame.height, np.float32)
    for model in model_paths:
        scores += predict_model(model, frame.select(feature_cols)) / len(model_paths)
    ranked = (frame.select("query_id", "item_id").with_columns(pl.Series("score", scores))
              .sort("query_id", "score", "item_id", descending=[False, True, False])
              .with_columns((pl.int_range(pl.len()).over("query_id") + 1).cast(pl.Int32).alias("rank")))
    answers = (ranked.filter(pl.col("rank") <= 50).group_by("query_id", maintain_order=True)
               .agg(pl.col("item_id").str.join(" ").alias("answer")))
    answers = paths.load_queries(split).select("query_id").join(answers, on="query_id", how="left")
    OUT.mkdir(exist_ok=True)
    answers.write_csv(OUT / f"{split}_{tag}_answer.csv")
    if publish:
        exchange.save_channel("agent_3", "pool", split, ranked.select("query_id", "item_id", "score", "rank"))
    print(f"{split}: wrote {answers.height} top-50 answers", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("all", "features", "train", "predict"), default="all")
    ap.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    ap.add_argument("--objective", choices=("binary", "lambdarank"), default="binary")
    ap.add_argument("--sampling", choices=("top_random", "all"), default="top_random")
    ap.add_argument("--rake", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[17])
    ap.add_argument("--tag", default="e1")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--baseline-only", action="store_true", help="freeze E1/E2 comparisons to the baseline channel")
    ap.add_argument("--channel", action="append", help="freeze source list with owner/name; repeat for each source")
    a = ap.parse_args()
    available = channels()
    if a.baseline_only:
        available = available[:1]
    elif a.channel:
        wanted = set(a.channel)
        available = [c for c in available if f"{c[0]}/{c[1]}" in wanted]
        if {f"{c[0]}/{c[1]}" for c in available} != wanted:
            raise ValueError(f"unavailable channels: {wanted - {f'{c[0]}/{c[1]}' for c in available}}")
    print("channels:", [(o, n) for o, n, _ in available], flush=True)
    feature_paths = build_features(available, a.splits if a.mode in ("all", "features") else [], a.refresh)
    if a.mode == "features": return
    sig = signature(available)
    feature_paths = {s: ART / f"features_{sig}_{s}.parquet" for s in SPLITS}
    models = [ART / f"model_{sig}_{a.objective}_{a.sampling}_{int(a.rake)}_{seed}.txt" for seed in a.seeds]
    if a.mode in ("all", "train"):
        if not feature_paths["rtrain"].exists():
            raise FileNotFoundError("Build rtrain features first")
        for seed, model in zip(a.seeds, models):
            fit_model(feature_paths["rtrain"], model, a.objective, a.sampling, a.rake, seed)
    if a.mode == "train": return
    for split in a.splits:
        if not feature_paths[split].exists():
            raise FileNotFoundError(feature_paths[split])
        write_outputs(split, feature_paths[split], models, a.tag, a.publish)
    if a.publish:
        exchange.write_meta("agent_3", "pool", kind="channel", fit_only=True,
                            description=f"Union pool reranked by {a.objective} LightGBM ensemble; sources: {[(o,n) for o,n,_ in available]}",
                            command=f".venv/bin/python experiments/agent_3/pipeline.py --mode all --objective {a.objective} --sampling {a.sampling} --seeds {' '.join(map(str,a.seeds))} --tag {a.tag} --publish " + " ".join(f"--channel {o}/{n}" for o,n,_ in available),
                            source_channels=[f"{o}/{n}" for o,n,_ in available], models=[str(x) for x in models])


if __name__ == "__main__":
    main()
