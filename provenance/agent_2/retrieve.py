"""Exact dense search with the shared, hist-fitted geographic prior.

Every scored val/test result is sent to the shared evaluation harness. The same
script publishes row-aligned embeddings and top-K channels for other agents.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(v, "3")

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

from avito_bootcamp_2026 import exchange, paths

ROOT = Path(__file__).parent
ZERO = ROOT / "artifacts" / "zero"
OUT = ROOT / "out"


def geo_log_prior(train: pl.DataFrame, item_loc: np.ndarray):
    trans = (train.group_by("search_location_id", "item_location_id").len()
             .with_columns((pl.col("len") / pl.col("len").sum().over("search_location_id")).alias("p")))
    table: dict[int, dict[int, float]] = {}
    for s, i, p in trans.select("search_location_id", "item_location_id", "p").iter_rows():
        table.setdefault(s, {})[i] = p
    unique_locs, inverse = np.unique(item_loc, return_inverse=True)
    cache: dict[int, np.ndarray] = {}

    def get(search_loc: int) -> np.ndarray:
        if search_loc not in cache:
            probs = table.get(search_loc, {})
            by_loc = np.array([probs.get(int(loc), 1e-6) for loc in unique_locs], dtype=np.float32)
            by_loc[unique_locs == search_loc] = np.maximum(by_loc[unique_locs == search_loc], 0.5)
            cache[search_loc] = np.log(by_loc)[inverse]
        return cache[search_loc]
    return get


def geo_v2_prior(split: str, item_loc: np.ndarray):
    """Agent_4's smoothed geo prior with its per-item city-density correction."""
    table_df = pl.read_parquet(paths.shared_dir("agent_4") / "geo" / f"{split}.parquet",
                               columns=["search_location_id", "item_location_id", "log_p"])
    table: dict[int, dict[int, float]] = {}
    min_log: dict[int, float] = {}
    for search_loc, loc, log_p in table_df.iter_rows():
        table.setdefault(search_loc, {})[loc] = log_p
        min_log[search_loc] = min(min_log.get(search_loc, float("inf")), log_p)
    unique_locs, inverse, counts = np.unique(item_loc, return_inverse=True, return_counts=True)
    density = 0.5 * np.log(counts.astype(np.float32))
    cache: dict[int, np.ndarray] = {}

    def get(search_loc: int) -> np.ndarray:
        if search_loc not in cache:
            probs = table.get(search_loc, {})
            floor = min_log.get(search_loc, -16.118095)
            by_loc = np.array([probs.get(int(loc), floor) for loc in unique_locs], dtype=np.float32)
            cache[search_loc] = (by_loc - density)[inverse]
        return cache[search_loc]
    return get


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", default="t2", choices=["t1", "t2", "t3", "t2p", "t3p"])
    ap.add_argument("--query", default="filters", choices=["plain", "filters", "city", "filters_city"])
    ap.add_argument("--alpha", type=float, default=0.04)
    ap.add_argument("--geo", choices=["v1", "v2"], default="v1")
    ap.add_argument("--name", required=True)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--splits", nargs="+", default=["val"], choices=exchange.SPLITS)
    ap.add_argument("--score", action="store_true", help="run shared answer/pool harness for val")
    ap.add_argument("--score-test", action="store_true", help="explicitly score a test release candidate")
    ap.add_argument("--publish-name", help="publish channel under agent_2/<name>")
    ap.add_argument("--publish-embeddings", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run outside sandbox under data/gpu.lock")
    if not (0 < args.k <= 2000):
        raise ValueError("k must be 1..2000")

    base = ZERO / args.model
    item_vectors = np.load(base / args.items / "items.npy", mmap_mode="r")
    item_tensor = torch.from_numpy(np.asarray(item_vectors)).to("cuda", dtype=torch.float16)
    corpus = paths.load_corpus(["item_id", "item_location_id"])
    ids = np.array(corpus["item_id"].to_list())
    item_loc = corpus["item_location_id"].to_numpy()
    OUT.mkdir(exist_ok=True)
    run_dir = ROOT / "artifacts" / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.publish_embeddings:
        if not args.publish_name:
            raise ValueError("--publish-embeddings requires --publish-name")
        exchange.save_embeddings("agent_2", args.publish_name + "_emb", "items", np.asarray(item_vectors))

    for split in args.splits:
        queries = paths.load_queries(split)
        qvec = np.load(base / "queries" / args.query / f"{split}.npy", mmap_mode="r")
        if args.publish_embeddings:
            exchange.save_embeddings("agent_2", args.publish_name + "_emb", split, np.asarray(qvec))
        geo = (geo_v2_prior(split, item_loc) if args.geo == "v2" else
               geo_log_prior(paths.load_train_rows(split, ["search_location_id", "item_location_id"]), item_loc))
        qids = queries["query_id"].to_list()
        locs = queries["search_location_id"].to_list()
        answers = []
        pools = []
        for start in range(0, len(qids), 32):
            q_tensor = torch.from_numpy(np.asarray(qvec[start:start + 32])).to("cuda", dtype=torch.float16)
            sims = (q_tensor @ item_tensor.T).float().cpu().numpy()
            for j, raw in enumerate(sims):
                qi = start + j
                scores = raw + args.alpha * geo(locs[qi]) if args.alpha else raw
                top = np.argpartition(-scores, args.k - 1)[:args.k]
                top = top[np.lexsort((ids[top], -scores[top]))]
                answers.append(" ".join(ids[top[:50]]))
                pools.append(pl.DataFrame({
                    "query_id": qids[qi], "item_id": ids[top], "score": scores[top].astype(np.float32),
                    "rank": np.arange(1, args.k + 1, dtype=np.int32),
                }))
            if start % 512 == 0:
                print(f"{split}: {min(start + 32, len(qids))}/{len(qids)}", flush=True)
        answer_path = OUT / f"{split}_{args.name}.csv"
        pool_path = run_dir / f"{split}.parquet"
        pl.DataFrame({"query_id": qids, "answer": answers}).write_csv(answer_path)
        pool = pl.concat(pools)
        pool.write_parquet(pool_path)
        if args.publish_name:
            exchange.save_channel("agent_2", args.publish_name, split, pool)
        if (args.score and split == "val") or (args.score_test and split == "test"):
            common = [sys.executable, "-m", "avito_bootcamp_2026.evaluate"]
            exp = f"agent_2/{args.name}"
            subprocess.run(common + ["answer", "--split", split, "--file", str(answer_path),
                                     "--exp", exp, "--note", f"{args.model} {args.items} {args.query} alpha={args.alpha} geo={args.geo}"], check=True)
            subprocess.run(common + ["pool", "--split", split, "--file", str(pool_path),
                                     "--exp", exp + "_pool"], check=True)
    if args.publish_name:
        description = f"{args.model} {args.items}/{args.query} cosine + {args.alpha} {args.geo} geo prior, top-{args.k}"
        exchange.write_meta("agent_2", args.publish_name, "channel", description, fit_only=True,
                            command=" ".join(sys.argv), alpha=args.alpha, geo=args.geo, k=args.k)
        if args.publish_embeddings:
            exchange.write_meta("agent_2", args.publish_name + "_emb", "embeddings",
                                f"L2-normalised embeddings, {args.model} {args.items}/{args.query}",
                                fit_only=True, command=" ".join(sys.argv), model_meta=str(base / "meta.json"))
    (run_dir / "meta.json").write_text(json.dumps({"command": " ".join(sys.argv), "model_meta": str(base / "meta.json")}, indent=2))


if __name__ == "__main__":
    main()
