"""Offline reproduction of the frozen Avito e3_four solution.

The prediction path never reads answer.csv, reference answers or val/test labels.
It rebuilds a union of retrieval candidates, extracts 58 pair features from the
inputs and saved embeddings, and applies three trained LightGBM checkpoints.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(v, "3")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import hashlib
import importlib.metadata as md
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unpack(runtime: Path) -> None:
    """Reconstruct local files, validating hashes; no downloads or symlinks."""
    runtime.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "bundle_manifest.json").read_text())
    rebuilt = 0
    for record in manifest["files"]:
        target = runtime / record["path"]
        if not target.resolve().is_relative_to(runtime.resolve()):
            raise ValueError(f"Invalid artifact path: {record['path']}")
        if target.exists() and target.stat().st_size == record["bytes"] and file_sha(target) == record["sha256"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        h = hashlib.sha256()
        with partial.open("wb") as stream:
            for part in record["parts"]:
                source = ROOT / part["path"]
                if not source.resolve().is_relative_to((ROOT / "bundle").resolve()):
                    raise ValueError(f"Invalid bundle part: {part['path']}")
                block = source.read_bytes()
                if len(block) != part["bytes"] or hashlib.sha256(block).hexdigest() != part["sha256"]:
                    raise ValueError(f"Corrupt or incomplete bundle part: {source}")
                stream.write(block)
                h.update(block)
        if partial.stat().st_size != record["bytes"] or h.hexdigest() != record["sha256"]:
            raise ValueError(f"Incorrect reconstructed file: {record['path']}")
        partial.replace(target)
        rebuilt += 1
    print(f"Artifacts verified; reconstructed {rebuilt} files in {runtime}", flush=True)


def configure(runtime: Path) -> dict:
    release = json.loads((ROOT / "release.json").read_text())
    for name, expected in release["runtime_dependencies"].items():
        actual = md.version(name)
        if actual != expected:
            raise RuntimeError(f"Expected {name}=={expected}, found {actual}; install requirements.txt for exact reproduction")
    unpack(runtime)
    # Explicit root prevents the canonical loader from finding the surrounding
    # development checkout through Git or a user-provided AVITO_ROOT variable.
    os.environ["AVITO_ROOT"] = str(runtime.resolve())
    from avito_bootcamp_2026 import paths
    paths.main_root.cache_clear()
    return release


def build_pool(split: str, release: dict):
    """Frozen source order and quotas; channel discovery is deliberately absent."""
    import polars as pl
    from avito_bootcamp_2026 import exchange
    parts = []
    channels = [name.split("/", 1) for name in release["channels"]]
    for (owner, name), quota in zip(channels, release["pool_quotas"]):
        ch = exchange.load_channel(owner, name, split)
        parts.append(ch.filter(pl.col("rank") <= quota).select("query_id", "item_id"))
    keys = pl.concat(parts).unique(subset=["query_id", "item_id"]).sort("query_id", "item_id")
    for j, (owner, name) in enumerate(channels):
        ch = exchange.load_channel(owner, name, split).select(
            "query_id", "item_id", pl.col("score").alias(f"c{j}_score"), pl.col("rank").alias(f"c{j}_rank"))
        keys = keys.join(ch, on=["query_id", "item_id"], how="left")
    ranks = [f"c{j}_rank" for j in range(len(channels))]
    keys = keys.with_columns(pl.sum_horizontal([1 / (60 + pl.col(c).fill_null(2001)) for c in ranks]).alias("rrf"))
    keys = (keys.sort("query_id", "rrf", "item_id", descending=[False, True, False])
            .with_columns((pl.int_range(pl.len()).over("query_id") + 1).alias("pool_rank"))
            .filter(pl.col("pool_rank") <= release["max_pool"]).drop("rrf", "pool_rank"))
    print(f"{split}: {keys.height:,} union candidates", flush=True)
    return keys


def make_frame(split: str, release: dict, item_state=None):
    from avito_solution.features import item_data, make_features
    if item_state is None:
        item_state = item_data()
    cols = [f"c{j}_{part}" for j in range(len(release["channels"])) for part in ("score", "rank")]
    return make_features(split, build_pool(split, release), *item_state, cols)


def model_paths(release: dict, alternate: Path | None = None) -> list[Path]:
    paths = [ROOT / name for name in release["models"]]
    return [alternate / p.name for p in paths] if alternate else paths


def predict(split: str, target: Path, release: dict, item_state=None, alternate: Path | None = None) -> dict:
    import numpy as np
    import polars as pl
    from avito_bootcamp_2026 import evaluate, paths
    from avito_solution.ranker import predict_model
    frame = make_frame(split, release, item_state)
    cols = [c for c in frame.columns if c not in ("query_id", "item_id", "label")]
    scores = np.zeros(frame.height, dtype=np.float32)
    models = model_paths(release, alternate)
    for model in models:
        scores += predict_model(model, frame.select(cols)) / len(models)
    ranked = (frame.select("query_id", "item_id").with_columns(pl.Series("score", scores))
              .sort("query_id", "score", "item_id", descending=[False, True, False])
              .with_columns((pl.int_range(pl.len()).over("query_id") + 1).cast(pl.Int32).alias("rank")))
    answer = (ranked.filter(pl.col("rank") <= release["topk"]).group_by("query_id", maintain_order=True)
              .agg(pl.col("item_id").str.join(" ").alias("answer")))
    answer = paths.load_queries(split).select("query_id").join(answer, on="query_id", how="left")
    problems = evaluate.check_answer(answer, set(paths.load_queries(split)["query_id"]),
                                    set(paths.load_corpus(["item_id"])["item_id"]), strict=True)
    if problems:  # This solution always returns exactly 50 unique valid IDs.
        raise ValueError("; ".join(problems))
    target.parent.mkdir(parents=True, exist_ok=True)
    answer.write_csv(target)
    result = {"split": split, "queries": answer.height, "pairs": frame.height,
              "features": len(cols), "sha256": file_sha(target), "file": str(target.resolve())}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def verify(runtime: Path, release: dict, alternate: Path | None = None) -> None:
    from avito_bootcamp_2026 import evaluate
    from avito_solution.features import item_data
    report = {"release": release["release"], "external_api_calls": 0,
              "method": "rebuild candidate union + extract features + run frozen 3-seed ensemble",
              "splits": {}, "seconds": 0}
    started = time.monotonic()
    state = item_data()
    for split in ("bench", "val", "test"):
        reference = ROOT / "answer.csv" if split == "bench" else ROOT / "verification" / f"reference_{split}.csv"
        if file_sha(reference) != release["reference_sha256"][split]:
            raise ValueError(f"Reference file has changed: {reference}")
        output = runtime / "generated" / ("answer.csv" if split == "bench" else f"{split}_answer.csv")
        result = predict(split, output, release, state, alternate)
        result["identical_bytes"] = result["sha256"] == release["reference_sha256"][split]
        if not result["identical_bytes"]:
            raise AssertionError(f"{split} answer differs from the frozen release")
        if split != "bench":
            # Use the unchanged shared harness for the quality check. Validation
            # and test labels are read only here, after predictions are complete.
            summary = evaluate.summarize(evaluate.per_query_recall(
                evaluate.to_lists(evaluate.read_answer(output)), split), split)
            evaluate.print_summary(summary)
            if round(summary["recall"], 4) != release["expected_recall"][split]:
                raise AssertionError(f"Incorrect {split} Recall@50: {summary['recall']}")
            result["recall_at_50"] = summary["recall"]
            result["ci95"] = summary["ci95"]
            evaluate.log_result({"exp": f"agent_2/package_reproduce_{split}", "split": split,
                                 "kind": "answer", "recall": round(summary["recall"], 4),
                                 "file": str(output.resolve()), "file_sha": result["sha256"][:16],
                                 "note": "identical release reproduction; no model selection"})
        report["splits"][split] = result
    report["seconds"] = round(time.monotonic() - started, 2)
    report_path = runtime / "verification_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"All three CSV files are byte-identical; report: {report_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime-dir", type=Path, default=ROOT / "runtime")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="Regenerate predictions without reading reference answers")
    p.add_argument("--split", choices=("bench", "val", "test"), default="bench")
    p.add_argument("--output", type=Path, default=ROOT / "answer.csv")
    p.add_argument("--models-dir", type=Path)
    p = sub.add_parser("verify", help="Recompute all three answer files and compare hashes and metrics")
    p.add_argument("--models-dir", type=Path)
    p = sub.add_parser("check", help="Check a benchmark answer file using the shared harness")
    p.add_argument("--file", type=Path, default=ROOT / "answer.csv")
    sub.add_parser("unpack", help="Only reconstruct and validate input/artifact files")
    p = sub.add_parser("train", help="Rebuild rtrain features and train the same three seeds")
    p.add_argument("--output-dir", type=Path)
    a = ap.parse_args()
    release = configure(a.runtime_dir.resolve())
    if a.command == "predict":
        predict(a.split, a.output, release, alternate=a.models_dir)
    elif a.command == "verify":
        verify(a.runtime_dir.resolve(), release, alternate=a.models_dir)
    elif a.command == "check":
        from avito_bootcamp_2026 import evaluate
        evaluate.cmd_check(argparse.Namespace(file=str(a.file)))
    elif a.command == "train":
        from avito_solution.ranker import fit_model
        output = a.output_dir or a.runtime_dir / "retrained_models"
        output.mkdir(parents=True, exist_ok=True)
        feature_path = a.runtime_dir / "rtrain_features.parquet"
        make_frame("rtrain", release).write_parquet(feature_path, compression="zstd")
        for seed in release["seeds"]:
            fit_model(feature_path, output / f"ranker_{seed}.txt", "binary", "all", False, seed)
        print(f"Trained models: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
