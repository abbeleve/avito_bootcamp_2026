"""Encode Avito corpus and query variants with a pinned Hugging Face encoder.

Example:
  flock data/gpu.lock ./.venv/bin/python experiments/agent_2/encode.py \
    --model e5_base --items t2 --queries both

The model SHA, seed, text variant and complete command are saved next to the vectors.
The val/test/rtrain query vectors never depend on validation labels or the full train.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(v, "3")

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from avito_bootcamp_2026 import paths
import polars as pl

ROOT = Path(__file__).parent
STORE = ROOT / "artifacts" / "zero"
MODELS = {
    "user_bge_m3": "deepvk/USER-bge-m3",
    "e5_base": "intfloat/multilingual-e5-base",
    "e5_large": "intfloat/multilingual-e5-large",
    "frida": "ai-forever/FRIDA",
    "qwen3_06b": "Qwen/Qwen3-Embedding-0.6B",
    "embeddinggemma": "google/embeddinggemma-300m",
    "user2_base": "deepvk/USER2-base",
}
MODEL_REVISIONS = {
    "user_bge_m3": "0cc6cfe48e260fb0474c753087a69369e88709ae",
    "e5_base": "d128750597153bb5987e10b1c3493a34e5a4502a",
    "e5_large": "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
    "frida": "850455b605544a944739b25f81ddf812b6e3d0d5",
    "qwen3_06b": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "embeddinggemma": "57c266a740f537b4dc058e1b0cda161fd15afa75",
    "user2_base": "94bbff0cc696f26dc93b275fc874657727df0c29",
}
MODEL_LICENSES = {
    "user_bge_m3": "apache-2.0", "e5_base": "mit", "e5_large": "mit",
    "frida": "mit", "qwen3_06b": "apache-2.0", "embeddinggemma": "gemma",
    "user2_base": "apache-2.0",
}

SERVICE_STOP = re.compile(
    r"(?=Место оказания услуг|Тип стоимости|Начальная цена|График работы|Время работы|Стоимость|"
    r"Продолжительность|Опыт работы|Куда выезжаете|Как вы работаете|Где вы оказываете услуги|$)"
)
PARAM_FIELD = re.compile(r"(?:Вид услуги|Тип услуги|Название услуги|Услуга)\s+(.{0,120}?)" + SERVICE_STOP.pattern)


def params_short(raw: str | None) -> str:
    """Simple, dependency-free extraction of service type and price-list names."""
    if not raw:
        return ""
    values = []
    for match in PARAM_FIELD.finditer(raw):
        value = match.group(1).strip(" .,;:")
        if value and value not in {"Своя услуга", "Другое"} and value not in values:
            values.append(value)
        if len(values) >= 8:
            break
    return "; ".join(values)


def item_texts(variant: str) -> list[str]:
    parsed = variant in ("t2p", "t3p")
    cols = ["item_id", "item_title_raw"] if variant == "t1" or parsed else ["item_id", "item_title_raw", "item_infm_params_text"]
    if variant in ("t3", "t3p"):
        cols.append("item_description_raw")
    corpus = paths.load_corpus(cols)
    title = corpus["item_title_raw"].fill_null("").to_list()
    if variant == "t1":
        return title
    if parsed:
        table_path = paths.shared_dir("agent_1") / "items_params" / "items.parquet"
        if not table_path.exists():
            raise FileNotFoundError(f"agent_1 parsed params table is required for {variant}: {table_path}")
        table = pl.read_parquet(table_path, columns=["item_id", "core_text"])
        params = [value[:500] for value in corpus.select("item_id").join(
            table, on="item_id", how="left", maintain_order="left")["core_text"].fill_null("").to_list()]
    else:
        params = [params_short(v) for v in corpus["item_infm_params_text"].to_list()]
    if variant in ("t2", "t2p"):
        return [f"{t}. {p}" if p else t for t, p in zip(title, params)]
    desc = corpus["item_description_raw"].fill_null("").to_list()
    return [f"{t}. {p}. {d[:300]}" for t, p, d in zip(title, params, desc)]


def query_texts(split: str, variant: str) -> list[str]:
    queries = paths.load_queries(split)
    plain = queries["search_query"].fill_null("").to_list()
    if variant == "plain":
        return plain
    filters = queries["search_infm_params_text"].fill_null("").to_list()
    texts = ([f"{q}. {f}" if f else q for q, f in zip(plain, filters)]
             if variant in ("filters", "filters_city") else plain)
    if variant in ("city", "filters_city"):
        table_path = paths.shared_dir("agent_1") / "items_params" / "items.parquet"
        places = pl.read_parquet(table_path, columns=["item_id", "place_city"])
        corpus_loc = paths.load_corpus(["item_id", "item_location_id"])
        modes = (corpus_loc.join(places, on="item_id", how="left")
                 .filter(pl.col("place_city").is_not_null() & (pl.col("place_city") != ""))
                 .group_by("item_location_id", "place_city").len()
                 .sort("item_location_id", "len", "place_city", descending=[False, True, False])
                 .unique(subset=["item_location_id"], keep="first", maintain_order=True))
        city_by_loc = dict(modes.select("item_location_id", "place_city").iter_rows())
        locs = queries["search_location_id"].to_list()
        texts = [f"{text}. Город: {city_by_loc[loc]}" if loc in city_by_loc else text
                 for text, loc in zip(texts, locs)]
    return texts


def prompt(model: str, text: str, is_query: bool) -> str:
    if model.startswith("e5_"):
        return ("query: " if is_query else "passage: ") + text
    if model == "frida":
        return ("search_query: " if is_query else "search_document: ") + text
    if model == "qwen3_06b" and is_query:
        return "Instruct: Retrieve Russian service ads that satisfy the user search.\nQuery: " + text
    return text


def encode(model: SentenceTransformer, texts: list[str], kind: str, model_name: str, batch_size: int) -> np.ndarray:
    texts = [prompt(model_name, text, kind != "items") for text in texts]
    vectors = model.encode(
        texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True,
        normalize_embeddings=True, precision="float32",
    )
    return vectors.astype(np.float16)


def encode_to_file(model: SentenceTransformer, texts: list[str], kind: str, model_name: str,
                   batch_size: int, chunk_size: int, out: Path) -> None:
    """Write chunks durably; an interrupted CUDA job resumes at the last full chunk."""
    partial = out.with_suffix(".partial.npy")
    progress = out.with_suffix(".progress.json")
    if partial.exists() and progress.exists():
        state = json.loads(progress.read_text())
        if state["n_total"] != len(texts):
            raise ValueError(f"Cannot resume {partial}: input length changed")
        done = state["n_done"]
        matrix = np.lib.format.open_memmap(partial, mode="r+")
        print(f"Resuming {out}: {done}/{len(texts)}", flush=True)
    else:
        done = 0
        matrix = None
    for start in range(done, len(texts), chunk_size):
        end = min(start + chunk_size, len(texts))
        vectors = encode(model, texts[start:end], kind, model_name, batch_size)
        if matrix is None:
            matrix = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16,
                                               shape=(len(texts), vectors.shape[1]))
        matrix[start:end] = vectors
        matrix.flush()
        progress.write_text(json.dumps({"n_total": len(texts), "n_done": end}))
        print(f"{out}: {end}/{len(texts)}", flush=True)
    del matrix
    os.replace(partial, out)
    progress.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--items", choices=["t1", "t2", "t3", "t2p", "t3p", "all", "allp", "none"], default="t2")
    ap.add_argument("--queries", choices=["both", "plain", "filters", "city", "filters_city", "none"], default="both")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--run-name", help="embedding cache name; required for fine-tuned checkpoints")
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run this command outside the sandbox under data/gpu.lock")

    if args.checkpoint and not args.run_name:
        raise ValueError("--checkpoint requires --run-name to keep fine-tuned vectors separate")
    base = STORE / (args.run_name or args.model)
    base.mkdir(parents=True, exist_ok=True)
    hf_cache = ROOT / "artifacts" / "hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    model_id = args.checkpoint or MODELS[args.model]
    if args.checkpoint:
        revision = "local-checkpoint"
        license_name = "derived-from-" + MODELS[args.model]
    else:
        # Exact SHAs and licenses were fetched from the Hugging Face API on 2026-09-28.
        # Pinning them here also avoids a network request for every text variant.
        revision = MODEL_REVISIONS[args.model]
        license_name = MODEL_LICENSES[args.model]
    print(f"Loading {model_id} at {revision}", flush=True)
    model = SentenceTransformer(model_id, revision=None if args.checkpoint else revision,
                                device="cuda", cache_folder=str(hf_cache), trust_remote_code=True,
                                model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = args.max_seq_length
    model.eval()
    item_variants = (["t1", "t2", "t3"] if args.items == "all" else
                     ["t1", "t2p", "t3p"] if args.items == "allp" else
                     [] if args.items == "none" else [args.items])
    for variant in item_variants:
        out = base / variant / "items.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            texts = item_texts(variant)
            print(f"Encoding {len(texts)} items, {variant}", flush=True)
            encode_to_file(model, texts, "items", args.model, args.batch_size, args.chunk_size, out)
            print(f"Saved {out}", flush=True)
    variants = ["plain", "filters"] if args.queries == "both" else ([] if args.queries == "none" else [args.queries])
    for variant in variants:
        for split in ("rtrain", "val", "test", "bench"):
            out = base / "queries" / variant / f"{split}.npy"
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                continue
            texts = query_texts(split, variant)
            print(f"Encoding {len(texts)} queries, {variant}/{split}", flush=True)
            encode_to_file(model, texts, "queries", args.model, args.batch_size, args.chunk_size, out)
            print(f"Saved {out}", flush=True)
    meta = {"model_id": MODELS[args.model], "revision": revision, "license": license_name,
            "checkpoint": args.checkpoint, "seed": 42, "batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length, "parsed_core_text_max_chars": 500,
            "chunk_size": args.chunk_size,
            "command": " ".join(sys.argv), "torch": torch.__version__,
            "sentence_transformers": __import__("sentence_transformers").__version__}
    (base / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
