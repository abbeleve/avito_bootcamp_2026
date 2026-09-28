"""Deterministic pair features for the agent_3 candidate ranker."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(v, "3")

import re
from functools import lru_cache

import numpy as np
import polars as pl
import snowballstemmer

from avito_bootcamp_2026 import paths

TOKEN = re.compile(r"[a-zа-я0-9]+")
STEMMER = snowballstemmer.stemmer("russian")


@lru_cache(maxsize=100000)
def stem(word: str) -> str:
    return STEMMER.stemWord(word)


def tokens(value: str | None) -> frozenset[str]:
    return frozenset(stem(w) for w in TOKEN.findall((value or "").lower().replace("ё", "е")))


def safe_float(col: pl.Series, fill: float = 0.0) -> np.ndarray:
    return col.cast(pl.Float32, strict=False).fill_null(fill).to_numpy()


def item_data() -> tuple[pl.DataFrame, dict[str, np.ndarray | list]]:
    cols = ["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw",
            "item_location_id", "item_microcat_id", "item_category_id", "item_rating_reviews_count",
            "item_rating", "item_price", "item_latitude", "item_longitude", "item_is_phone_hidden",
            "item_is_message_forbidden"]
    items = paths.load_corpus(cols).with_row_index("item_idx")
    title = items["item_title_raw"].fill_null("").to_list()
    params = items["item_infm_params_text"].fill_null("").to_list()
    description = items["item_description_raw"].fill_null("").to_list()
    title_tokens = [tokens(x) for x in title]
    param_tokens = [tokens(x[:600]) for x in params]
    lat = safe_float(items["item_latitude"], np.nan)
    lon = safe_float(items["item_longitude"], np.nan)
    loc = items["item_location_id"].fill_null(-1).to_numpy()
    location_centers = {}
    for key, g in items.select("item_location_id", "item_latitude", "item_longitude").group_by("item_location_id"):
        if key[0] is not None:
            location_centers[int(key[0])] = (float(g["item_latitude"].cast(pl.Float64).median() or np.nan),
                                             float(g["item_longitude"].cast(pl.Float64).median() or np.nan))
    numeric = {
        "item_idx": np.arange(items.height, dtype=np.int32),
        "loc": loc,
        "microcat": items["item_microcat_id"].fill_null(-1).to_numpy(),
        "category": items["item_category_id"].fill_null(-1).to_numpy(),
        "reviews": np.log1p(np.maximum(safe_float(items["item_rating_reviews_count"]), 0)),
        "rating": safe_float(items["item_rating"]),
        "price": np.log1p(np.maximum(safe_float(items["item_price"]), 0)),
        "lat": lat, "lon": lon,
        "title_len": np.log1p(np.array([len(x) for x in title], dtype=np.float32)),
        "params_len": np.log1p(np.array([len(x) for x in params], dtype=np.float32)),
        "desc_len": np.log1p(np.array([len(x) for x in description], dtype=np.float32)),
        "phone_hidden": items["item_is_phone_hidden"].fill_null(False).to_numpy().astype(np.float32),
        "message_forbidden": items["item_is_message_forbidden"].fill_null(False).to_numpy().astype(np.float32),
        "title_tokens": title_tokens, "param_tokens": param_tokens,
        "title_lower": [x.lower().replace("ё", "е") for x in title],
        "centers": location_centers,
    }
    parsed_path = paths.shared_dir("agent_1") / "items_params" / "items.parquet"
    if parsed_path.exists():
        parsed = (items.select("item_id", "item_idx")
                  .join(pl.read_parquet(parsed_path).select(
                      "item_id", "vid", "tip", "tip_auto", "core_text", "services",
                      "n_services", "n_prices", "min_list_price", "has_price", "experience_years",
                      "where_online", "where_home", "where_visit", "no_visit", "n_keys"),
                        on="item_id", how="left").sort("item_idx"))
        numeric.update({
            "vid": parsed["vid"].fill_null("").to_list(),
            "tip": parsed["tip"].fill_null("").to_list(),
            "tip_auto": parsed["tip_auto"].fill_null("").to_list(),
            "core_tokens": [tokens(x) for x in parsed["core_text"].fill_null("").to_list()],
            "service_tokens": [tokens(" ".join(x or [])) for x in parsed["services"].to_list()],
            "n_services": np.log1p(safe_float(parsed["n_services"])),
            "n_prices": np.log1p(safe_float(parsed["n_prices"])),
            "list_price": np.log1p(np.maximum(safe_float(parsed["min_list_price"]), 0)),
            "has_price": parsed["has_price"].fill_null(False).to_numpy().astype(np.float32),
            "experience": np.log1p(np.maximum(safe_float(parsed["experience_years"]), 0)),
            "where_online": parsed["where_online"].fill_null(False).to_numpy().astype(np.float32),
            "where_home": parsed["where_home"].fill_null(False).to_numpy().astype(np.float32),
            "where_visit": parsed["where_visit"].fill_null(False).to_numpy().astype(np.float32),
            "no_visit": parsed["no_visit"].fill_null(False).to_numpy().astype(np.float32),
            "n_keys": np.log1p(safe_float(parsed["n_keys"])),
        })
    return items.select("item_id", "item_idx"), numeric


def geo_table(split: str) -> dict[int, dict[int, float]]:
    rows = paths.load_train_rows(split, ["search_location_id", "item_location_id"])
    trans = (rows.group_by("search_location_id", "item_location_id").len()
             .with_columns((pl.col("len") / pl.col("len").sum().over("search_location_id")).alias("p")))
    out: dict[int, dict[int, float]] = {}
    for s, i, p in trans.select("search_location_id", "item_location_id", "p").iter_rows():
        if s is not None and i is not None:
            out.setdefault(int(s), {})[int(i)] = float(p)
    return out


def _km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    a = np.radians(lat1); b = np.radians(lat2)
    da = b - a; dl = np.radians(lon2 - lon1)
    h = np.sin(da / 2) ** 2 + np.cos(a) * np.cos(b) * np.sin(dl / 2) ** 2
    return (12742 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))).astype(np.float32)


def make_features(split: str, pool: pl.DataFrame, item_lookup: pl.DataFrame,
                  items: dict, channel_cols: list[str]) -> pl.DataFrame:
    queries = paths.load_queries(split).with_row_index("query_idx")
    meta = paths.load_meta(split).select("query_id", "loc_type", "n_words", "has_filter", "seen_text", "text_rows_in_history")
    q = queries.join(meta, on="query_id", how="left")
    filters_path = paths.shared_dir("agent_1") / "query_filters" / f"{split}.parquet"
    if filters_path.exists():
        q = q.join(pl.read_parquet(filters_path).select(
            "query_id", pl.col("vid").alias("f_vid"), pl.col("tip").alias("f_tip"),
            pl.col("tip_auto").alias("f_tip_auto"), pl.col("n_filters").alias("f_n_filters")),
            on="query_id", how="left")
    pairs = (pool.join(item_lookup, on="item_id", how="left")
             .join(q.select("query_id", "query_idx"), on="query_id", how="left")
             .sort("query_idx", "item_id"))
    ii = pairs["item_idx"].to_numpy().astype(np.int32)
    qi = pairs["query_idx"].to_numpy().astype(np.int32)
    n = pairs.height
    print(f"{split}: extracting features for {n:,} pairs", flush=True)
    qtext = q["search_query"].fill_null("").to_list()
    qtext_norm = [x.lower().replace("ё", "е").strip() for x in qtext]
    qt = [tokens(x) for x in qtext]
    qfilter = [tokens(x) for x in q["search_infm_params_text"].fill_null("").to_list()]
    title = items["title_tokens"]; params = items["param_tokens"]
    title_cov = np.zeros(n, np.float32); params_cov = np.zeros(n, np.float32)
    title_jacc = np.zeros(n, np.float32); filter_cov = np.zeros(n, np.float32)
    phrase = np.zeros(n, np.float32)
    # The text is tokenised once per item/query; this loop only intersects tiny sets.
    title_lower = items["title_lower"]
    for j in range(n):
        a = qt[qi[j]]; b = title[ii[j]]; c = params[ii[j]]
        if a:
            title_cov[j] = len(a & b) / len(a)
            params_cov[j] = len(a & c) / len(a)
            title_jacc[j] = len(a & b) / max(1, len(a | b))
        f = qfilter[qi[j]]
        if f:
            filter_cov[j] = len(f & c) / len(f)
        phrase[j] = float(bool(qtext_norm[qi[j]]) and qtext_norm[qi[j]] in title_lower[ii[j]])
    loc_q = q["search_location_id"].fill_null(-1).to_numpy()[qi]
    loc_i = items["loc"][ii]
    centers = items["centers"]
    qlat = np.array([centers.get(int(x), (np.nan, np.nan))[0] for x in q["search_location_id"].fill_null(-1)], np.float32)[qi]
    qlon = np.array([centers.get(int(x), (np.nan, np.nan))[1] for x in q["search_location_id"].fill_null(-1)], np.float32)[qi]
    distance = _km(qlat, qlon, items["lat"][ii], items["lon"][ii])
    distance = np.nan_to_num(np.log1p(distance), nan=15, posinf=15)
    geo = geo_table(split)
    log_geo = np.empty(n, np.float32)
    for j in range(n):
        p = geo.get(int(loc_q[j]), {}).get(int(loc_i[j]), 1e-6)
        if loc_q[j] == loc_i[j]: p = max(p, 0.5)
        log_geo[j] = np.log(p)
    loc_types = q["loc_type"].fill_null("").to_list()
    lt_codes = {v: k for k, v in enumerate(sorted(set(loc_types)))}
    features = {
        "title_cov": title_cov, "params_cov": params_cov, "title_jacc": title_jacc,
        "filter_cov": filter_cov, "exact_phrase_title": phrase,
        "same_loc": (loc_q == loc_i).astype(np.float32), "log_geo": log_geo, "log_km": distance,
        "log_reviews": items["reviews"][ii], "rating": items["rating"][ii],
        "log_price": items["price"][ii], "phone_hidden": items["phone_hidden"][ii],
        "message_forbidden": items["message_forbidden"][ii],
        "log_title_len": items["title_len"][ii], "log_params_len": items["params_len"][ii],
        "log_desc_len": items["desc_len"][ii],
        "microcat": items["microcat"][ii].astype(np.int32),
        "item_category": items["category"][ii].astype(np.int32),
        "category_match": (items["category"][ii] == q["search_category"].to_numpy()[qi]).astype(np.float32),
        "query_words": q["n_words"].fill_null(0).to_numpy()[qi].astype(np.float32),
        "query_has_filter": q["has_filter"].fill_null(False).to_numpy()[qi].astype(np.float32),
        "query_seen": q["seen_text"].fill_null(False).to_numpy()[qi].astype(np.float32),
        "query_hist_rows": np.log1p(q["text_rows_in_history"].fill_null(0).to_numpy()[qi]).astype(np.float32),
        "query_loc_type": np.array([lt_codes[x] for x in loc_types], np.int32)[qi],
    }
    geo2_path = paths.shared_dir("agent_4") / "geo" / f"{split}.parquet"
    if geo2_path.exists():
        geo2 = {}
        for s, i, logp, logp_item, nloc, dk, ntrans in pl.read_parquet(geo2_path).select(
                "search_location_id", "item_location_id", "log_p", "log_p_item", "n_items_loc", "dist_km", "n_trans").iter_rows():
            geo2[(int(s), int(i))] = (logp, logp_item, nloc, dk, ntrans)
        geo_names = ("geo2_logp", "geo2_logp_item", "geo2_nloc", "geo2_km", "geo2_ntrans")
        geo_arrays = [np.empty(n, np.float32) for _ in geo_names]
        for j in range(n):
            v = geo2.get((int(loc_q[j]), int(loc_i[j])))
            if v is None: v = (-16.12, -20.0, 1, np.nan, 0)
            for k in range(5): geo_arrays[k][j] = v[k]
        for name, array in zip(geo_names, geo_arrays):
            features[name] = np.nan_to_num(array, nan=1000).astype(np.float32)
    if "core_tokens" in items:
        core = items["core_tokens"]; services = items["service_tokens"]
        core_cov = np.zeros(n, np.float32); service_cov = np.zeros(n, np.float32)
        for j in range(n):
            a = qt[qi[j]]
            if a:
                core_cov[j] = len(a & core[ii[j]]) / len(a)
                service_cov[j] = len(a & services[ii[j]]) / len(a)
        features.update(core_cov=core_cov, service_cov=service_cov)
        for name in ("n_services", "n_prices", "list_price", "has_price", "experience",
                     "where_online", "where_home", "where_visit", "no_visit", "n_keys"):
            features[name] = items[name][ii]
        if "f_vid" in q.columns:
            for part in ("vid", "tip", "tip_auto"):
                fv = q[f"f_{part}"].fill_null("").to_list()
                iv = items[part]
                features[f"{part}_filter_match"] = np.array([
                    0 if not fv[qi[j]] else (1 if fv[qi[j]] == iv[ii[j]] else -1)
                    for j in range(n)], np.float32)
            features["n_filters"] = q["f_n_filters"].fill_null(0).to_numpy()[qi].astype(np.float32)
    emb_dir = paths.shared_dir("agent_2") / "e5_base_zs_geo_emb"
    if (emb_dir / "items.npy").exists() and (emb_dir / f"{split}.npy").exists():
        item_emb = np.load(emb_dir / "items.npy", mmap_mode="r")
        query_emb = np.load(emb_dir / f"{split}.npy", mmap_mode="r")
        cos = np.empty(n, np.float32)
        for start in range(0, n, 20000):
            end = min(start + 20000, n)
            a = np.asarray(item_emb[ii[start:end]], dtype=np.float32)
            b = np.asarray(query_emb[qi[start:end]], dtype=np.float32)
            cos[start:end] = np.einsum("ij,ij->i", a, b)
        features["e5_cosine"] = cos
    for col in channel_cols:
        features[col] = pairs[col].fill_null(-100 if col.endswith("score") else 2001).to_numpy().astype(np.float32)
        if col.endswith("score"):
            max_score = pairs.group_by("query_id").agg(pl.col(col).max().alias("max"))
            mx = pairs.select("query_id").join(max_score, on="query_id")["max"].fill_null(-100).to_numpy()
            features[col + "_gap"] = (features[col] - mx).astype(np.float32)
    micro_path = paths.shared_dir("agent_1") / "query_microcat" / f"{split}.parquet"
    if micro_path.exists():
        qids = q["query_id"].to_list()
        qindex = {v: j for j, v in enumerate(qids)}
        micro = {(qindex[qid], int(cat)): (float(prob), int(rank))
                 for qid, cat, prob, rank in pl.read_parquet(micro_path).iter_rows()
                 if qid in qindex}
        prob = np.empty(n, np.float32); rank = np.empty(n, np.float32)
        item_micro = items["microcat"]
        for j in range(n):
            p, r = micro.get((int(qi[j]), int(item_micro[ii[j]])), (0.0, 11))
            prob[j] = p; rank[j] = r
        features["microcat_prob"] = prob
        features["microcat_rank"] = rank
    user_dir = paths.shared_dir("agent_2") / "user_bge_m3_zs_geo_emb"
    if (user_dir / "items.npy").exists() and (user_dir / f"{split}.npy").exists():
        item_emb = np.load(user_dir / "items.npy", mmap_mode="r")
        query_emb = np.load(user_dir / f"{split}.npy", mmap_mode="r")
        cos = np.empty(n, np.float32)
        for start in range(0, n, 20000):
            end = min(start + 20000, n)
            a = np.asarray(item_emb[ii[start:end]], dtype=np.float32)
            b = np.asarray(query_emb[qi[start:end]], dtype=np.float32)
            cos[start:end] = np.einsum("ij,ij->i", a, b)
        features["user_cosine"] = cos
    frame = pl.DataFrame({"query_id": pairs["query_id"], "item_id": pairs["item_id"], **features})
    if split == "rtrain":
        labels = paths.load_qrels("rtrain").with_columns(pl.lit(1).alias("label"))
        frame = frame.join(labels, on=["query_id", "item_id"], how="left").with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
    return frame
