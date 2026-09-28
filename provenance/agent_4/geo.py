"""E1 - geo model v2: P(item_location | search_location) smoothed toward a distance kernel.

    P(l | s) = (n(s, l) + m * b(l | s)) / (N_s + m)

* n(s, l)  - rows of the training rows (hist for rtrain/val/test, full train for bench) where a search at s
             chose an item at l; N_s = sum over l.
* b(l | s) - backoff distribution, by the kind of search location:
    - city   (s is itself an item location): r * [l = s] + (1 - r) * ((1 - eps) * K_s(l) + eps * nat(l))
    - region (s is never an item location, e.g. 107620 Moscow + oblast): (1 - eps) * K_s(l) + eps * nat(l)
    - national (choices spread over the whole country, e.g. 621540 all Russia): as a region, but its kernel
      scale (2 x d75 of its choices, ~thousands of km) makes it close to the size prior nat(l)
  K_s(l) ~ w_l^gamma * exp(-d(s, l) / tau_s) over l != s  (distance kernel between location centroids),
  nat(l) ~ w_l^gamma  (national prior), w_l = number of corpus items at l.
* centroids: median lat/lon of the items at a location (corpus + training rows); for a region/national
  search id, the median lat/lon of the items chosen from it. tau_s = tau for cities; for regions, the median
  distance of its choices from its centroid (region_mult x the d75 quantile, floored at tau).
* per-item prior: log P(l | s) - c * log n_items(l); with the baseline BM25 the best val combination is
  score = BM25 + 2 * (log P(l | s) - 0.5 * log n_items(l))  (val 0.8662 vs 0.859 for the baseline geo).

The published table (see ``publish``) holds, for every search location of a query set and every corpus item
location: log_p = log P(l | s), dist_km between centroids, and the kind of s.
"""

from __future__ import annotations

import common  # noqa: F401  (thread limits)

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from avito_bootcamp_2026 import exchange, paths

NATIONAL_SPREAD_KM = 500.0      # a search id whose choices sit this far (median) from their centroid = national


@dataclass
class GeoParams:
    # defaults = best mean log-likelihood of the relevant items' locations on rtrain + val (geo_ll.py)
    m: float = 20.0          # pseudo-count of the backoff (city searches)
    m_wide: float = 50.0     # pseudo-count of the backoff (region / national searches)
    r: float = 0.8           # backoff self-mass for city searches
    tau: float = 60.0        # km, distance kernel scale for city searches
    region_q: str = "d75"    # which spread quantile of a region's choices sets its kernel scale
    region_mult: float = 2.0 # multiplier of that scale
    eps: float = 0.1         # national share inside the backoff
    gamma: float = 0.75      # exponent of the location size weight
    dedup: bool = False      # count unique (search text, loc, item) instead of rows
    floor: float = 1e-7      # minimum probability


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class GeoModel:
    def __init__(self, rows: pl.DataFrame, corpus: pl.DataFrame, prm: GeoParams):
        self.prm = prm
        ll = [pl.col("item_latitude").cast(pl.Float64), pl.col("item_longitude").cast(pl.Float64)]
        corpus = corpus.select("item_id", "item_location_id", *ll)
        rows = rows.select("search_query", "search_location_id", "item_id", "item_location_id", *ll)
        # --- item locations of the corpus (the axis of every geo vector) ---
        self.locs, self.item_loc_idx = np.unique(corpus["item_location_id"].to_numpy(), return_inverse=True)
        self.loc_pos = {int(l): i for i, l in enumerate(self.locs)}
        w = np.bincount(self.item_loc_idx, minlength=len(self.locs)).astype(np.float64)
        self.w = w
        # --- centroids of item locations: median over unique items of corpus + training rows ---
        items = pl.concat([corpus, rows.select(corpus.columns)]).unique("item_id")
        cen = items.group_by("item_location_id").agg(pl.col("item_latitude").median().alias("lat"),
                                                     pl.col("item_longitude").median().alias("lon"))
        cen_d = {l: (a, b) for l, a, b in cen.iter_rows()}
        self.lat = np.array([cen_d[int(l)][0] for l in self.locs])
        self.lon = np.array([cen_d[int(l)][1] for l in self.locs])
        # --- transition counts ---
        if prm.dedup:
            rows = rows.unique(["search_query", "search_location_id", "item_id"])
        self.trans = rows.group_by("search_location_id", "item_location_id").len()
        self.n_s = dict(self.trans.group_by("search_location_id").agg(pl.col("len").sum()).iter_rows())
        by_s: dict[int, dict[int, int]] = {}
        for s, l, n in self.trans.iter_rows():
            by_s.setdefault(int(s), {})[int(l)] = int(n)
        self.by_s = by_s
        # --- centroids / spread of search locations (median of chosen items' coordinates) ---
        sc = rows.group_by("search_location_id").agg(pl.col("item_latitude").median().alias("lat"),
                                                     pl.col("item_longitude").median().alias("lon"))
        self.s_cen = {int(s): (a, b) for s, a, b in sc.iter_rows()}
        chosen = rows.join(sc, on="search_location_id").with_columns(
            pl.struct("item_latitude", "item_longitude", "lat", "lon").map_batches(
                lambda st: pl.Series(haversine(st.struct.field("lat").to_numpy(), st.struct.field("lon").to_numpy(),
                                               st.struct.field("item_latitude").to_numpy(),
                                               st.struct.field("item_longitude").to_numpy()))).alias("d"))
        spread = chosen.group_by("search_location_id").agg(
            pl.col("d").median().alias("d50"), pl.col("d").quantile(0.75).alias("d75"),
            pl.col("d").quantile(0.9).alias("d90"))
        self.s_spread = dict(spread.select("search_location_id", "d50").iter_rows())
        self.s_spread_q = {q: dict(spread.select("search_location_id", q).iter_rows()) for q in ("d50", "d75", "d90")}
        self._cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------------------------------------
    def kind(self, s: int) -> str:
        if s in self.loc_pos:
            return "city"
        if self.s_spread.get(s, 0.0) >= NATIONAL_SPREAD_KM:
            return "national"
        return "region" if s in self.s_cen else "unknown"

    def centroid(self, s: int) -> tuple[float, float] | None:
        if s in self.loc_pos:
            i = self.loc_pos[s]
            return float(self.lat[i]), float(self.lon[i])
        return self.s_cen.get(s)

    def dist_km(self, s: int) -> np.ndarray:
        c = self.centroid(s)
        if c is None:
            return np.full(len(self.locs), np.nan)
        return haversine(c[0], c[1], self.lat, self.lon)

    def prob(self, s: int) -> np.ndarray:
        """P(l | s) over the corpus item locations (sums to <= 1: mass on non-corpus locations is dropped)."""
        if s in self._cache:
            return self._cache[s]
        p = self.prm
        size = self.w ** p.gamma
        nat = size / size.sum()
        kind = self.kind(s)
        if kind == "unknown":
            back = nat
        else:
            d = self.dist_km(s)
            tau = p.tau if kind == "city" else max(p.tau, p.region_mult * float(self.s_spread_q[p.region_q].get(s, p.tau)))
            k = size * np.exp(-d / tau)
            if kind == "city":
                k[self.loc_pos[s]] = 0.0
            k = k / k.sum() if k.sum() > 0 else nat
            back = (1 - p.eps) * k + p.eps * nat
            if kind == "city":
                back = (1 - p.r) * back
                back[self.loc_pos[s]] += p.r
        counts = np.zeros(len(self.locs))
        for l, n in self.by_s.get(s, {}).items():
            if l in self.loc_pos:
                counts[self.loc_pos[l]] = n
        n_s = self.n_s.get(s, 0)
        m = p.m if kind == "city" else p.m_wide
        prob = (counts + m * back) / (n_s + m)
        prob = np.maximum(prob, p.floor)
        self._cache[s] = prob
        return prob

    def log_prob_items(self, s: int, c: float = 0.0) -> np.ndarray:
        """log P(item_location(j) | s) - c * log n_items(location(j)) for every corpus item j (corpus order).

        c = 1 spreads the location's probability uniformly over its items (per-item prior); c = 0 is the
        location-level prior of the reference baseline."""
        return (np.log(self.prob(s)) - c * np.log(self.w))[self.item_loc_idx]


def load_model(split: str, prm: GeoParams) -> GeoModel:
    corpus = paths.load_corpus(["item_id", "item_location_id", "item_latitude", "item_longitude"])
    rows = paths.load_train_rows(split, ["search_query", "search_location_id", "item_id", "item_location_id",
                                         "item_latitude", "item_longitude"])
    return GeoModel(rows, corpus, prm)


# ----------------------------------------------------------------------------------------------------
# Baseline geo (experiments/00_baseline): empirical transitions, 1e-6 floor, self floored at 0.5
# ----------------------------------------------------------------------------------------------------

class BaselineGeo:
    def __init__(self, split: str):
        corpus = paths.load_corpus(["item_location_id"])
        rows = paths.load_train_rows(split, ["search_location_id", "item_location_id"])
        self.locs, self.item_loc_idx = np.unique(corpus["item_location_id"].to_numpy(), return_inverse=True)
        trans = (rows.group_by("search_location_id", "item_location_id").len()
                 .with_columns((pl.col("len") / pl.col("len").sum().over("search_location_id")).alias("p")))
        self.table: dict[int, dict[int, float]] = {}
        for s, l, pr in trans.select("search_location_id", "item_location_id", "p").iter_rows():
            self.table.setdefault(s, {})[l] = pr
        self._cache: dict[int, np.ndarray] = {}

    def log_prob_items(self, s: int) -> np.ndarray:
        if s not in self._cache:
            probs = self.table.get(s, {})
            v = np.array([probs.get(int(l), 1e-6) for l in self.locs])
            v[self.locs == s] = np.maximum(v[self.locs == s], 0.5)
            self._cache[s] = np.log(v)[self.item_loc_idx]
        return self._cache[s]


# ----------------------------------------------------------------------------------------------------
# Evaluation: BM25 (baseline text model) + alpha * geo, per loc_type
# ----------------------------------------------------------------------------------------------------

def evaluate(split: str, geo, alphas, bm=None, write: str | None = None, c: float = 0.0) -> dict:
    bm = common.baseline_bm25_scores(split) if bm is None else bm
    q = paths.load_queries(split)
    meta = paths.load_meta(split).select("query_id", "loc_type")
    lt = dict(meta.iter_rows())
    qids, locs = q["query_id"].to_list(), q["search_location_id"].to_list()
    ids = paths.load_corpus(["item_id"])["item_id"].to_numpy()
    res = {}
    for a in alphas:
        answers = []
        for j in range(len(qids)):
            s = bm[j].toarray().ravel() + a * (geo.log_prob_items(locs[j], c) if c else geo.log_prob_items(locs[j]))
            answers.append(list(ids[common.top_k(s, ids, 50)]))
        if split != "bench":
            r = common.recall_at(split, qids, answers)
            seg = pl.DataFrame({"lt": [lt[x] for x in qids], "r": r}).group_by("lt").agg(pl.col("r").mean()).sort("lt")
            res[a] = {"recall": float(r.mean()), **{k: round(v, 4) for k, v in seg.iter_rows()}}
        if write:
            common.write_answer(common.OUT / f"{split}_{write}.csv", qids, answers)
    return res


# ----------------------------------------------------------------------------------------------------
# Publishing the table
# ----------------------------------------------------------------------------------------------------

def geo_table(model: GeoModel, search_locs) -> pl.DataFrame:
    parts = []
    for s in sorted(set(int(x) for x in search_locs)):
        lp = np.log(model.prob(s))
        parts.append(pl.DataFrame({
            "search_location_id": np.full(len(model.locs), s, dtype=np.int64),
            "item_location_id": model.locs.astype(np.int64),
            "log_p": lp.astype(np.float32),
            "log_p_item": (lp - np.log(model.w)).astype(np.float32),
            "n_items_loc": model.w.astype(np.int32),
            "dist_km": model.dist_km(s).astype(np.float32),
            "n_trans": np.array([model.by_s.get(s, {}).get(int(l), 0) for l in model.locs], dtype=np.int32),
            "search_kind": [model.kind(s)] * len(model.locs),
        }))
    return pl.concat(parts)


def publish(prm: GeoParams, alpha: float, c: float) -> None:
    t0 = time.time()
    name = "geo"
    for split in common.SPLITS:
        model = load_model(split, prm)
        locs = paths.load_queries(split)["search_location_id"].unique().to_list()
        tab = geo_table(model, locs)
        tab.write_parquet(exchange.artifact_dir("agent_4", name) / f"{split}.parquet")
        cen = pl.DataFrame({"location_id": model.locs.astype(np.int64), "lat": model.lat, "lon": model.lon,
                            "n_corpus_items": model.w.astype(np.int32)})
        if split == "bench":
            cen.write_parquet(exchange.artifact_dir("agent_4", name) / "item_location_centroids.parquet")
        sc = pl.DataFrame([{"search_location_id": s, "kind": model.kind(s),
                            "lat": (model.centroid(s) or (None, None))[0], "lon": (model.centroid(s) or (None, None))[1],
                            "spread_km": model.s_spread.get(s), "n_rows": model.n_s.get(s, 0)} for s in sorted(set(locs))])
        sc.write_parquet(exchange.artifact_dir("agent_4", name) / f"search_locations_{split}.parquet")
        print(f"geo {split}: {tab.height:,} rows ({time.time() - t0:.0f}s)")
    exchange.write_meta(
        "agent_4", name, kind="table", fit_only=True,
        description=("Geo model v2: log P(item_location | search_location) smoothed toward a distance kernel "
                     "between location centroids; region ids use their own spread, all-Russia a national prior. "
                     "Key (search_location_id, item_location_id); files {rtrain,val,test}.parquet from hist, "
                     "bench.parquet from full train. Recommended use: score = text + alpha * (log_p - c * log n_items_loc) "
                     f"(alpha={alpha}, c={c} with baseline BM25: val 0.8662 vs 0.859 baseline geo)."),
        columns={"log_p": "log P(l|s), smoothed, floored", "log_p_item": "log_p - log n_items_loc (uniform per-item prior)",
                 "n_items_loc": "number of corpus items at item_location_id",
                 "dist_km": "haversine between centroids (NaN if unknown)",
                 "n_trans": "raw transition count in the training rows", "search_kind": "city / region / national"},
        extra_files={"item_location_centroids.parquet": "location_id, lat, lon, n_corpus_items (bench fit)",
                     "search_locations_{split}.parquet": "search_location_id, kind, centroid, spread_km, n_rows"},
        params=asdict(prm), alpha=alpha, c=c, command="uv run python experiments/agent_4/geo.py publish --alphas 2 --c 0.5")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["baseline", "grid", "publish"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--alphas", default="2")
    ap.add_argument("--c", type=float, default=0.5)
    ap.add_argument("--write", default=None)
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]
    prm = GeoParams(**json.loads(a.params))
    if a.cmd == "baseline":
        print(json.dumps(evaluate(a.split, BaselineGeo(a.split), alphas), indent=1))
    elif a.cmd == "grid":
        print(json.dumps(evaluate(a.split, load_model(a.split, prm), alphas, write=a.write, c=a.c), indent=1))
    else:
        publish(prm, alphas[0], a.c)


if __name__ == "__main__":
    main()
