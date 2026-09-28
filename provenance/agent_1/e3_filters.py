"""E3: query filters and category -> table agent_1/query_filters and penalties on top of BM25F + geo.

    uv run python experiments/agent_1/e3_filters.py table              # publish agent_1/query_filters
    uv run python experiments/agent_1/e3_filters.py tune --split rtrain --base <bm25f json>

Filters (search_infm_params_text) use the item-params keys. Measured on hist (rows with a filter): the chosen
item satisfies `Вид услуги X` in 98.4 %, `Тип услуги X` 95.7 %, `Тип услуги автосервиса` 96.2 %,
`Предмет или специальность` 93.9 %, `Чем вы занимаетесь` 92.7 %, `Ваши клиенты` 91.7 %; `Онлайн-запись` only
58.6 %; `Кто оказывает услуги` values ("Частный исполнитель") are not item params (unusable). Items chosen
in category-114 searches are category 114 in 99.99 % of hist rows.

score = BM25F + alpha * geo - b_vid * [vid filter unmet] - b_tip * [tip unmet] - b_auto * [tip_auto unmet]
        - b_gen * #(other filter keys the item has with a different value) - b_cat * [cat != 114 for a 114 search]
An empty item value counts as unmet (the item's microcat has no such parameter).
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(v, "3")

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import lexical as lx  # noqa: E402
import pipeline as pp  # noqa: E402
from avito_bootcamp_2026 import exchange, paths  # noqa: E402

KV_PATH = pp.ART / "items_kv.parquet"
MAIN_KEYS = {"Вид услуги": "vid", "Тип услуги": "tip", "Тип услуги автосервиса": "tip_auto"}
IGNORED_KEYS = {"Кто оказывает услуги", "Рейтинг пользователя", "Участие в пилоте CPL",
                "Срочная услуга (мультистатус)", "Слова в описании"}   # not verifiable on the item side


def build_items_kv() -> pl.DataFrame:
    """(item_idx, key, value) for every parsed param of every corpus item (value '' for flags)."""
    if KV_PATH.exists():
        return pl.read_parquet(KV_PATH)
    texts = paths.load_corpus(["item_infm_params_text"])["item_infm_params_text"].to_list()
    rows_i, rows_k, rows_v = [], [], []
    for i, t in enumerate(texts):
        seen = set()
        for k, v in lx.parse_params(t):
            if k and (k, v) not in seen:
                seen.add((k, v))
                rows_i.append(i)
                rows_k.append(k)
                rows_v.append(v)
    df = pl.DataFrame({"item_idx": np.array(rows_i, dtype=np.int32), "key": rows_k, "value": rows_v})
    df.write_parquet(KV_PATH)
    return df


class FilterIndex:
    """Item-side lookups for filter matching (boolean vectors over the corpus)."""

    def __init__(self):
        kv = build_items_kv()
        self.n = paths.load_corpus(["item_id"]).height
        prm = pl.read_parquet(pp.PARAMS_TABLE, columns=["vid", "tip", "tip_auto", "item_category_id"])
        self.attr = {a: prm[a].to_numpy() for a in ("vid", "tip", "tip_auto")}
        self.not114 = (prm["item_category_id"] != 114).to_numpy()
        self.key_items: dict[str, np.ndarray] = {}
        self.kv_items: dict[tuple[str, str], np.ndarray] = {}
        for (k,), g in kv.group_by(["key"]):
            self.key_items[k] = np.unique(g["item_idx"].to_numpy())
        for (k, v), g in kv.filter(pl.col("value") != "").group_by(["key", "value"]):
            self.kv_items[(k, v)] = g["item_idx"].to_numpy()

    def unmet(self, filters: dict[str, list[str]]) -> dict[str, np.ndarray]:
        """Per penalty group, a float vector over items: 1 where the filter is not satisfied."""
        out = {}
        for key, name in MAIN_KEYS.items():
            vals = filters.get(key)
            if vals:
                out[name] = (~np.isin(self.attr[name], vals)).astype(np.float32)
        gen = np.zeros(self.n, dtype=np.float32)
        for key, vals in filters.items():
            if key in MAIN_KEYS or key in IGNORED_KEYS:
                continue
            has_key = np.zeros(self.n, dtype=bool)
            if key in self.key_items:
                has_key[self.key_items[key]] = True
            ok = np.zeros(self.n, dtype=bool)
            if vals:
                for v in vals:
                    if (key, v) in self.kv_items:
                        ok[self.kv_items[(key, v)]] = True
            else:                                          # a flag filter: the item must have the flag
                ok = has_key
            gen += (~ok).astype(np.float32)
        if filters and gen.any():
            out["gen"] = gen
        return out


def query_filter_table(split: str) -> pl.DataFrame:
    q = paths.load_queries(split)
    rows = []
    for qid, f, cat in zip(q["query_id"], q["search_infm_params_text"].fill_null(""), q["search_category"]):
        d = lx.parse_filters(f)
        rows.append({
            "query_id": qid,
            "vid": (d.get("Вид услуги") or [""])[0],
            "tip": (d.get("Тип услуги") or [""])[0],
            "tip_auto": (d.get("Тип услуги автосервиса") or [""])[0],
            "filters": [f"{k}\t{v}" for k, vs in d.items() for v in (vs or [""])],
            "n_filters": sum(max(1, len(vs)) for vs in d.values()),
            "has_filter": bool(d),
            "search_category": int(cat),
        })
    return pl.DataFrame(rows, schema_overrides={"filters": pl.List(pl.String)})


def cmd_table(a) -> None:
    for split in exchange.SPLITS:
        df = query_filter_table(split)
        df.write_parquet(exchange.artifact_dir("agent_1", "query_filters") / f"{split}.parquet")
        print(split, df.height, "has_filter", round(df["has_filter"].mean(), 3), "vid", round((df["vid"] != "").mean(), 3),
              "tip", round((df["tip"] != "").mean(), 3))
    exchange.write_meta(
        "agent_1", "query_filters", kind="table", fit_only=True, key="query_id",
        files=[f"{s}.parquet" for s in exchange.SPLITS],
        description="Parsed search_infm_params_text per query (all four sets): vid / tip / tip_auto = filter values "
                    "of 'Вид услуги' / 'Тип услуги' / 'Тип услуги автосервиса' ('' = no such filter; 'Вид услуги' with "
                    "an empty value = no filter); filters = every 'key<TAB>value' pair ('' value = flag filter, e.g. "
                    "'Онлайн-запись'); n_filters; has_filter; search_category. Item side: agent_1/items_params "
                    "(vid/tip/tip_auto are functions of item_microcat_id). Parser: lexical.parse_filters.",
        command="uv run python experiments/agent_1/e3_filters.py table")


class Evaluator:
    """BM25F + geo scores per batch, plus the filter / category unmet vectors (exact evaluation)."""

    def __init__(self, split_name: str, base: dict):
        from e2b_bm25f import Evaluator as BM25FEval
        self.ev = BM25FEval(base.get("analyzer", "stem"), split_name)
        self.base = base
        self.fi = FilterIndex()
        self.ftexts = self.ev.split.filters
        self.cats = self.ev.split.cats
        self._unmet: dict[str, dict[str, np.ndarray]] = {}
        for f in set(self.ftexts):
            self._unmet[f] = self.fi.unmet(lx.parse_filters(f))

    def evaluate(self, pen: dict[str, float]) -> dict:
        split = self.ev.split
        ranks = []
        for s, e, sc in self.ev.scores_batches(self.base):
            for j in range(e - s):
                qi = s + j
                for name, vec in self._unmet[self.ftexts[qi]].items():
                    if pen.get(name):
                        sc[j] -= pen[name] * vec
                if pen.get("cat") and self.cats[qi] == 114:
                    sc[j] -= pen["cat"] * self.fi.not114
            ranks += pp.ranks_of_relevant(sc, split.rel[s:e])
        w = split.w
        has_f = np.array([bool(self._unmet[f]) for f in self.ftexts])
        per_q = np.array([(r <= 50).mean() if len(r) else 0 for r in ranks])
        return {"r50_w": round(pp.recall_from_ranks(ranks, 50, w), 5), "r50": round(pp.recall_from_ranks(ranks, 50), 5),
                "r50_filtered": round(float(per_q[has_f].mean()), 5),
                "r1000": round(pp.recall_from_ranks(ranks, 1000), 5)}


def cmd_tune(a) -> None:
    base = json.loads(Path(a.base).read_text())
    ev = Evaluator(a.split, base)
    print("no penalties:", ev.evaluate({}), flush=True)
    grid = [
        {"vid": 1.0}, {"vid": 2.0}, {"vid": 4.0}, {"vid": 8.0},
    ]
    best, best_s = {}, ev.evaluate({})["r50_w"]
    for g in grid:
        r = ev.evaluate(g)
        print(g, r, flush=True)
        if r["r50_w"] > best_s:
            best, best_s = g, r["r50_w"]
    for name, vals in [("tip", [0.5, 1.0, 2.0, 4.0]), ("tip_auto", [0.5, 1.0, 2.0, 4.0]),
                       ("gen", [0.25, 0.5, 1.0, 2.0]), ("cat", [2.0, 5.0, 10.0])]:
        for v in vals:
            g = dict(best, **{name: v})
            r = ev.evaluate(g)
            print(g, r, flush=True)
            if r["r50_w"] > best_s + 1e-5:
                best, best_s = g, r["r50_w"]
    print("best penalties:", best, best_s)
    (pp.ART / f"e3_best_{a.split}.json").write_text(json.dumps(best))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("table")
    p.set_defaults(fn=cmd_table)
    p = sub.add_parser("tune")
    p.add_argument("--split", default="rtrain")
    p.add_argument("--base", required=True)
    p.set_defaults(fn=cmd_tune)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
