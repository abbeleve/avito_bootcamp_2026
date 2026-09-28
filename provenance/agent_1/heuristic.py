"""agent_1 best heuristic: BM25F + geo prior - filter/category penalties + microcategory prior.

    uv run python experiments/agent_1/heuristic.py tune  --split rtrain --config <json>   # line search
    uv run python experiments/agent_1/heuristic.py run   --config <json> --splits val,test,rtrain,bench \
        --name <out folder> [--publish lex_fields]

score(q, i) = BM25F(q, i)                                    text (title / core / description / params / address)
            + alpha * log P(loc_i | loc_q)                   geo prior (baseline definition)
            - pen_vid * [Вид filter unmet] - pen_tip * [Тип unmet] - pen_auto * [Тип автосервиса unmet]
            - pen_gen * #(other filters unmet) - pen_cat * [category != 114 in a category-114 search]
            + beta * log(P(microcat_i | q) + eps)            microcategory prior (E4 model)

Every evaluation is exact over the full corpus (expected rank of the relevant items, ties counted half).
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(v, "3")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import lexical as lx  # noqa: E402
import pipeline as pp  # noqa: E402
from avito_bootcamp_2026 import exchange, paths  # noqa: E402

BATCH = 250
FIELDS = ("title", "core", "desc4000", "other", "addr")


def load_cfg(s: str) -> dict:
    return json.loads(Path(s).read_text()) if s.endswith(".json") else json.loads(s)


class Scorer:
    def __init__(self, split_name: str, analyzer: str = "stem", microcat: str | None = None):
        t0 = time.time()
        c = paths.load_corpus(["item_id", "item_location_id", "item_microcat_id"])
        self.ids = np.array(c["item_id"].to_list())
        index = {i: k for k, i in enumerate(self.ids)}
        self.split = pp.Split(split_name, index)
        self.geo = pp.GeoPrior(split_name, c["item_location_id"].to_numpy())
        self.an = pp.Analyzed(analyzer)
        self.bm = pp.BM25F(self.an, FIELDS)
        self.q = self.an.query_matrix(self.split.texts)
        self._wt_key, self._wt = None, None
        # filters
        from e3_filters import FilterIndex
        self.fi = FilterIndex()
        self._unmet = {f: self.fi.unmet(lx.parse_filters(f)) for f in set(self.split.filters)}
        # microcategory prior: P (Q x classes) and the class index of every item (-1 = unseen class)
        self.P = None
        if microcat:
            fit_on = "bench" if split_name == "bench" else "val"
            classes = np.load(pp.ART / f"e4_classes_{fit_on}.npy")
            self.P = np.load(pp.ART / f"e4_P_{microcat}_{split_name}.npy")
            cidx = {m: k for k, m in enumerate(classes)}
            self.item_class = np.array([cidx.get(m, -1) for m in c["item_microcat_id"].to_list()])
        pp.log(f"scorer ready ({split_name})", t0)

    def text_wt(self, cfg: dict):
        key = json.dumps([cfg["w"], cfg["b"], cfg["k1"]], sort_keys=True)
        if key != self._wt_key:
            self._wt_key, self._wt = key, self.bm.weights(cfg["w"], cfg["b"], cfg["k1"])
        return self._wt

    def batches(self, cfg: dict):
        wt = self.text_wt(cfg)
        pen = cfg.get("pen", {})
        beta = cfg.get("beta", 0.0)
        eps = cfg.get("eps", 1e-3)
        for s in range(0, len(self.split), BATCH):
            e = min(len(self.split), s + BATCH)
            sc = (self.q[s:e] @ wt).toarray()
            sc += cfg["alpha"] * np.stack([self.geo(l) for l in self.split.locs[s:e]])
            for j in range(e - s):
                qi = s + j
                for name, vec in self._unmet[self.split.filters[qi]].items():
                    if pen.get(name):
                        sc[j] -= pen[name] * vec
                if pen.get("cat") and self.split.cats[qi] == 114:
                    sc[j] -= pen["cat"] * self.fi.not114
                if beta and self.P is not None:
                    p = np.where(self.item_class >= 0, self.P[qi][np.maximum(self.item_class, 0)], 0.0)
                    sc[j] += beta * np.log(p + eps).astype(np.float32)
            yield s, e, sc

    def evaluate(self, cfg: dict) -> dict:
        ranks = []
        for s, e, sc in self.batches(cfg):
            ranks += pp.ranks_of_relevant(sc, self.split.rel[s:e])
        w = self.split.w
        has_f = np.array([bool(self._unmet[f]) for f in self.split.filters])
        per_q = np.array([(r <= 50).mean() if len(r) else 0 for r in ranks])
        return {"r50_w": round(pp.recall_from_ranks(ranks, 50, w), 5), "r50": round(pp.recall_from_ranks(ranks, 50), 5),
                "r50_filt": round(float(per_q[has_f].mean()), 4), "r200": round(pp.recall_from_ranks(ranks, 200), 4),
                "r1000": round(pp.recall_from_ranks(ranks, 1000), 4)}

    def run(self, cfg: dict, k: int = 2000):
        tops, tscores = [], []
        for s, e, sc in self.batches(cfg):
            top = pp.topk(sc, k, self.ids)
            tops += top
            tscores += [row[t] for row, t in zip(sc, top)]
        return tops, tscores


def fmt(cfg: dict) -> str:
    s = f"k1={cfg['k1']:g} a={cfg['alpha']:g} " + " ".join(f"{f}={cfg['w'][f]:g}/{cfg['b'][f]:g}" for f in cfg["w"])
    if cfg.get("pen"):
        s += " pen=" + ",".join(f"{k}:{v:g}" for k, v in cfg["pen"].items())
    if cfg.get("beta"):
        s += f" beta={cfg['beta']:g} eps={cfg.get('eps', 1e-3):g}"
    return s


def set_path(cfg: dict, path: str, v) -> dict:
    c = json.loads(json.dumps(cfg))
    node = c
    parts = path.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = v
    return c


def line_search(sc: Scorer, cfg: dict, steps: list[tuple[str, list]], metric: str = "r50_w") -> dict:
    best = json.loads(json.dumps(cfg))
    best_s = sc.evaluate(best)[metric]
    print(f"start {metric}={best_s:.4f} {fmt(best)}", flush=True)
    for path, values in steps:
        for v in values:
            c = set_path(best, path, v)
            if c == best:
                continue
            s = sc.evaluate(c)[metric]
            if s > best_s + 1e-5:
                best, best_s = c, s
        print(f"  {path:12s} -> {metric}={best_s:.4f} {fmt(best)}", flush=True)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "eval", "run"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="rtrain")
    ap.add_argument("--splits", default="val")
    ap.add_argument("--steps", default="", help="json list of [path, [values]]")
    ap.add_argument("--microcat", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--publish", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    mc = a.microcat or cfg.get("microcat") or None
    if a.cmd in ("tune", "eval"):
        sc = Scorer(a.split, cfg.get("analyzer", "stem"), mc)
        if a.cmd == "eval":
            print(fmt(cfg), sc.evaluate(cfg))
            return
        best = line_search(sc, cfg, json.loads(a.steps))
        if mc:
            best["microcat"] = mc
        print("best:", fmt(best), sc.evaluate(best))
        if a.out:
            Path(a.out).write_text(json.dumps(best, ensure_ascii=False))
        return
    t0 = time.time()
    for split_name in a.splits.split(","):
        sc = Scorer(split_name, cfg.get("analyzer", "stem"), mc)
        tops, tscores = sc.run(cfg)
        pp.write_answer(sc.split, tops, sc.ids, pp.OUT / a.name / f"{split_name}_answer.csv")
        if a.publish:
            exchange.save_channel("agent_1", a.publish, split_name, pp.channel_frame(sc.split, tops, tscores, sc.ids))
        pp.log(f"{split_name}: written", t0)
    (pp.OUT / a.name / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    if a.publish:
        exchange.write_meta("agent_1", a.publish, kind="channel", fit_only=True, k=2000, config=cfg,
                            description=a.note or fmt(cfg),
                            command=f"uv run python experiments/agent_1/heuristic.py run --config "
                                    f"experiments/agent_1/out/{a.name}/config.json --splits {a.splits} "
                                    f"--name {a.name} --publish {a.publish}")


if __name__ == "__main__":
    main()
