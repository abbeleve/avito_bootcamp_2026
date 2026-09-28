"""E1: parse item params of the benchmark corpus -> table agent_1/items_params (+ parser diagnostics).

    uv run python experiments/agent_1/e1_params.py [--check]

--check prints the diagnostics used to curate the key vocabulary in lexical.py: per-key coverage and top
values, capitalised segments inside values (candidate missing keys) and 100 random parses to verify by eye.
"""

from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(v, "3")

import argparse
import collections
import random
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import lexical as lx  # noqa: E402
from avito_bootcamp_2026 import exchange, paths  # noqa: E402

OWNER, NAME = "agent_1", "items_params"


def build_table() -> pl.DataFrame:
    corpus = paths.load_corpus(["item_id", "item_infm_params_text", "item_category_id", "item_microcat_id"])
    recs = [lx.item_params_record(t) for t in corpus["item_infm_params_text"].to_list()]
    df = pl.DataFrame(recs, schema_overrides={"experience_years": pl.Float32, "min_list_price": pl.Int64})
    df = pl.concat([corpus.select("item_id", "item_category_id", "item_microcat_id"), df], how="horizontal")
    df = df.with_columns(
        pl.Series("core_text", [lx.core_text(r) for r in recs]),
        pl.Series("other_params_text", [lx.other_params_text(r) for r in recs]),
    )
    return df.drop("unparsed_head")


def diagnostics(corpus_texts: list[str], n_show: int = 100) -> None:
    key_items = collections.Counter()
    key_vals: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    inner_caps = collections.Counter()
    unparsed = 0
    for t in corpus_texts:
        parsed = lx.parse_params(t)
        seen = set()
        for k, v in parsed:
            seen.add(k)
            key_vals[k][v] += 1
            if k == "":
                unparsed += 1
            # capitalised words inside values of closed-vocabulary keys hint at a missing key
            if lx.KEY_KIND.get(k) in ("desc", "where", "who", "clients", "misc", "money", "schedule", "flag"):
                toks = v.split(" ")
                for i, tok in enumerate(toks[1:], 1):
                    if tok[:1].isupper() and not toks[i - 1].endswith(","):
                        inner_caps[(k, " ".join(toks[i:i + 3]))] += 1
        key_items.update(seen)
    n = len(corpus_texts)
    print(f"items: {n}, with an unparsed head: {unparsed}")
    for k, c in key_items.most_common():
        top = ", ".join(f"{v[:40]!r}:{m}" for v, m in key_vals[k].most_common(6))
        print(f"{c / n:6.1%} {k!r:40s} [{lx.KEY_KIND.get(k, '?')}] {top}")
    print("\ncapitalised segments inside values (possible missing keys):")
    for (k, seg), c in inner_caps.most_common(60):
        print(f"{c:7d}  {k!r} ... {seg!r}")
    rnd = random.Random(0)
    print("\nrandom parses:")
    for t in rnd.sample(corpus_texts, n_show):
        rec = lx.item_params_record(t)
        print("-" * 100)
        print(t[:700])
        print({k: rec[k] for k in ("vid", "tip", "tip_auto", "services", "custom_services", "place", "place_city",
                                   "where_online", "where_home", "where_visit", "who", "experience_years",
                                   "desc_values")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.check:
        texts = paths.load_corpus(["item_infm_params_text"])["item_infm_params_text"].fill_null("").to_list()
        diagnostics(texts)
        return
    df = build_table()
    print(f"parsed {df.height} items in {time.time() - t0:.0f}s")
    cat114 = df.filter(pl.col("item_category_id") == 114)
    print(f"vid non-empty: all {df['vid'].ne('').mean():.3%}, category 114 {cat114['vid'].ne('').mean():.3%}")
    print(f"tip non-empty: {df['tip'].ne('').mean():.3%}; services>0: {(df['n_services'] > 0).mean():.3%}; "
          f"place non-empty: {df['place'].ne('').mean():.3%}; place_city: {df['place_city'].ne('').mean():.3%}")
    out = exchange.artifact_dir(OWNER, NAME) / "items.parquet"
    df.write_parquet(out)
    exchange.write_meta(
        OWNER, NAME, kind="table", fit_only=True,
        description="Parsed item_infm_params_text of the benchmark corpus, one row per item (key: item_id). "
                    "vid/tip/tip_auto = Вид/Тип услуги (автосервиса); services = price-list 'Услуга' names; "
                    "custom_services = 'Название услуги'; place = 'Место оказания услуг' (+ place_city guess); "
                    "where_* flags from 'Как вы работаете'/'Где вы оказываете услуги'/'Куда выезжаете'; who; "
                    "clients; experience_years; n_services; n_prices; min_list_price; has_price; desc_values = "
                    "descriptive enumerations; core_text = vid+tip+services; other_params_text = desc_values. "
                    "Parser: experiments/agent_1/lexical.py (parse_params / item_params_record) - use it for hist "
                    "items too. No training involved.",
        command="uv run python experiments/agent_1/e1_params.py", key="item_id", files=["items.parquet"])
    print(f"wrote {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
