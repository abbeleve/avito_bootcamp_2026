"""Shared evaluation harness - the only sanctioned way to score val/test predictions.

Every agent scores through this module so that numbers are comparable, and every score is appended
to one shared log (data/results.jsonl) that the ``board`` command summarises.

    uv run python -m avito_bootcamp_2026.evaluate answer  --split val --file out/val_answer.csv --exp agent_1/bm25_geo
    uv run python -m avito_bootcamp_2026.evaluate pool    --split val --file out/val_pool.parquet --exp agent_1/pool_v1
    uv run python -m avito_bootcamp_2026.evaluate compare --split val --a out/a.csv --b out/b.csv
    uv run python -m avito_bootcamp_2026.evaluate check   --file answer.csv        # benchmark submission file
    uv run python -m avito_bootcamp_2026.evaluate board   [--split val]

File formats
* answer CSV (same as the submission): columns ``query_id,answer``; ``answer`` = up to 50 item_ids
  separated by single spaces.
* pool parquet: columns ``query_id, item_id`` and ``rank`` (1 = best) or ``score`` (higher = better).

Metric: Recall@50 = mean over queries of |top50 ∩ relevant| / |relevant|; a query missing from the file
scores 0. ``recall_bench_w`` re-weights queries (raking) so that the marginal distributions of location
type, text frequency, filter presence and query length match the benchmark.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from avito_bootcamp_2026 import paths

K = 50
ID_RE = re.compile(r"^[0-9a-f]{16}$")
RAKE_COLS = ["loc_coarse", "f_bucket", "has_filter", "words_bucket"]
SEGMENTS = ["loc_type", "f_bucket", "has_filter", "words_bucket", "n_rel_b", "pos_reviews_b"]
POOL_KS = (50, 100, 200, 300, 500, 1000, 2000)


# ----------------------------------------------------------------------------------------------------
# Reading and checking answers
# ----------------------------------------------------------------------------------------------------

def read_answer(path: str | Path) -> pl.DataFrame:
    """Read an answer CSV keeping ids as strings (never let a CSV reader turn them into numbers)."""
    df = pl.read_csv(path, schema_overrides={"query_id": pl.String, "answer": pl.String}, infer_schema=False)
    return df.with_columns(pl.col("answer").fill_null(""))


def check_answer(df: pl.DataFrame, expected: set[str], corpus: set[str], strict: bool) -> list[str]:
    """Return a list of problems; the checks mirror the platform's requirements.

    strict=True (benchmark file): the query set must match exactly. For val/test a missing query only
    produces a warning because it simply scores 0.
    """
    problems: list[str] = []
    if df.columns != ["query_id", "answer"]:
        problems.append(f"columns must be exactly ['query_id', 'answer'], got {df.columns}")
        return problems
    qids = df["query_id"].to_list()
    if len(qids) != len(set(qids)):
        problems.append(f"{len(qids) - len(set(qids))} duplicated query_id rows")
    if bad := [q for q in qids if len(q) != 16]:
        problems.append(f"{len(bad)} query_ids are not 16 characters, e.g. {bad[:3]}")
    if extra := set(qids) - expected:
        problems.append(f"{len(extra)} unknown query_ids, e.g. {sorted(extra)[:3]}")
    if missing := expected - set(qids):
        problems.append(("" if strict else "warning: ") + f"{len(missing)} query_ids missing (score 0)")
    n_over = n_dup = n_fmt = n_unknown = n_short = 0
    for ans in df["answer"].to_list():
        ids = ans.split(" ") if ans else []
        n_fmt += any(not ID_RE.match(i) for i in ids)
        n_unknown += any(i not in corpus for i in ids if ID_RE.match(i))
        n_over += len(ids) > K
        n_dup += len(ids) != len(set(ids))
        n_short += len(ids) < K
    for n, what in [(n_fmt, "badly formatted ids (need 16 lowercase hex chars, single spaces)"),
                    (n_unknown, "ids that are not in the corpus"), (n_over, f"more than {K} ids"),
                    (n_dup, "duplicated ids")]:
        if n:
            problems.append(f"{n} rows contain {what}")
    if n_short:
        problems.append(f"warning: {n_short} rows have fewer than {K} ids (free recall left on the table)")
    return problems


def to_lists(df: pl.DataFrame) -> dict[str, list[str]]:
    return {q: (a.split(" ") if a else []) for q, a in df.iter_rows()}


# ----------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------

def per_query_recall(pred: dict[str, list[str]], split: str, k: int = K) -> pl.DataFrame:
    """One row per split query: n_rel, n_hit, recall (missing predictions count as zero hits)."""
    rel = paths.load_qrels(split).group_by("query_id").agg(pl.col("item_id"))
    rows = []
    for qid, items in rel.iter_rows():
        top = set(pred.get(qid, [])[:k])
        hit = sum(i in top for i in items)
        rows.append((qid, len(items), hit, hit / len(items)))
    return pl.DataFrame(rows, schema=["query_id", "n_rel", "n_hit", "recall"], orient="row")


def bench_weights(meta: pl.DataFrame, iters: int = 50) -> np.ndarray:
    """Raking (iterative proportional fitting) weights that match the benchmark's marginals."""
    bench = paths.load_meta("bench")
    w = np.ones(meta.height)
    for _ in range(iters):
        for col in RAKE_COLS:
            vals = meta[col].cast(pl.String).to_numpy()
            target = dict(bench[col].cast(pl.String).value_counts(normalize=True).iter_rows())
            for v in np.unique(vals):
                m = vals == v
                cur = w[m].sum() / w.sum()
                if cur > 0 and target.get(v, 0) > 0:
                    w[m] *= target[v] / cur
    return w / w.mean()


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def with_segments(per_q: pl.DataFrame, split: str) -> pl.DataFrame:
    meta = paths.load_meta(split)
    rv = pl.col("pos_reviews_min")
    return per_q.join(meta.drop("n_rel"), on="query_id", how="left").with_columns(
        pl.when(pl.col("n_rel") >= 3).then(pl.lit("3+")).otherwise(pl.col("n_rel").cast(pl.String)).alias("n_rel_b"),
        pl.when(rv == 0).then(pl.lit("0")).when(rv <= 10).then(pl.lit("1-10")).when(rv <= 50).then(pl.lit("11-50"))
        .otherwise(pl.lit("51+")).alias("pos_reviews_b"))


def summarize(per_q: pl.DataFrame, split: str) -> dict:
    df = with_segments(per_q, split).sort("query_id")
    r = df["recall"].to_numpy()
    w = bench_weights(df)
    lo, hi = bootstrap_ci(r)
    out = {"recall": float(r.mean()), "recall_bench_w": float((w * r).sum() / w.sum()),
           "ci95": [round(lo, 4), round(hi, 4)], "n_queries": df.height, "segments": {}}
    for seg in SEGMENTS:
        g = df.group_by(seg).agg(pl.col("recall").mean(), pl.len().alias("n")).sort(seg)
        out["segments"][seg] = {str(k): [round(v, 4), n] for k, v, n in g.iter_rows()}
    return out


def pool_recall(pool: pl.DataFrame, split: str) -> dict:
    """Recall of a candidate pool at several depths (rank <= k). Upper bound for any re-ranker on top."""
    if "rank" not in pool.columns:
        pool = pool.with_columns(pl.col("score").rank("ordinal", descending=True).over("query_id").alias("rank"))
    qrels = paths.load_qrels(split)
    n_rel = qrels.group_by("query_id").len().rename({"len": "n_rel"})
    hits = qrels.join(pool.select("query_id", "item_id", "rank"), on=["query_id", "item_id"], how="left")
    out = {"mean_pool_size": float(pool.group_by("query_id").len()["len"].mean() or 0)}
    for k in POOL_KS:
        per_q = (hits.group_by("query_id").agg((pl.col("rank") <= k).sum().alias("hit"))
                 .join(n_rel, on="query_id").with_columns(pl.col("hit") / pl.col("n_rel")))
        out[f"recall@{k}"] = round(float(per_q["hit"].sum() / n_rel.height), 4)
    return out


# ----------------------------------------------------------------------------------------------------
# Shared results log
# ----------------------------------------------------------------------------------------------------

def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "?"


def log_result(record: dict) -> None:
    record = {"ts": dt.datetime.now().isoformat(timespec="seconds"), "cwd": str(Path.cwd()),
              "git": f"{_git('rev-parse', '--abbrev-ref', 'HEAD')}@{_git('rev-parse', '--short', 'HEAD')}", **record}
    log = paths.results_log()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:          # flock: parallel agents append safely
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------

def print_summary(s: dict) -> None:
    print(f"Recall@{K} = {s['recall']:.4f}  95% CI [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]  "
          f"bench-weighted = {s['recall_bench_w']:.4f}  (n = {s['n_queries']})")
    for seg, vals in s["segments"].items():
        print(f"  {seg:14s} " + "  ".join(f"{k}: {v[0]:.3f} ({v[1]})" for k, v in vals.items()))


def cmd_answer(a: argparse.Namespace) -> None:
    df = read_answer(a.file)
    problems = check_answer(df, set(paths.load_queries(a.split)["query_id"]),
                            set(paths.load_corpus(["item_id"])["item_id"]), strict=False)
    errors = [p for p in problems if not p.startswith("warning")]
    for p in problems:
        print(("ERROR: " if p in errors else "") + p)
    if errors:
        sys.exit(1)
    s = summarize(per_query_recall(to_lists(df), a.split), a.split)
    print_summary(s)
    if a.exp:
        log_result({"exp": a.exp, "split": a.split, "kind": "answer", "recall": round(s["recall"], 4),
                    "recall_bench_w": round(s["recall_bench_w"], 4), "ci95": s["ci95"], "note": a.note,
                    "file": str(Path(a.file).resolve()), "file_sha": file_sha(a.file), "segments": s["segments"]})


def cmd_pool(a: argparse.Namespace) -> None:
    pool = pl.read_parquet(a.file)
    s = pool_recall(pool, a.split)
    print("  ".join(f"{k}={v}" for k, v in s.items()))
    if a.exp:
        log_result({"exp": a.exp, "split": a.split, "kind": "pool", **s, "note": a.note,
                    "file": str(Path(a.file).resolve()), "file_sha": file_sha(a.file)})


def cmd_compare(a: argparse.Namespace) -> None:
    ra = per_query_recall(to_lists(read_answer(a.a)), a.split).sort("query_id")
    rb = per_query_recall(to_lists(read_answer(a.b)), a.split).sort("query_id")
    d = rb["recall"].to_numpy() - ra["recall"].to_numpy()
    lo, hi = bootstrap_ci(d)
    print(f"B - A = {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"queries better: {(d > 0).sum()}  worse: {(d < 0).sum()}  (A = {ra['recall'].mean():.4f}, B = {rb['recall'].mean():.4f})")


def cmd_check(a: argparse.Namespace) -> None:
    problems = check_answer(read_answer(a.file), set(paths.load_queries("bench")["query_id"]),
                            set(paths.load_corpus(["item_id"])["item_id"]), strict=True)
    for p in problems:
        print(p)
    if any(not p.startswith("warning") for p in problems):
        sys.exit(1)
    print("benchmark answer file OK")


def cmd_board(a: argparse.Namespace) -> None:
    log = paths.results_log()
    if not log.exists():
        print("no results yet")
        return
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pl.DataFrame([{k: r.get(k) for k in ("ts", "exp", "split", "kind", "recall", "recall_bench_w", "recall@200",
                                              "recall@1000", "note")} for r in rows])
    df = df.filter(pl.col("split") == a.split) if a.split else df
    answers = (df.filter(pl.col("kind") == "answer").sort("ts").group_by("exp", "split").last()
               .sort("recall", descending=True).select("exp", "split", "recall", "recall_bench_w", "ts", "note"))
    pools = (df.filter(pl.col("kind") == "pool").sort("ts").group_by("exp", "split").last()
             .sort("recall@1000", descending=True).select("exp", "split", "recall@200", "recall@1000", "ts", "note"))
    with pl.Config(tbl_rows=a.top, fmt_str_lengths=60, tbl_width_chars=200):
        print("latest Recall@50 per experiment:"); print(answers.head(a.top))
        print("latest pool recall per experiment:"); print(pools.head(a.top))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("answer", cmd_answer), ("pool", cmd_pool)]:
        p = sub.add_parser(name)
        p.add_argument("--split", choices=["val", "test", "rtrain"], required=True)
        p.add_argument("--file", required=True)
        p.add_argument("--exp", help="<agent>/<experiment>; when given, the score is appended to the shared log")
        p.add_argument("--note", default="")
        p.set_defaults(fn=fn)
    p = sub.add_parser("compare")
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("check")
    p.add_argument("--file", required=True)
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("board")
    p.add_argument("--split", choices=["val", "test", "rtrain"])
    p.add_argument("--top", type=int, default=40)
    p.set_defaults(fn=cmd_board)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
