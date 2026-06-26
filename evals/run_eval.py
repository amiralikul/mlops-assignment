"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"
MAX_REPORTED_ITERATIONS = 3


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    payload = {
        "question": question["question"],
        "db": question["db_id"],
        "tags": {"run_type": "eval", "db": question["db_id"]},
    }
    started = time.monotonic()
    try:
        response = httpx.post(agent_url, json=payload, timeout=120.0)
        response.raise_for_status()
        agent = response.json()
    except Exception as e:  # noqa: BLE001
        return {
            "question": question["question"],
            "db_id": question["db_id"],
            "gold_sql": question["gold_sql"],
            "sql": "",
            "agent_ok": False,
            "correct": False,
            "iterations": 0,
            "latency_seconds": time.monotonic() - started,
            "error": f"{type(e).__name__}: {e}",
            "attempts": [],
        }

    gold_ok, gold_rows, gold_error = run_sql(question["db_id"], question["gold_sql"])
    pred_sql = agent.get("sql", "")
    pred_ok, pred_rows, pred_error = run_sql(question["db_id"], pred_sql) if pred_sql else (False, None, "missing SQL")
    final_correct = gold_ok and pred_ok and matches(gold_rows, pred_rows)

    attempts = []
    sql_history = [
        h for h in agent.get("history", [])
        if h.get("node") in {"generate_sql", "revise"}
    ]
    for i, item in enumerate(sql_history, 1):
        attempt_sql = item.get("sql", "")
        attempt_ok, attempt_rows, attempt_error = (
            run_sql(question["db_id"], attempt_sql)
            if attempt_sql
            else (False, None, "missing SQL")
        )
        attempts.append({
            "iteration": i,
            "node": item.get("node"),
            "sql": attempt_sql,
            "exec_ok": attempt_ok,
            "correct": gold_ok and attempt_ok and matches(gold_rows, attempt_rows),
            "error": attempt_error,
        })

    return {
        "question": question["question"],
        "db_id": question["db_id"],
        "gold_sql": question["gold_sql"],
        "gold_exec_ok": gold_ok,
        "gold_error": gold_error,
        "sql": pred_sql,
        "agent_ok": bool(agent.get("ok", False)),
        "correct": final_correct,
        "iterations": int(agent.get("iterations", len(attempts)) or 0),
        "latency_seconds": time.monotonic() - started,
        "error": agent.get("error") or pred_error,
        "attempts": attempts,
        "history": agent.get("history", []),
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    errors = sum(1 for r in results if r.get("error"))
    max_iteration = max(
        [MAX_REPORTED_ITERATIONS]
        + [int(a.get("iteration", 0) or 0) for r in results for a in r.get("attempts", [])]
    )

    per_iteration = {}
    for iteration in range(1, max_iteration + 1):
        iter_correct = 0
        for result in results:
            attempts = result.get("attempts", [])
            seen = [a for a in attempts if int(a.get("iteration", 0) or 0) <= iteration]
            if seen:
                iter_correct += int(bool(seen[-1].get("correct")))
            else:
                iter_correct += int(bool(result.get("correct")) and not attempts)
        per_iteration[str(iteration)] = {
            "correct": iter_correct,
            "total": total,
            "accuracy": (iter_correct / total) if total else 0.0,
        }

    latencies = sorted(float(r.get("latency_seconds", 0.0) or 0.0) for r in results)

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        k = int(round(p * (len(latencies) - 1)))
        return latencies[k]

    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "errors": errors,
        "avg_iterations": (
            sum(int(r.get("iterations", 0) or 0) for r in results) / total
            if total
            else 0.0
        ),
        "latency_p50": pct(0.50),
        "latency_p95": pct(0.95),
        "per_iteration": per_iteration,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
