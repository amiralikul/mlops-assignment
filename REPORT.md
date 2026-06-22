# MLOps Assignment Report

## Scope and Current Status

This report currently documents the parts validated without the final H100-hosted
vLLM run. Agent development and tracing were tested locally against Nebius Token
Factory, which exposes an OpenAI-compatible API. Final serving configuration,
Grafana metrics, eval pass rates, and SLO claims must still be collected against
`Qwen/Qwen3-30B-A3B-Instruct-2507` served by vLLM on one H100.

## Phase 1: vLLM Serving Configuration

Status: pending H100 validation.

The final report should be filled from the actual vLLM command used on the H100.
The initial configuration should be chosen around this workload:

- prompts are roughly 1.5K-3K tokens,
- outputs are short SQL or JSON snippets,
- each agent request usually performs 2-3 dependent LLM calls,
- the target SLO is P95 end-to-end agent latency under 5 seconds at 10+ RPS over
  a 5-minute window.

Planned configuration fields to report:

| Flag | Value | Reason |
|---|---:|---|
| `--model` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Fixed assignment model. |
| `--host` / `--port` | `0.0.0.0` / `8000` | Exposes an OpenAI-compatible endpoint for the agent and manual checks. |
| `--max-model-len` | TBD | Should be large enough for schema + question prompts without wasting KV cache. |
| `--max-num-seqs` | TBD | Controls concurrency; tune against KV cache headroom and queueing. |
| `--max-num-batched-tokens` | TBD | Main prefill batching lever; tune for P95 TTFT rather than only throughput. |
| `--enable-chunked-prefill` | TBD | Expected to help tail latency for 1.5K-3K token prompts under bursty load. |

Required artifact still pending: `screenshots/vllm_manual_query.png`.

## Phase 2: Serving Observability

Status: pending real vLLM metrics.

Token Factory was useful for agent development, but it does not expose this
assignment's vLLM `/metrics` endpoint. The Grafana dashboard and final serving
screenshots should therefore be completed against the H100 vLLM process.

The dashboard should answer three questions:

- Latency: are requests slow, and is the time in queue, prefill/TTFT, or decode?
- Throughput: how many requests and generated tokens are being served?
- KV cache: is there enough cache headroom for the current concurrency?

Required artifacts still pending:

- `screenshots/grafana_serving.png`
- `screenshots/grafana_eval_run.png`
- `screenshots/grafana_before.png`
- `screenshots/grafana_after.png`

## Phase 3: Agent Design

Status: implemented and manually validated through the HTTP endpoint.

The agent is a LangGraph workflow:

```text
attach_schema -> generate_sql -> execute -> verify
                                      |
                           ok=false  v
                                  revise -> execute -> verify
```

The loop is capped at 3 total generate/revise attempts. The agent returns the
final SQL, rows, iteration count, success flag, and per-node history.

Key implementation choices:

- The schema prompt is generated from SQLite introspection.
- BIRD metadata from `dev_tables.json` is included as column comments, so encoded
  columns such as `A15` are visible to the model as "no. of committed crimes
  1995".
- The verifier asks for JSON shaped as `{"ok": bool, "issue": str}` and parses it
  defensively because models may wrap JSON in markdown fences or prose.
- A deterministic duplicate-row guard rejects repeated identical rows when the
  SQL did not use `DISTINCT` or `GROUP BY`; this made the revise loop reliable
  for entity-attribute questions.

Manual endpoint validation used the first five questions from `evals/eval_set.jsonl`.
All returned HTTP 200 and `ok=true`.

| # | DB | Result |
|---:|---|---|
| 1 | `formula_1` | Triggered revise; final SQL added `DISTINCT` and returned one coordinate row. |
| 2 | `superhero` | Returned Ajax's powers in one iteration. |
| 3 | `california_schools` | Returned top 5 NCES school IDs in one iteration. |
| 4 | `financial` | Used `A15` for committed crimes in 1995 after schema metadata enrichment. |
| 5 | `financial` | Returned the male client count for `Hl.m. Praha` in one iteration. |

The revise example:

```sql
SELECT DISTINCT c."lat", c."lng"
FROM "circuits" c
JOIN "races" r ON c."circuitId" = r."circuitId"
WHERE r."name" = 'Australian Grand Prix';
```

Result:

```json
[[-34.9272, 138.617]]
```

## Phase 4: Agent Tracing

Status: wired and observed in local Langfuse.

Langfuse was configured through `.env`:

```env
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=http://localhost:3001
```

After restarting the agent server, `/answer` requests appeared in Langfuse. The
trace view showed the agent run structure and LLM spans. The most useful trace is
the Formula 1 duplicate-row case because it exercises:

```text
generate_sql -> execute -> verify -> revise -> execute -> verify
```

Available artifact:

- `screenshots/langfuse_trace.png`

Pending artifact:

- `screenshots/langfuse_tags.png`

For Phase 6, traces should be sent with metadata tags such as backend, model,
run type, and tuning iteration so slow requests can be filtered by experiment.

## Phase 5: Offline Evals

Status: pending implementation.

`evals/run_eval.py` still needs the Phase 5 implementation. The eval should:

- read `evals/eval_set.jsonl`,
- call `http://localhost:8001/answer`,
- execute both predicted SQL and gold SQL against the target SQLite DB,
- canonicalize result rows before comparison,
- report overall execution accuracy,
- report per-iteration pass rates to measure whether the verify/revise loop adds
  value,
- write `results/eval_baseline.json`.

Final pass rates should be reported from the real Qwen3-30B-A3B vLLM backend on
the H100, not from Token Factory development runs.

## Phase 6: SLO Diagnosis

Status: pending H100 load testing.

The target is:

```text
P95 end-to-end agent latency < 5s at 10+ RPS over 5 minutes
```

The final report should include:

- baseline load-test numbers,
- whether the target was hit,
- one or more metric-grounded tuning iterations,
- before/after Grafana screenshots,
- `results/eval_after_tuning.json` to show whether quality survived tuning.

Iteration log template:

```text
saw <metric symptom> -> hypothesized <cause> -> changed <one config/prompt lever> -> result was <measured change>
```

## Agent Value

The agent loop already demonstrated value qualitatively. In the Formula 1 manual
case, the first SQL returned repeated identical coordinate rows. The verifier
rejected the result, the revise node added `DISTINCT`, and the final answer
matched the expected unique coordinate row. This is exactly the intended
architecture: generation can be imperfect, execution exposes concrete behavior,
and verification/revision can repair the served answer.

The final quantitative claim should come from Phase 5 per-iteration pass rates.
If pass rate after revision is higher than pass rate after the first generation,
the loop is earning its extra latency. If not, the report should say that the
architecture did not pay for itself yet.

## What I Would Do With More Time

- Add richer schema context from BIRD evidence fields, not only column semantic
  names, so questions with domain-specific wording map more reliably to encoded
  columns and enum values.
- Add deterministic validators for common SQL mistakes: missing `LIMIT` for
  top-k questions, missing aggregation for "average/count" questions, and
  suspicious zero-row results when exact string matching may need normalization.
- Add a small prompt-regression set containing known failure modes discovered
  during manual testing, separate from the final eval set.
- Track token counts and latency per graph node in the eval output so quality
  gains can be compared against added latency.
- Tune prompts against the real vLLM endpoint after the H100 setup, because
  Token Factory behavior and local vLLM behavior may differ.
