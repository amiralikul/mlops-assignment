# MLOps Assignment Report

## Scope and Current Status

This report currently documents the parts validated locally plus the H100
bring-up work completed so far. Agent development and tracing were first tested
locally against Nebius Token Factory, which exposes an OpenAI-compatible API.
The H100 VM is now reachable over SSH and vLLM startup has progressed through
model download/loading, but final Grafana metrics, eval pass rates, and SLO
claims still need to be collected after vLLM is serving successfully.

## Phase 1: vLLM Serving Configuration

Status: H100 bring-up in progress.

The H100 VM was created on Nebius with:

- `NVIDIA H100 80GB HBM3` / `NVIDIA H100 NVLink`,
- 16 vCPUs,
- 200 GiB RAM,
- 1.28 TiB boot disk,
- Ubuntu 24.04 LTS for NVIDIA GPUs.

SSH access required two fixes:

- attach a public IPv4 address to the VM,
- use the username from cloud-init user data (`amir`) and the matching local
  private key (`~/.ssh/github_work`).

The working SSH command with assignment port forwarding is:

```bash
ssh -i ~/.ssh/github_work \
  -L 3000:localhost:3000 \
  -L 9090:localhost:9090 \
  -L 3001:localhost:3001 \
  -L 8000:localhost:8000 \
  -L 8001:localhost:8001 \
  amir@89.169.124.184
```

On the VM, GPU and base tooling were verified:

- `nvidia-smi` showed one idle H100 80GB,
- Docker and Docker Compose were installed,
- Python 3.12 and git were installed,
- `uv` was installed under `~/.local/bin`.

The first vLLM startup attempts exposed three setup/configuration issues:

1. `Qwen2Tokenizer has no attribute all_special_tokens_extended`
   - Cause: `transformers==5.9.0` was installed with `vllm==0.10.2`.
   - Fix: pin/install `transformers<5`.
2. `fatal error: Python.h: No such file or directory`
   - Cause: Triton/Torch compile path needs Python development headers.
   - Fix: `sudo apt install -y python3-dev build-essential`.
3. KV cache startup failure with Qwen's default `max_model_len=262144`
   - Cause: vLLM needed 24 GiB of KV cache to serve one max-length request, but
     only 8.68 GiB was available after loading the 30B MoE weights.
   - Fix: reduce context length for this workload, starting with
     `--max-model-len 4096`.

The initial serving configuration is chosen around this workload:

- prompts are roughly 1.5K-3K tokens,
- outputs are short SQL or JSON snippets,
- each agent request usually performs 2-3 dependent LLM calls,
- the target SLO is P95 end-to-end agent latency under 5 seconds at 10+ RPS over
  a 5-minute window.

Current startup configuration to validate:

| Flag | Value | Reason |
|---|---:|---|
| `--model` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Fixed assignment model. |
| `--host` / `--port` | `0.0.0.0` / `8000` | Exposes an OpenAI-compatible endpoint for the agent and manual checks. |
| `--max-model-len` | `8192` | Avoids the observed 4096-token context failures for larger BIRD schemas while staying far below the model's 262K default. |
| `--max-num-seqs` | `64` | Allows enough concurrent short SQL generations for 10 RPS without letting the queue grow unbounded. |
| `--max-num-batched-tokens` | `8192` | Matches the tuned context window so long schema prompts can prefill without rejection. |
| `--enable-prefix-caching` | enabled | Reuses common prompt/schema prefixes when possible. |
| chunked prefill | enabled by vLLM | Helps prevent long prefills from blocking decode steps under bursty load. |

The intended command shape is:

```bash
uv run python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --enable-prefix-caching
```

Available artifact: `screenshots/vllm_manual_query.png`.

## Phase 2: Serving Observability

Status: H100 vLLM metrics are available and the Grafana dashboard has been
expanded beyond the starter panels.

Token Factory was useful for agent development, but it does not expose this
assignment's vLLM `/metrics` endpoint. The serving dashboard was therefore
validated against the H100 vLLM process on the Nebius VM.

Locally, the provisioned Grafana dashboard was found at:

```text
http://localhost:3000/d/vllm-serving/vllm-serving
```

The first local check, before vLLM was running on port `8000`, showed Prometheus
scraping the configured target but receiving connection refused:

```text
job: vllm
scrapeUrl: http://host.docker.internal:8000/metrics
health: down
error: connection refused
```

This confirmed that the dashboard/provisioning path was correct. After starting
vLLM on the H100 VM, `/metrics` exposed the expected counters, gauges, and
histograms, including:

- `vllm:num_requests_running`
- `vllm:num_requests_waiting`
- `vllm:request_success_total`
- `vllm:prompt_tokens_total`
- `vllm:generation_tokens_total`
- `vllm:e2e_request_latency_seconds_bucket`
- `vllm:time_to_first_token_seconds_bucket`
- `vllm:inter_token_latency_seconds_bucket`
- `vllm:request_queue_time_seconds_bucket`
- `vllm:request_prefill_time_seconds_bucket`
- `vllm:request_decode_time_seconds_bucket`
- `vllm:kv_cache_usage_perc`

The dashboard was expanded to cover:

- request concurrency and queueing,
- request throughput by finish reason,
- prompt and generation token throughput,
- p50/p95/p99 end-to-end latency,
- p95 queue, prefill, decode, and inference time,
- p50/p95 time to first token,
- p50/p95 inter-token latency,
- KV cache usage,
- prefix cache hit ratio,
- preemptions,
- prompt/generation token distribution,
- engine step token distribution.

The dashboard should answer three questions:

- Latency: are requests slow, and is the time in queue, prefill/TTFT, or decode?
- Throughput: how many requests and generated tokens are being served?
- KV cache: is there enough cache headroom for the current concurrency?

Available artifact:

- `screenshots/grafana_serving.png`

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

On the H100 VM, Langfuse keys were added to `.env` with `LANGFUSE_HOST` pointing
at the locally forwarded Langfuse service:

```env
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=http://localhost:3001
```

The agent server should be restarted after `.env` changes so the callback handler
is initialized with the new keys.

## Phase 5: Offline Evals

Status: implemented and run against the H100 vLLM-backed agent.

`evals/run_eval.py` now:

- reads `evals/eval_set.jsonl`,
- calls `http://localhost:8001/answer`,
- executes both predicted SQL and gold SQL against the target SQLite DB,
- canonicalizes result rows before comparison,
- scores the final answer and each generated/revised SQL attempt from the
  agent history,
- reports overall execution accuracy,
- reports per-iteration pass rates with carry-forward for stopped runs,
- writes `results/eval_baseline.json`.

Baseline eval command:

```bash
uv run python evals/run_eval.py --out results/eval_baseline.json
```

Baseline eval results:

| Metric | Value |
|---|---:|
| Questions | 30 |
| Correct | 14 |
| Execution accuracy | 46.7% |
| Errors | 0 |
| Average iterations | 1.53 |
| Agent latency P50 | 1.01s |
| Agent latency P95 | 2.74s |

Per-iteration carry-forward pass rate:

| Iteration | Correct | Accuracy |
|---:|---:|---:|
| 1 | 11 / 30 | 36.7% |
| 2 | 12 / 30 | 40.0% |
| 3 | 14 / 30 | 46.7% |

The verify/revise loop improved the eval score from 36.7% at the first SQL
attempt to 46.7% by the final served answer. That is a meaningful gain, so the
loop is doing useful work, although the remaining failures show that prompt and
schema grounding can still improve.

Available artifact:

- `results/eval_baseline.json`
- `screenshots/grafana_eval_run.png`

## Phase 6: SLO Diagnosis

Status: tuned configuration hit the latency target with clean request success.

The target is:

```text
P95 end-to-end agent latency < 5s at 10+ RPS over 5 minutes
```

Baseline load test command:

```bash
uv run python load_test/driver.py --rps 10 --duration 300 --out results/load_baseline.json
```

Baseline result:

| Metric | Value |
|---|---:|
| Requested RPS | 10.0 |
| Duration | 300s |
| Wall clock | 360.0s |
| Total requests | 3000 |
| Achieved RPS | 8.33 |
| OK | 673 |
| Timeouts | 1546 |
| HTTP errors | 323 |
| Client errors | 458 |
| Latency P50 | 68.69s |
| Latency P95 | 119.81s |
| Latency P99 | 120.51s |
| Max latency | 120.94s |

The baseline did not hit the SLO. The system could not sustain the offered 10
agent runs per second: most requests failed or timed out, and P95 latency was
close to the 120s client timeout. This pointed to queue saturation plus request
failure, not a small tail-latency miss.

Tuning iteration log:

```text
saw 2-3 LLM calls per agent run and vLLM/agent queues saturating
-> hypothesized verifier LLM calls were the biggest avoidable latency multiplier
-> added AGENT_VERIFY_MODE=fast, which accepts successful non-duplicate SQL
   executions without calling the LLM verifier
-> 60s smoke improved to 600/600 OK with P95 2.33s
```

```text
saw HTTP 500s during load and vLLM logs showing prompts slightly above 4096 tokens
-> hypothesized some DB schemas exceeded the configured context window
-> increased vLLM max context from 4096 to 8192 and set max_num_batched_tokens to
   8192, with max_num_seqs=64
-> context-length failures disappeared
```

```text
saw remaining HTTP 500s for european_football_2 and debit_card_specializing
-> traced root cause to schema rendering of SQLite foreign keys with NULL target
   columns and NULL BIRD metadata descriptions
-> skipped malformed FK targets and ignored NULL descriptions
-> 120-request reproduction went from repeated 500s to 120/120 OK
```

```text
saw a few long-tail timeouts with no vLLM errors
-> hypothesized generated SQL could run too long because sqlite3 timeout only
   covers lock waits, not query execution time
-> added a SQLite progress handler to interrupt queries after the execution
   budget
-> final 5-minute run completed 3000/3000 OK with P95 2.15s
```

Final load test command:

```bash
uv run python load_test/driver.py --rps 10 --duration 300 --out results/load_tuned_timeout_final.json
```

Final load test result:

| Metric | Value |
|---|---:|
| Requested RPS | 10.0 |
| Duration | 300s |
| Wall clock | 301.1s |
| Total requests | 3000 |
| Achieved RPS | 9.96 |
| OK | 3000 |
| Timeouts | 0 |
| HTTP errors | 0 |
| Client errors | 0 |
| Latency P50 | 0.85s |
| Latency P95 | 2.15s |
| Latency P99 | 6.01s |
| Max latency | 23.19s |

The final run met the latency target and served the offered 3000 requests over
the 5-minute window without errors. The reported achieved RPS is 9.96 because
the driver includes about 1s of startup/drain overhead in wall-clock time; the
offered load was 10 RPS for 300s.

Post-tuning eval quality survived:

| Metric | Baseline | Tuned |
|---|---:|---:|
| Execution accuracy | 46.7% | 46.7% |
| Correct | 14 / 30 | 14 / 30 |
| Average iterations | 1.53 | 1.13 |
| Eval latency P50 | 1.01s | 0.55s |
| Eval latency P95 | 2.74s | 1.33s |

Available artifacts:

- `results/load_baseline.json`
- `results/load_tuned_timeout_final.json`
- `results/eval_after_tuning.json`
- `screenshots/grafana_before.png`
- `screenshots/grafana_after.png`

## Agent Value

The agent loop already demonstrated value qualitatively. In the Formula 1 manual
case, the first SQL returned repeated identical coordinate rows. The verifier
rejected the result, the revise node added `DISTINCT`, and the final answer
matched the expected unique coordinate row. This is exactly the intended
architecture: generation can be imperfect, execution exposes concrete behavior,
and verification/revision can repair the served answer.

The final eval supports that claim quantitatively. First-generation SQL was
correct on 12/30 questions (40.0%). After verification and revision, final
accuracy reached 14/30 (46.7%). The two extra correct answers came from cases
where execution feedback exposed a concrete issue that could be repaired. After
the fast verifier tuning, this did not require paying an LLM-verifier call on
every clean execution: average iterations dropped from 1.53 to 1.13, eval P95
latency dropped from 2.74s to 1.33s, and final accuracy stayed at 46.7%.

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
