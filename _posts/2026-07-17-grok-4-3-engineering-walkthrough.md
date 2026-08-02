---
layout: post
title: "Benchmarking Grok 4.3 on Amazon Bedrock Mantle vs. the xAI API: The Engineering Walkthrough"
description: "A practical engineering walkthrough for reproducing Loka's Grok 4.3 benchmark across Amazon Bedrock Mantle and the xAI API."
author: shekswess
date: 2026-07-17 00:00:00 +0200
permalink: /grok-4-3-engineering-walkthrough.html
categories: [AI, LLM]
tags: [xAI, Grok 4.3, Amazon Bedrock, Bedrock Mantle, Benchmarking, GSM8K, AWS]
image: /assets/images/grok-benchmark/background.png
source_title: "Benchmarking Grok 4.3 on Amazon Bedrock Mantle vs. the xAI API: The Engineering Walkthrough"
source_url: "https://medium.com/loka-engineering/benchmarking-grok-4-3-07ecfa9a48f3"
source_authors: ["Nina Cvetkovska", "Bojan Jakimovski"]
---

_**Original Source of the blog post: [Benchmarking Grok 4.3 on Amazon Bedrock Mantle vs. the xAI API: The Engineering Walkthrough with Loka](https://medium.com/loka-engineering/benchmarking-grok-4-3-07ecfa9a48f3)**_

_**Blog was written with my colleague from Loka & AWS: Nina Cvetkovska**_

## The Engineering Walkthrough

The main post covers what we found: Grok 4.3 lands in the same 96–97% accuracy band on GSM8K whether you reach it through Amazon Bedrock Mantle or the first-party xAI API, and `low` reasoning effort captures most of the accuracy gain at a fraction of the latency and token cost of `medium` or `high`. This post covers how we ran it. If you want to reproduce the numbers or adapt the harness for your own workload, this is the companion piece.

The concrete question behind the benchmark was simple:

> *Does routing Grok 4.3 through Amazon Bedrock Mantle cost you anything in answer quality, latency, or reliability compared to the native xAI API?*

We ran the full GSM8K test split (1,319 grade-school math problems) through both provider paths at four reasoning-effort levels (`none`, `low`, `medium`, `high`), recording per-request streaming latency, output throughput, token usage, and failures. That is eight runs and 10,552 scored requests in total. The benchmark harness is open and the results are saved in the repo. This post walks through the setup, the exact commands, the output format, and the design choices that made the comparison clean.

For the main findings (accuracy tables, latency charts, and the enterprise deployment read), see the main Loka writeup: [The Many Paths to Grok 4.3]({{ '/the-many-paths-to-grok-4-3.html' | relative_url }}).

## Why GSM8K

We needed a dataset that was long enough to produce stable latency distributions (1,319 requests per run means the p95 figures mean something), short enough to complete in a single session without batching or parallelism, and one where answer correctness is unambiguous and automatable. GSM8K satisfies all three. The final `####` answer format makes accuracy scoring a string comparison after normalization, with no LLM judge in the loop. The harness pulls the split directly from the raw JSONL files in the openai/grade-school-math GitHub repository, so there is no dataset-library dependency to pin.

The math content also makes token budgets predictable. Each question fits in well under a hundred tokens, and answers are a few lines of working plus a numeric result. That keeps the benchmark from inflating latency through unusual output lengths and keeps costs bounded before you start. It also gave us a clean lens on Grok 4.3's configurable reasoning effort: when the task is this uniform, any accuracy or latency difference between effort levels is easy to attribute.

## Tested Setup

The measured runs used:

| Setting | Value |
|---|---|
| Bedrock model ID | `xai.grok-4.3` |
| Bedrock endpoint | Bedrock Mantle, `us-west-2` (`https://bedrock-mantle.us-west-2.api.aws/openai/v1`) |
| xAI model ID | `grok-4.3` |
| xAI endpoint | `https://api.x.ai/v1` |
| Reasoning efforts | `none`, `low`, `medium`, `high` |
| Dataset | GSM8K test split, 1,319 questions |
| Concurrency | 1 (sequential) |
| Max output tokens | 4096 |
| Streaming | enabled |
| Auth (Bedrock) | AWS SigV4 via an SSO profile, signed for the `bedrock-mantle` service (bearer tokens also supported) |
| Auth (xAI) | Bearer token via `XAI_API_KEY` |

Sequential requests were a deliberate choice. Client-side concurrency is a real variable in production, but it conflates queue effects with the underlying provider infrastructure. Running one request at a time isolates the thing we were measuring: what does each provider's serving path cost a single in-flight request?

One more Grok-specific note on the setup: reasoning in Grok 4.3 is always active by design, and the `reasoning.effort` parameter controls the budget, with `none` disabling it. The harness sends `"reasoning": {"effort": "<level>"}` in the request body on every run, including the `none` runs, where the value is passed through explicitly. It does not request reasoning summaries or encrypted reasoning content, so reasoning shows up only as a token count in the usage data, never as visible stream text.

## Bootstrap the Environment

The repo uses `uv` for dependency management and requires Python 3.12 or newer.

```
git clone https://github.com/NineCvet/grok-bedrock-xapi-benchmark.git
cd grok-bedrock-xapi-benchmark
uv sync
```

This installs the four runtime dependencies: `httpx` (streaming HTTP client), `boto3` (SigV4 signing via botocore), `python-dotenv` (loads the `.env` file), and `tiktoken` (token counting in the cost estimator). The dev group brings in `pytest` and `ruff`; `uv run pytest` and `uv run ruff check scripts tests` are the repo's checks.

## Configure Authentication

The benchmark script supports two auth modes for Bedrock Mantle and picks between them automatically: if a bearer key is present in the environment it uses that, otherwise it falls back to SigV4 request signing with whatever AWS credentials botocore can resolve.

### Bedrock: AWS SigV4 (what we used)

Our runs used an AWS SSO profile. Log in, then pass the profile to the script through the standard `AWS_PROFILE` variable:

```
aws sso login --profile loka-ml
```

The harness signs each request with SigV4 for the `bedrock-mantle` service in the target region. The default endpoint is `https://bedrock-mantle.<region>.api.aws/openai/v1`; override it with `BEDROCK_GROK_BASE_URL` if you need to.

Before spending anything, verify the auth path with `--auth-debug`. It builds and signs a request but never sends it:

```
AWS_PROFILE=loka-ml uv run python scripts/benchmark_grok_latency.py \
  --provider bedrock \
  --auth-debug
```

```
auth_mode=sigv4
base_url=https://bedrock-mantle.us-west-2.api.aws/openai/v1
region=us-west-2
sigv4_service=bedrock-mantle
signed_header_names=Authorization, Content-Type, X-Amz-Date, X-Client-Request-Id
```

### Bedrock: bearer token

If you have a Bedrock API key instead, set `BEDROCK_GROK_API_KEY` (or `AWS_BEARER_TOKEN_BEDROCK`); when either is present it takes precedence over SigV4 and the script sends a plain `Authorization: Bearer` header. The same `--auth-debug` command confirms which mode is active.

### xAI API

There is no committed env template; create a `.env` in the repo root with your key:

```
cat > .env <<'EOF'
XAI_API_KEY=xai-your-key-here
EOF
```

The script loads it automatically at startup. The default endpoint is `https://api.x.ai/v1`, overridable with `XAI_BASE_URL`. The `.env` file is excluded from version control. Never commit it.

## Estimate Cost Before Running

The repo has a standalone estimator that downloads the GSM8K split, counts input tokens with `tiktoken`, and projects cost from a configurable output-token budget:

```
uv run python scripts/estimate_grok_eval_cost.py \
  --dataset gsm8k \
  --split test \
  --model xai.grok-4.3 \
  --visible-output-tokens 512 \
  --reasoning-output-tokens 1024 \
  --output-dir result/cost-estimates
```

Its output (this is the real pre-run estimate saved in the repo):

```
| Model          | Samples | Avg input tok | P95 input tok | Output tok/sample | Standard cost | Batch cost |
| -------------- | ------: | ------------: | ------------: | ----------------: | ------------: | ---------: |
| `xai.grok-4.3` |    1319 |          93.5 |           136 |             1,536 |       $5.2191 |    $2.6095 |
```

The `--reasoning-output-tokens` figure is deliberately an estimate: Grok bills reasoning tokens as output tokens even though they are not visible in responses, so the ceiling depends on the effort level you choose. The $5.22 projection assumed every request burns the full budget; the actual runs came in well under it because most GSM8K answers finish in a few hundred tokens. The realized totals from the result files: Bedrock runs cost $0.4418 (`none`) to $2.4400 (`high`), and xAI runs cost $0.5883 (`none`) to $2.2431 (`high`).

## Run the Latency Benchmark

### Bedrock Mantle (one reasoning effort)

```
AWS_PROFILE=loka-ml uv run python scripts/benchmark_grok_latency.py \
  --provider bedrock \
  --region us-west-2 \
  --model xai.grok-4.3 \
  --dataset gsm8k \
  --split test \
  --warmups 1 \
  --reasoning-effort low \
  --max-output-tokens 4096 \
  --concurrency 1 \
  --timeout-seconds 360 \
  --max-attempts 0 \
  --retry-backoff-seconds 2 \
  --retry-policy all-retryable \
  --max-billable-tokens 750000 \
  --max-estimated-grok-cost-usd 5.00 \
  --output-dir result/bedrock-grok43-low
```

The retry flags matter on the Bedrock path, for reasons covered in the reliability section below: `--retry-policy all-retryable` treats a provider response that explicitly asks for a retry as retryable, and `--max-attempts 0` retries until the request goes through, so the run finishes with a complete 1,319-row dataset.

### xAI API (one reasoning effort)

```
uv run python scripts/benchmark_grok_latency.py \
  --provider xai \
  --model grok-4.3 \
  --dataset gsm8k \
  --split test \
  --warmups 1 \
  --reasoning-effort low \
  --max-output-tokens 4096 \
  --concurrency 1 \
  --timeout-seconds 360 \
  --max-attempts 3 \
  --retry-backoff-seconds 2 \
  --max-billable-tokens 750000 \
  --max-estimated-grok-cost-usd 5.00 \
  --output-dir result/xai-grok43-low
```

Change the reasoning-effort flag to `none`, `medium`, or `high` and update the output directory accordingly for the other runs.

### What the eight main commands were

The eight benchmark runs that produced the numbers in the main post used these output folders:

| Effort | Provider | Output folder |
|---|---|---|
| none | Bedrock | `result/bedrock-grok43-none` |
| low | Bedrock | `result/bedrock-grok43-low` |
| medium | Bedrock | `result/bedrock-grok43-medium` |
| high | Bedrock | `result/bedrock-grok43-high` |
| none | xAI | `result/xai-grok43-none` |
| low | xAI | `result/xai-grok43-low` |
| medium | xAI | `result/xai-grok43-medium` |
| high | xAI | `result/xai-grok43-high` |

There is one extra folder next to these: `result/bedrock-grok43-low-error-examples` holds the partial JSONL from an early Bedrock `low` pass that we stopped after ten measured prompts. It exists on purpose, as raw evidence for the reliability section below.

## What the Harness Does During a Run

Each request goes through `run_once`, which opens a streaming connection and measures three timestamps:

- **TTFB** (time to first byte): the moment the first non-empty line arrives from the server, before any token is parsed. This is infrastructure only: connection setup, auth, admission, routing.
- **TTFT** (time to first token): the moment the first visible output token event arrives. With reasoning enabled this includes the model's thinking time before the first output character, which is why the main post's TTFT numbers diverge so sharply between the two paths at `low` effort (3.14s on Bedrock vs 0.71s on xAI). Concretely, the harness marks TTFT on the first streamed `response.output_text.delta` event that carries text.
- **Total latency**: the moment the stream closes.

Token usage (input, output, and reasoning tokens separately) comes from the `usage` object on the stream's completion event, where reasoning tokens sit under `output_tokens_details.reasoning_tokens`. Output throughput is derived per request as output tokens divided by total stream duration.

Each completed or failed record is written to a partial JSONL file immediately, so a crash partway through does not lose the records collected so far.

The live progress line looks like this (a real line from the Bedrock `low` run, reconstructed from the saved record):

```
bedrock xai.grok-4.3 run 44/1320: ok; in=151 out=307 reasoning=258; cost=$0.000956; cumulative_tokens=21,509; cumulative_est_grok_cost=$0.046901
```

Three run guards keep a misbehaving run from burning budget: `--max-billable-tokens` and `--max-estimated-grok-cost-usd` stop the run when cumulative usage crosses a cap (we set 750,000 tokens and $5.00), and `--timeout-seconds` bounds any single request (we set 360).

## Output Files

Each run writes three files into the output directory, plus the partial JSONL that accumulates during the run. The accuracy scorer adds two more later. This is the real Bedrock `low` folder:

```
result/bedrock-grok43-low/
├── grok_responses_latency_20260621T185415Z_partial.jsonl
├── grok_responses_latency_20260621T225049Z.jsonl
├── grok_responses_latency_20260621T225049Z_summary.json
├── grok_responses_latency_20260621T225049Z.md
├── gsm8k_accuracy_summary.json   (written later by the scorer)
└── gsm8k_accuracy_summary.md     (written later by the scorer)
```

The JSONL file has one record per request. This is a real record from the Bedrock `low` run, with the request ID shortened:

```json
{
  "attempts": 1,
  "error": null,
  "estimated_grok_cost_usd": 0.000697,
  "expected_answer": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3",
  "input_tokens": 98,
  "model": "xai.grok-4.3",
  "ok": true,
  "output_chars": 125,
  "output_chars_per_second": 77.96,
  "output_text": "Blue fiber requires 2 bolts. White fiber is half of that amount, or 1 bolt.  \nAdding both gives a total of 3 bolts.  \n\n#### 3",
  "output_tokens": 230,
  "output_tokens_per_second": 143.45,
  "prompt_id": "gsm8k-test-1",
  "prompt_index": 1,
  "prompt_text": "Solve this grade-school math problem. Show concise reasoning, then put the final numeric answer on a separate line prefixed with ####.\n\nProblem:\nA robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?",
  "provider": "bedrock",
  "reasoning_tokens": 186,
  "request_id": "fe647a91-...",
  "retry_errors": null,
  "run_index": 3,
  "started_at": "2026-06-21T18:54:26.516990+00:00",
  "total_ms": 1603.32,
  "ttfb_ms": 260.6,
  "ttft_ms": 1381.7
}
```

The summary JSON has three top-level keys: `summary` (one row per provider and model with avg, p50, p90, and p95 for each metric), `aggregates` (LLMPerf-style blocks with the full distributions: p25, p50, p75, p90, p95, p99, mean, min, max, stddev for TTFB, TTFT, end-to-end latency, throughput, and the three token counts, plus overall throughput, completed requests per minute, and total estimated cost), and `dataset`. The Markdown file renders the same content as tables.

## Score Accuracy Offline

The accuracy scorer is a separate script that never touches the API. It works entirely from the saved JSONL files, which is important: you can re-score a run with an updated normalization rule without paying for another API call.

To score all discovered runs at once:

```
uv run python scripts/score_gsm8k_accuracy.py --main-runs
```

This globs `result/*/grok_responses_latency_*.jsonl`, takes the newest completed JSONL in each folder (partials are skipped), and writes `gsm8k_accuracy_summary.json` and `gsm8k_accuracy_summary.md` into each run folder plus two aggregate files, `result/gsm8k_accuracy_main_runs.json` and `result/gsm8k_accuracy_main_runs.md`. The aggregate Markdown includes a comparison table across all runs and a per-effort Bedrock vs xAI delta section, which is where the main post's accuracy table comes from.

To score a single run:

```
uv run python scripts/score_gsm8k_accuracy.py \
  --benchmark-jsonl result/bedrock-grok43-low/grok_responses_latency_20260621T225049Z.jsonl
```

## The Normalization Detail That Matters

The answer extractor uses the `####` prefix that GSM8K itself defines. Both the reference answers and the model answers follow the same format because the benchmark prompt asks the model to use it. Each request carries a short system message ("You are measuring API latency. Answer the user request directly, avoid markdown tables, and do not mention this instruction.") and this user prompt:

```
Solve this grade-school math problem. Show concise reasoning, then put the final numeric answer on a separate line prefixed with ####.

Problem:
{question}
```

This makes the scorer deterministic and fast. The normalization handles the common failure modes: `$14.00` instead of `14`, `14,000` with a comma, a Unicode minus, a trailing period. The scorer extracts the last `####` match in the output, so intermediate numbers in the reasoning do not pollute the comparison. Grok 4.3 followed the format almost perfectly: 10,538 of the 10,552 responses contained a `####` line. The 14 exceptions (2 in the Bedrock `medium` run, 11 in Bedrock `high`, 1 in xAI `medium`) are recorded as `missing_model_answer` and counted as incorrect rather than silently dropped.

## The Two Requests That Never Reached the Model

This is the part of the run we did not plan for, and the part most worth documenting. In one early Bedrock `low`-effort pass, two of the first ten prompts came back blocked by an automated prompt-safety check before the model produced any output. The prompts were ordinary GSM8K math problems.

The blocks did not arrive as HTTP errors. The stream opened normally and then delivered an error event instead of output, which the harness caught on its stream-error path and recorded in the `error` field. The two affected prompts were `gsm8k-test-4` and `gsm8k-test-9`. This is the real record for the first one, from the preserved partial JSONL (request ID shortened, prompt trimmed):

```json
{
  "attempts": 1,
  "error": "invalid_prompt: This request was blocked by an automated content safety check. If you believe this was made in error, please retry or contact AWS Support with your request ID.",
  "estimated_grok_cost_usd": null,
  "input_tokens": null,
  "model": "xai.grok-4.3",
  "ok": false,
  "output_tokens": null,
  "prompt_id": "gsm8k-test-4",
  "prompt_text": "Solve this grade-school math problem. [...] Problem:\nEvery day, Wendi feeds each of her chickens three cups [...]",
  "provider": "bedrock",
  "reasoning_tokens": null,
  "request_id": "147396ed-...",
  "run_index": 6,
  "total_ms": 3159.35,
  "ttfb_ms": null,
  "ttft_ms": null
}
```

Note what the record structure gives you: `ok` is false, the token fields are null, TTFB and TTFT are null because no byte of model output ever arrived, and the exact provider message is preserved. We stopped that pass after ten prompts and kept its partial JSONL in `result/bedrock-grok43-low-error-examples` as evidence.

The error message itself says to retry, so for the measured runs we set `--retry-policy all-retryable` with no attempt cap, and every blocked request eventually went through with exponential backoff. The retries are not hidden: each record carries an `attempts` count and a `retry_errors` list, so the full history is auditable in the saved artifacts. Across the completed Bedrock runs, the number of requests that hit at least one content-safety block before succeeding was 58 (`none`), 581 (`low`), 550 (`medium`), and 567 (`high`), with the worst case needing 10 attempts. The four xAI runs saw zero content-safety blocks; the only retry on that path in all 5,276 requests was a single connection reset. All eight runs finished 1,319 completed, 0 failed. The recorded latency for a retried request is the latency of the final successful attempt, so the retry mechanics inflate wall-clock time, not the per-request distributions.

The engineering takeaway is that the harness treats a blocked request, a timeout, and a wrong answer as three different record types, and the scorer keeps them separate. Accuracy is computed over scoreable answers; reliability is reported alongside it, not blended into it.

## Running at Reduced Scale

Before committing to the full 1,319-question run, use `--limit` to test against a small slice:

```
AWS_PROFILE=loka-ml uv run python scripts/benchmark_grok_latency.py \
  --provider bedrock \
  --region us-west-2 \
  --model xai.grok-4.3 \
  --dataset gsm8k \
  --split test \
  --limit 10 \
  --warmups 1 \
  --reasoning-effort none \
  --max-output-tokens 4096 \
  --concurrency 1 \
  --output-dir result/bedrock-grok43-none-pilot
```

This runs the first 10 questions, produces all the same output files, and lets you verify auth, output format, and metric collection before spending the full budget. Our own pilot is saved in `result/bedrock-grok43-none-pilot`.

## Caveats

This benchmark is intentionally narrow:

- It measures sequential single-stream requests from one client location. Concurrent-request behavior and multi-region comparisons are out of scope.
- The dataset is grade-school math. Short prompts with compact answers are not representative of all enterprise workloads; longer prompts or outputs would shift the TTFT and throughput numbers. In particular, the flat accuracy curve above `low` effort is a property of this workload, not a universal claim about reasoning budgets.
- Latency, tail behavior, and safety-filter behavior are point-in-time properties of a serving fleet. Our numbers reflect the Mantle path as it stood in June 2026, weeks after the Bedrock launch. All of it can change as the fleet matures.
- Cost figures are estimates from list prices as of June 2026. The harness prices both paths at xAI's published grok-4.3 rates ($1.25 per million input tokens, $2.50 per million output tokens, with reasoning billed as output); if your Bedrock pricing differs, override with `--input-price-per-mtok` and `--output-price-per-mtok` and the per-record costs recompute.

Those caveats do not change the main result: **same model quality through both doors, with the differences showing up in latency shape and reliability behavior rather than accuracy.** But they matter when you translate these numbers into a production SLO.

## Takeaway

The useful part is not only the numbers. It is the measurement discipline.

The harness records every request, writes partial results during the run, separates blocked requests from wrong answers, and scores accuracy offline from saved artifacts. That structure is what made an honest comparison possible, including the part we did not expect to find. You can rerun it against your own region, your own workload shape, and your own concurrency level and get numbers that mean something for your system, not just ours.

For teams evaluating Grok 4.3 on Bedrock Mantle against the first-party xAI API, that is the starting point: run the benchmark against your own traffic shape before committing an SLO to results from someone else's setup.
