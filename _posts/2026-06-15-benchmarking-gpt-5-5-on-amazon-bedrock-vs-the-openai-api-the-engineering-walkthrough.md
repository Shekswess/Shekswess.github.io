---
layout: post
title: "Benchmarking GPT-5.5 on Amazon Bedrock vs. the OpenAI API: The Engineering Walkthrough"
description: "A practical engineering walkthrough for reproducing Loka's GPT-5.5 benchmark across Amazon Bedrock and the OpenAI API."
author: shekswess
date: 2026-06-15 00:00:00 +0800
categories: [AI, LLM]
tags: [OpenAI, Amazon Bedrock, GPT-5.5, Benchmarking, GSM8K, AWS, Latency]
image: https://lokahq.github.io/gpt-bedrock-openai-benchmark/blog/assets/cover-banner-gpt.png
---

_**Original Source of the blog post: [Benchmarking GPT-5.5 on Amazon Bedrock vs. the OpenAI API: The Engineering Walkthrough](https://medium.com/loka-engineering/benchmarking-gpt-5-5-on-amazon-bedrock-vs-the-openai-api-the-engineering-walkthrough-3d57899fc058)**_

_**Written by Petar Kalinovski and Bojan Jakimovski**_

## The Engineering Walkthrough

The main post covers what we found: GPT-5.5 on Amazon Bedrock matches OpenAI API accuracy and comes out faster on every latency and throughput dimension we measured. This post covers how we ran it. If you want to reproduce the GSM8K numbers or adapt the harness for your own workload, this is the companion piece.

The concrete question behind the benchmark was simple:

> Does routing GPT-5.5 through Amazon Bedrock Mantle cost you anything in answer quality, latency, or reliability compared to the native OpenAI API?

We ran the full GSM8K test split, 1,319 grade-school math problems, through both provider paths at three reasoning levels (`medium`, `high`, `xhigh`), recording per-request streaming latency, output throughput, token usage, and failures. The benchmark harness is open and the results are saved in the repo. This post walks through the setup, the exact commands, the output format, and the design choices that made the comparison clean.

For the main findings, including the accuracy tables, latency charts, and the two regulated-industry copilots we built on the path that passed, read the main Loka writeup here: [How Loka Evaluates and Builds with Frontier Models on AWS](https://lokahq.github.io/gpt-bedrock-openai-benchmark/blog/)

## Why GSM8K

We needed a dataset that was long enough to produce stable latency distributions (1,319 requests per run means the p90 and p95 figures mean something), short enough to complete in a single session without batching or parallelism, and one where answer correctness is unambiguous and automatable. GSM8K satisfies all three. The final `####` answer format makes accuracy scoring a string comparison after normalization, with no LLM judge in the loop.

The math content also makes token budgets predictable. Each question fits in well under a hundred tokens. Reasoning answers are short — a few lines of working plus a numeric result. That keeps the benchmark from inflating latency through unusual output lengths and keeps costs bounded before you start.

## Tested Setup

The measured runs used:

| Setting | Value |
| --- | --- |
| Bedrock model ID | `openai.gpt-5.5` |
| Bedrock endpoint | Bedrock Mantle, `us-east-2` |
| OpenAI model ID | `gpt-5.5` |
| OpenAI endpoint | `https://api.openai.com/v1` |
| Reasoning efforts | `medium`, `high`, `xhigh` |
| Dataset | GSM8K test split, 1,319 questions |
| Concurrency | 1 (sequential) |
| Max output tokens | 4,096 |
| Auth (Bedrock) | AWS SigV4, `bedrock-mantle` service |
| Auth (OpenAI) | Bearer token via `OPENAI_API_KEY` |

Setting Value Bedrock model ID `openai.gpt-5.5` Bedrock endpoint Bedrock Mantle, `us-east-2` OpenAI model ID `gpt-5.5` OpenAI endpoint `https://api.openai.com/v1` Reasoning efforts `medium`, `high`, `xhigh` Dataset GSM8K test split, 1,319 questions Concurrency 1 (sequential) Max output tokens 4,096 Auth (Bedrock) AWS SigV4, `bedrock-mantle` service Auth (OpenAI) Bearer token via `OPENAI_API_KEY`

Sequential requests were a deliberate choice. Client-side concurrency is a real production variable, but it would have conflated provider queue effects with the underlying serving path. Running one request at a time isolated the thing we actually wanted to measure.

## Bootstrap The Environment

The repo uses `uv` for dependency management. Python 3.12 or later is required.

```bash
git clone https://github.com/LokaHQ/gpt-5-5-bedrock-openai-benchmark
cd gpt-5-5-bedrock-openai-benchmark
uv sync
```

This installs `httpx`, `boto3`, `python-dotenv`, and `tiktoken`. The dev group adds `pytest` and `ruff` if you also want the test suite and linter.

## Configure Authentication

The benchmark supports two authentication paths for Bedrock and one for OpenAI.

### Bedrock: AWS SigV4

This is the path used for the main runs. For a named profile:

```bash
export AWS_PROFILE=your-bedrock-profile
```

The SigV4 signer is built in via `botocore`. No extra env vars are needed as long as the active profile has `bedrock-mantle:*` and `aws-marketplace:*` permissions.

To verify authentication without making model calls:

```bash
AWS_PROFILE=your-bedrock-profile uv run python scripts/benchmark_openai_latency.py \
  --provider bedrock \
  --auth-debug
```

This prints the auth mode, endpoint, and signed header names without touching the model. A clean output looks like:

```text
auth_mode=sigv4
base_url=https://bedrock-mantle.us-east-2.api.aws/openai/v1
region=us-east-2
sigv4_service=bedrock-mantle
signed_header_names=authorization, content-type, host, x-amz-date, x-amz-security-token, x-client-request-id
```

### Bedrock: Bearer Token

If your setup authenticates with a Bedrock API key instead of AWS credentials, set it in `.env`. You can generate one in the Bedrock console under API keys (short-term keys last up to 12 hours), with the `aws-bedrock-token-generator package`, or for a long-term key via `aws iam create-service-specific-credential --service-name bedrock.amazonaws.com`:

```text
BEDROCK_OPENAI_API_KEY=your-bearer-token
```

### OpenAI API

Copy `.env.example` to `.env` and set your key:

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
```

The .env file is excluded from version control. Never commit it.

## Estimate Cost Before Running

Before running against the full test split, use the cost estimator to know what you are committing to:

```bash
uv run python scripts/estimate_openai_eval_cost.py \
  --dataset gsm8k \
  --split test \
  --model gpt-5.5 \
  --visible-output-tokens 256 \
  --reasoning-output-tokens 256
```

The estimator downloads the GSM8K test split, counts input tokens with `tiktoken`, and outputs a table like:

| Model | Samples | Avg input tok | P95 input tok | Output tok/sample | Standard cost | Batch cost |
| --- | --- | --- | --- | --- | --- | --- |
| `gpt-5.5` | 1,319 | 93.5 | 136 | 512 | $20.88 | $10.44 |


The `--reasoning-output-tokens` figure is an estimate. OpenAI bills reasoning tokens as output tokens even though they are not returned in API responses. The actual reasoning budget at `xhigh` is higher than `medium`, so this estimate is a lower bound for `xhigh` runs. The cost caps in the benchmark command give you a hard stop regardless.

## Run The Latency Benchmark

### Bedrock

```bash
AWS_PROFILE=your-bedrock-profile uv run python scripts/benchmark_openai_latency.py \
  --provider bedrock \
  --region us-east-2 \
  --model openai.gpt-5.5 \
  --dataset gsm8k \
  --split test \
  --warmups 1 \
  --reasoning-effort medium \
  --max-output-tokens 4096 \
  --concurrency 1 \
  --timeout-seconds 240 \
  --max-billable-tokens 750000 \
  --max-estimated-openai-cost-usd 10.00 \
  --output-dir result/gsm8k-gpt55-medium
```

### OpenAI API

```bash
uv run python scripts/benchmark_openai_latency.py \
  --provider openai \
  --model gpt-5.5 \
  --dataset gsm8k \
  --split test \
  --warmups 1 \
  --reasoning-effort medium \
  --max-output-tokens 4096 \
  --concurrency 1 \
  --timeout-seconds 240 \
  --max-billable-tokens 750000 \
  --max-estimated-openai-cost-usd 10.00 \
  --output-dir result/openai-gsm8k-gpt55-medium
```

Change `--reasoning-effort` to `high` or `xhigh` for the other runs.

### What the six main commands were

The six benchmark runs that produced the numbers in the main post used these exact folder names:

| Effort | Provider | Output folder |
| --- | --- | --- |
| `medium` | Bedrock | `result/gsm8k-gpt55-medium` |
| `high` | Bedrock | `result/gsm8k-gpt55-high` |
| `xhigh` | Bedrock | `result/gsm8k-gpt55-xhigh` |
| `medium` | OpenAI | `result/openai-gsm8k-gpt55-medium` |
| `high` | OpenAI | `result/openai-gsm8k-gpt55-high` |
| `xhigh` | OpenAI | `result/openai-gsm8k-gpt55-xhigh-merged-clean` |

The OpenAI `xhigh` folder name ends in `-merged-clean` because the original run hit a client-side DNS outage near the end — 181 requests failed to resolve on our own network, not on the provider's. We retried those separately and merged the clean records back into a single JSONL before scoring, so the merged result shows 1,319 / 0 — all prompts answered, no failures — which is the number reported in the main post.

That DNS outage was on our side and counts against neither provider. The genuine provider-side errors showed up elsewhere: the OpenAI `medium` and `high` runs each returned 5xx responses from OpenAI's edge (2× HTTP 503, 1× HTTP 520), while Bedrock returned zero errors across all 3,957 requests at every effort level.



## What the Harness Does During a Run

Each request goes through `run_once()`, which opens an `httpx` streaming connection and measures three timestamps:

- `TTFB` (time to first byte): when the first non-empty line arrives from the server. This is infrastructure-only: connection setup, auth, admission, routing.
- `TTFT` (time to first token): when the first `response.output_text.delta` event arrives. This includes model reasoning before the first output character is produced.
- total latency: when the stream closes after `[DONE]`

Token usage, input, output, and reasoning separately, comes from the `usage` block in the final `response.done` event. Output throughput is derived as:

```text
output_tokens / total_seconds
```

Each completed or failed record is written immediately to a partial JSONL file, so a mid-run crash does not destroy the entire sample.

The live progress line looks like:

```text
bedrock openai.gpt-5.5 run 142/1320: ok; in=87 out=44 reasoning=312; cost=$0.003456; cumulative_tokens=18,432; cumulative_est_openai_cost=$0.4891
```

The cost caps (`--max-billable-tokens`, `--max-estimated-openai-cost-usd`) check after each completed request and stop the run cleanly if either threshold is crossed.

## Output Files

Each run writes three files into the output directory:

```text
result/gsm8k-gpt55-medium/
├── openai_responses_latency_20260610T083101Z.jsonl
├── openai_responses_latency_20260610T083101Z_summary.json
└── openai_responses_latency_20260610T083101Z.md
```

The JSONL file has one flat record per request:

```json
{
  "provider": "bedrock",
  "model": "openai.gpt-5.5",
  "run_index": 142,
  "prompt_index": 141,
  "prompt_id": "gsm8k-test-141",
  "started_at": "2026-06-10T08:52:13.441Z",
  "ok": true,
  "ttfb_ms": 218.4,
  "ttft_ms": 1847.2,
  "total_ms": 2613.9,
  "input_tokens": 87,
  "output_tokens": 44,
  "reasoning_tokens": 312,
  "output_tokens_per_second": 50.2,
  "output_text": "The total is 14 apples.\n\n#### 14",
  "expected_answer": "14 apples.\n\n#### 14",
  "estimated_openai_cost_usd": 0.003456,
  "request_id": "...",
  "error": null
}
```

The summary JSON contains the LLMPerf-style aggregate statistics: full distributions (p25, p50, p75, p90, p95, p99, mean, min, max, stddev) for each metric across all successful requests.

The Markdown file is the same summary in readable form, including a failure log if any requests errored.

## Score Accuracy Offline

The accuracy scorer is a separate script that never touches the API. It works entirely from the saved JSONL files, which is important: you can re-score a run with an updated normalization rule without paying for another API call.

To score all six main runs at once:

```bash
uv run python scripts/score_gsm8k_accuracy.py --main-runs
```

This downloads the original GSM8K test split from GitHub, extracts the `#### <answer>` line from both the reference and the saved model response, normalizes both (strips commas, dollar signs, trailing periods, handles Unicode minus), and compares them. The output goes into each run's folder plus an aggregate comparison:

```text
result/
├── gsm8k_accuracy_main_runs.json
├── gsm8k_accuracy_main_runs.md
├── gsm8k-gpt55-medium/
│   ├── gsm8k_accuracy_summary.json
│   └── gsm8k_accuracy_summary.md
└── ...
```

To score a single run, pass `--benchmark-jsonl` and optionally `--output-dir`:

```bash
uv run python scripts/score_gsm8k_accuracy.py \
  --benchmark-jsonl result/gsm8k-gpt55-medium/openai_responses_latency_20260610T083101Z.jsonl
```

The normalization logic handles most of the common failure modes: a model writing `$14.00` instead of `14`, writing `14,000` with a comma, or using a Unicode minus. The scorer extracts the last `####` match in the output, so reasoning steps that mention intermediate numbers do not pollute the answer comparison.

## The Normalization Detail That Matters

The answer extractor uses the `####` prefix that GSM8K itself defines. Both the reference answers from the dataset and the model answers from the benchmark follow the same format because the benchmark prompt asks the model to use it:

```text
Solve this grade-school math problem. Show concise reasoning,
then put the final numeric answer on a separate line prefixed
with ####.
```

This makes the scorer deterministic and fast. The only edge cases are models that skip the `####` prefix entirely (scored as `missing_model_answer`) or produce a format the normalizer cannot parse to a canonical number. In practice, GPT-5.5 followed the format consistently across all 1,319 questions in every run.

## Running At Reduced Scale

Before committing to the full 1,319-question run, use `--limit` to test against a small slice:

```bash
AWS_PROFILE=your-bedrock-profile uv run python scripts/benchmark_openai_latency.py \
  --provider bedrock \
  --model openai.gpt-5.5 \
  --dataset gsm8k \
  --split test \
  --limit 20 \
  --reasoning-effort medium \
  --max-output-tokens 4096 \
  --output-dir result/smoke-test
```

This runs the first 20 questions, produces all the same output files, and lets you verify auth, output format, and metric collection before spending the full budget.

## Caveats

This benchmark is intentionally narrow:

- It measures sequential single-stream requests from one client location. Concurrent-request behavior and multi-region comparisons are out of scope.

- The dataset is grade-school math. Short math prompts with compact answers are not representative of all enterprise workloads. Longer prompts or longer outputs would shift the TTFT and throughput numbers.

- Latency and error rates are point-in-time properties of a serving fleet. Bedrock’s zero-error rate and lower TTFB reflect the fleet as it stood in June 2026, weeks after the Bedrock launch. Both can change as load grows.

- Cost estimates use OpenAI’s published list prices as of June 2026. Bedrock list prices differ slightly; the main post notes the specific numbers.

Those caveats do not change the main result: same model, same answer quality, faster serving on Bedrock with zero errors across nearly 4,000 requests. But they do matter when you translate these numbers to a production SLO.

## Takeaway

The useful part is not only the latency numbers. It is the measurement discipline.

The harness records every request, separates warmup from measured runs, writes partial results during the run, and scores accuracy offline from saved artifacts. That structure makes the comparison clean and reproducible. You can rerun against your own region, your own workload shape, and your own concurrency level and get numbers that mean something for your system, not just ours.

For teams evaluating OpenAI on Bedrock against the first-party API, that is the starting point: run the benchmark against your own traffic shape before committing an SLO to results from someone else’s setup.
