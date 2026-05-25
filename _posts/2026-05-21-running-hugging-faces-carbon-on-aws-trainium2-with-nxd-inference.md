---
layout: post
title: "Running Hugging Face's Carbon on AWS Trainium2: The Engineering Walkthrough"
description: "The engineering walkthrough for running Hugging Face Bio Carbon on a single AWS Trainium2 instance with NxD Inference."
author: shekswess
date: 2026-05-21 00:00:00 +0800
categories: [AI, LLM]
tags: [AWS Trainium2, NxD Inference, Neuron, Hugging Face, Carbon, Bio AI, Genomics]
image: https://lokahq.github.io/carbon-neuronx-distributed-inference/assets/carbon-trainium2-cover.png
---


_**Original Source of the blog post: [Running Hugging Face's Carbon on AWS Trainium2 with NxD Inference](https://medium.com/loka-engineering/running-hugging-faces-carbon-on-aws-trainium2-with-nxd-inference-ecdee6c2f4ce)**_

_**Written by Bojan Jakimovski**_

The Engineering Walkthrough

[Hugging Face Bio's Carbon](https://huggingface.co/collections/HuggingFaceBio/carbon) release is interesting for two reasons at once. First, it is a biology model. Second, it is not an infrastructure outlier. Many genomic models come with custom architectures, specialized operators, or runtime assumptions that make deployment harder to evaluate. Carbon is different: it is an open DNA foundation model family with 500M, 3B, and 8B checkpoints, Apache 2.0 weights, a Llama-style causal backbone, and a DNA-native 6-mer tokenizer.

That made it a natural Trainium2 test, with one concrete question behind it:

> Can a freshly released open genomic model run through AWS NxD Inference on a single `trn2.3xlarge` without rewriting the model?

The short answer is yes. We compiled and benchmarked all three Carbon checkpoints on one `trn2.3xlarge`: one Trainium2 accelerator, four logical NeuronCores, 96 GB of accelerator memory, BF16 weights, and tensor parallel degree 4.

This post is the [code-side companion](https://github.com/LokaHQ/carbon-neuronx-distributed-inference) to the main benchmark writeup. The main article covers the business and platform story. This one covers the practical path: tokenizer handling, compile shape, benchmark harness, result interpretation, and what did not improve the baseline.

For the full report, including the Trainium2 business case, A100 reference comparison, and cost-normalized throughput charts, read the main Loka writeup: [Running Hugging Face Carbon on AWS Trainium2 with NxD Inference](https://lokahq.github.io/carbon-neuronx-distributed-inference/).

### Why Carbon Was a Good Fit for NxD Inference

Carbon is domain-specific, but its architecture is deliberately familiar. Under the hood, the configuration advertises a Llama causal model path. That matters because NxD Inference already has strong support for Llama-style autoregressive serving on Trainium and Inferentia.

The unusual part is the tokenizer. Carbon uses non-overlapping 6-mer DNA tokens. Instead of treating every nucleotide as one token, it groups six bases into one model token. A 1080 bp DNA sequence becomes roughly 180 Carbon DNA tokens when the tokenizer is in DNA mode.

That token efficiency is the reason the benchmark is operationally interesting. Shorter token sequences reduce attention cost, keep the runtime path close to a standard Transformer serving stack, and make a single Trainium2 accelerator a credible first deployment unit.

### Tested Platform

The measured runs used:

| Component | Setting |
| --- | --- |
| Instance | `trn2.3xlarge` |
| Accelerator | 1 Trainium2 device, 4 logical NeuronCores, `LNC=2` |
| Runtime | PyTorch 2.9.1, torch-neuronx 2.9.0, torch-xla 2.9.0, Neuron SDK 2.29 stack |
| NxDI | `neuronx-distributed-inference==0.9.0` |
| Parallelism | `tp_degree=4`, `batch_size=16` for the A100-style benchmark |
| Baseline shape | `max_context_length=256`, `seq_len=512`, `max_new_tokens=170` |
| Prompt format | DNA sequence wrapped as `<dna>...</dna>` |

The most important non-obvious setting is `LNC=2`, which exposes four logical NeuronCores from the single Trainium2 device. We used `tp_degree=4` so the model shards across those logical cores.

### Bootstrap the Environment

Start from a Neuron DLAMI or Neuron DLC where the Neuron driver and runtime are already configured. The repository helper script does three things: creates the Python environment, installs the Neuron SDK 2.29/PyTorch 2.9 stack, and installs a patched NxD Inference checkout.

```bash
git clone <repository-url>
cd carbon-neuronx-distributed-inference
NXDI_REF=main ./scripts/bootstrap_nxdi.sh
source .venv-carbon-sdk29/bin/activate
```

The Carbon-specific NxD Inference patch is intentionally small. The important point is that Carbon does not require a model rewrite or model-specific kernels for the baseline path. The compatibility work is around configuration/tokenizer handling and keeping the model on NxDI's existing Llama causal path.

### Download the Carbon Checkpoints

Carbon uses Hugging Face custom tokenizer code, so make sure the checkpoints are downloaded with the repository files intact.

```bash
huggingface-cli login
./scripts/download_carbon_models.sh "$HOME/models/carbon"
```

To download a single checkpoint:

```bash
MODELS="Carbon-500M" ./scripts/download_carbon_models.sh "$HOME/models/carbon"
```

### The Tokenizer Detail That Matters

This is the part most likely to poison the benchmark if missed. Carbon requires trusted custom tokenizer code. DNA prompts should also use Carbon's DNA mode prefix: `<dna>{sequence}`. Without those two details, the model may still run, but you are not measuring the intended compact 6-mer DNA path.

In Python:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "/home/ubuntu/models/carbon/Carbon-500M",
    trust_remote_code=True,
    padding_side="right",
)
tokenizer.pad_token_id = 151643
inputs = tokenizer(
    ["<dna>ACGTTGCAACGTTGCAACGTTGCA"],
    return_tensors="pt",
    padding=True,
)
```

From the NxD Inference demo entrypoint, the same ideas show up as CLI flags and prompt formatting:

```bash
inference_demo \
 -model-type llama \
 -task-type causal-lm \
 run \
 -model-path "$HOME/models/carbon/Carbon-500M" \
 -compiled-model-path "$HOME/compiled/carbon/Carbon-500M-tp4-bs16-s512-c256-n170-bf16" \
 -trust-remote-code \
 -pad-token-id 151643 \
 -prompt "<dna>ACGTTGCAACGTTGCAACGTTGCAACGTTGCA"
```

The pad token id used in the measured artifacts was `151643`.

### Compile the Baseline Artifacts

The A100-style benchmark shape mirrors the high-level workload used in the Carbon technical report: 16 DNA prompts, approximately 1020 bp input per prompt, and approximately 1020 bp generated per prompt.

The compile defaults were:

| Setting | Value |
| --- | --- |
| Tensor parallel degree | `TP_DEGREE=4` |
| Batch size | `BATCH_SIZE=16` |
| Max context length | `MAX_CONTEXT_LENGTH=256` |
| Sequence length | `SEQ_LEN=512` |
| Max length | `MAX_LENGTH=512` |
| Max new tokens | `MAX_NEW_TOKENS=170` |
| Pad token id | `PAD_TOKEN_ID=151643` |
| Logical NeuronCore config | `LOGICAL_NC_CONFIG=2` |
| Dtype | `bfloat16` |
| Sampling | on-device sampling, `top_k=1`, `top_p=1.0`, `temperature=1.0` |

Compile all three baseline artifacts:

```bash
./scripts/compile_carbon.sh Carbon-500M
./scripts/compile_carbon.sh Carbon-3B
./scripts/compile_carbon.sh Carbon-8B
```

Compiled artifacts are written under `$HOME/compiled/carbon/`.

### Run the A100-Style Batch Benchmark

The batch benchmark creates 16 synthetic DNA prompts and measures generated base-pair throughput. Generation is greedy direct 6-mer decoding, with Factorized Nucleotide Supervision base-pair marginal decoding disabled.

```bash
./scripts/run_a100_style_benchmark.sh Carbon-500M
./scripts/run_a100_style_benchmark.sh Carbon-3B
./scripts/run_a100_style_benchmark.sh Carbon-8B
```

The benchmark harness does the same basic flow you would expect from a Hugging Face generation adapter:

```python
tokenizer = AutoTokenizer.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    padding_side="right",
)
tokenizer.pad_token_id = args.pad_token_id
model = NeuronLlamaForCausalLM(args.compiled_model_path)
model.load(args.compiled_model_path)
generation_model = HuggingFaceGenerationAdapter(model)
prompts = [f"<dna>{case.sequence}" for case in cases]
inputs = tokenizer(prompts, return_tensors="pt", padding=True)
```

For each timed run, the harness resets the model, generates the batch, decodes the generated suffix, gets A/C/G/T characters, and reports tokens/sec, bp/sec, and kbp/sec.

### Baseline Results

On one `trn2.3xlarge`, the batch-16 A100-style run measured:

| Model | p50 elapsed | Token throughput | Base-pair throughput |
| --- | --- | --- | --- |
| Carbon-500M | 0.867 s | 3137.6 tok/s | 18.83 kbp/s |
| Carbon-3B | 1.484 s | 1833.0 tok/s | 11.00 kbp/s |
| Carbon-8B | 1.986 s | 1369.6 tok/s | 8.21 kbp/s |

The main read is not that this is a perfect GPU-vs-Trainium shootout. The A100 reference in the Carbon report uses vLLM dynamic batching on a single A100-80GB. This Trainium2 run uses a static compiled NxDI batch on one `trn2.3xlarge`. The useful read is operational: Carbon can run quickly on a small AWS-native Trainium2 unit immediately, with a path to larger Trn2 fleet shapes once demand justifies it.

### Run the Batch-1 DNA Prompt Suite

The public article also reports a batch-1 prompt suite. That suite is not the main cost comparison; it is a sanity check over different synthetic DNA regimes.

The prompt cases were:

- Balanced DNA
- CpG-rich
- AT-rich
- GC-rich
- ORF-like
- TATA-box-like
- CAG triplet-repeat

Compile a batch-1 artifact first:

```bash
BATCH_SIZE=1 MAX_NEW_TOKENS=180 ./scripts/compile_carbon.sh Carbon-500M
```

Then run the suite:

```bash
./scripts/run_generation_suite.sh Carbon-500M
./scripts/run_generation_suite.sh Carbon-3B
./scripts/run_generation_suite.sh Carbon-8B
```

The batch-1 median results were:

| Model | Median elapsed | Median token throughput | Median bp throughput | Avg valid DNA |
| --- | --- | --- | --- | --- |
| Carbon-500M | 0.689 s | 261.3 tok/s | 1568 bp/s | 100.0% |
| Carbon-3B | 1.277 s | 140.9 tok/s | 846 bp/s | 100.0% |
| Carbon-8B | 1.589 s | 113.3 tok/s | 679 bp/s | 99.0% |

Carbon-8B's 99% valid-DNA average came from the CAG triplet-repeat stress case, where part of the generated stream moved outside pure DNA vocabulary. That is not a Trainium issue, but it is a useful measurement caveat: base-pair throughput is cleanest when the decoded stream remains in DNA mode.

### What We Would Optimize in Near Future

Carbon-3B is the cleanest tensor-parallel fit under `tp_degree=4` because it has four KV heads. Carbon-500M and Carbon-8B compile and run under `TP=4`, but they trigger a GQA-to-MHA sharding fallback. That makes GQA layout and kernel eligibility the first place we would look.

The next serious pass should profile:

- context encoding versus token generation time
- attention kernel eligibility
- QKV and output projection paths
- RoPE and normalization paths
- bucket shapes for real serving traffic, not only benchmark symmetry
- batch shape tradeoffs for scanning, scoring, and generation workloads

The key moment is not to add custom kernels because they sound impressive. Add them only after profiling proves that a hot path is blocking the workload.

### Caveats

This benchmark is intentionally narrow:

- It measures direct 6-mer generation, not full FNS (Factorized Nucleotide Supervision) base-pair marginal decoding or scoring.
- It is a throughput and compatibility proof, not a biological quality evaluation.
- The A100 comparison is a reference comparison, not a controlled accelerator shootout.
- The measured economics use public Capacity Blocks prices and do not include orchestration, storage, utilization loss, or production serving overhead.

Those caveats matter, but they do not weaken the main engineering result: Carbon can move from a fresh Hugging Face release to a compiled Trainium2 serving artifact quickly.

### Takeaway

The important part is not only the throughput number. It is the deployment shape.

Carbon gives teams an open DNA model family. Trainium2 gives them a small AWS-native accelerator unit with 96 GB of memory. NxD Inference gives them a standard compiled serving path. Together, that turns "interesting biology model" into something much closer to an operational artifact.

For bio and HCLS teams, that is the difference between a model demo and a procurement-ready infrastructure conversation.
