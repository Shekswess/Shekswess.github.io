---
layout: post
title: "Post-Training an Open MoE Model to Extract Drug-Protein Relations: Trinity-Mini-DrugProt-Think"
description: "A practical ablation study: GRPO-style reinforcement learning with LoRA on Arcee Trinity Mini for DrugProt drug-protein relation extraction."
author: shekswess
date: 2026-02-23 00:00:00 +0800
categories: [AI, LLM]
tags: [RLVR, GRPO, LoRA, Trinity Mini, DrugProt, Open Weight Models, SageMaker]
image: https://raw.githubusercontent.com/LokaHQ/Trinity-Mini-DrugProt-Think/main/assets/logo.png
---


_**Original Source of the blog post: [Post-Training an Open MoE Model to Extract Drug-Protein Relations: Trinity-Mini-DrugProt-Think](https://lokahq.github.io/Trinity-Mini-DrugProt-Think/)**_

_**Written by Bojan Jakimovski and Petar Kalinovski**_

## Introduction 

In the past two years,
Chinese labs have come to dominate open-weight model development in a way that
few in the West anticipated.
[Alibaba’s
Qwen family has overtaken Meta’s Llama in cumulative Hugging Face downloads](https://www.technologyreview.com/2026/02/12/1132811/whats-next-for-chinese-open-source-ai/),
and by mid-2025 Qwen-derived models accounted for over 40% of new language model
uploads on the platform. [DeepSeek R1](https://arxiv.org/abs/2501.12948)
briefly narrowed the frontier gap to
[as little as four months](https://epoch.ai/data-insights/us-vs-china-eci).
[GLM-5](https://huggingface.co/zai-org/GLM-5),
[Kimi K2.5](https://github.com/MoonshotAI/Kimi-K2.5), and
[MiniMax M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5), all
released as open weights under permissive licenses in early 2026, are
[frontier-adjacent
at a fraction of the inference cost](https://www.latent.space/p/ainews-zai-glm-5-new-sota-open-weights). Despite hardware constraints that leave
Chinese labs working around export controls on cutting-edge chips, they have
executed a remarkably effective strategy: flood the world with competitive
open-weight models and let the ecosystem do the rest.

Western open-source, by contrast, is thin.
AI2 ships [OLMo3](https://arxiv.org/abs/2512.13961),
Hugging Face has [SmolLM3](https://huggingface.co/blog/smollm3),
and Mistral has [its lineup](https://mistral.ai/).
Beyond that, the list of Western labs releasing serious open-weight
foundation models is short. This is a strange asymmetry: Western labs hold a
[nearly
tenfold advantage in compute infrastructure](https://techblog.comsoc.org/2026/01/27/chinas-open-source-ai-models-to-capture-a-larger-share-of-2026-global-ai-market/) (U.S. hyperscaler capex alone
exceeded $400 billion in 2025, versus roughly $57 billion across
China’s major platforms), yet produce far fewer open-weight models. China
is not winning open-source because of a technical edge in hardware. They are
winning because almost nobody in the West is trying.

That matters because open-weight models are not just a convenience; they are
infrastructure. Researchers, startups, and institutions that cannot afford
per-token API pricing or cannot send sensitive data to a third-party endpoint
depend on models they can run locally. If the only competitive options come from
Chinese labs, that shapes who builds what, on whose stack, and on whose terms.

[Arcee AI](https://www.arcee.ai) is one of the Western labs taking
this challenge seriously, and the name of their model family gives away exactly
how: **Trinity** is a three-way collaboration.
Arcee AI led the model architecture and training.
[Prime Intellect](https://www.primeintellect.ai) contributed their
distributed training infrastructure, a platform that lets researchers train large models
across decentralised GPU clusters without needing a hyperscaler budget.
[Datalogy](https://datalogy.ai) curated the training data.
Together they released a full family: **Trinity Nano** for
edge and on-device deployment, **Trinity Mini** as the practical mid-range option, and
**Trinity Large** at the frontier scale, competitive with
[GLM-4.5 base-to-base](https://www.arcee.ai/blog/trinity-large).
All three are fully open-weight under a permissive license. We use Trinity Mini
for this study because its sparse MoE keeps inference cost low even when generating
thousands of rollouts per step, which is exactly what RLVR post-training demands.

The open-weight advantage is not just about what a model scores at release; it is about
what you can build on top of it. Targeted RLVR post-training can unlock domain-specific
capabilities that general benchmarks do not capture, and that is where open-source models
have real room to close the gap. The question we set out to answer is simple:
**Can domain-specific RLVR post-training turn Trinity Mini into a meaningfully better
model for a real workload?**

The task we chose is biomedical relation extraction. At
[Loka](https://loka.com), healthcare and life sciences projects make
up a significant part of our client work, which is part of why this domain
appealed to us. Drug discovery, clinical decision support, and pharmacovigilance
all depend on extracting structured knowledge from unstructured text. PubMed
indexes over 36 million citations, and the relationships buried in those
abstracts (which drug inhibits which protein, which compound is a substrate for
which enzyme) are the raw material for computational pharmacology.
[DrugProt](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vii/track-1/)
(BioCreative VII) is a benchmark for exactly this: 13 types of drug-protein
interactions, annotated on real PubMed abstracts.

The method is **Reinforcement Learning from Verifiable Rewards
(RLVR)**: rather than training a separate reward model, the environment
provides a ground-truth accuracy signal directly (correct label = 1, wrong = 0).
[GRPO (Group Relative Policy
Optimization)](https://arxiv.org/abs/2402.03300) makes this tractable by estimating policy gradients from groups of sampled
completions, eliminating the critic network required by
[PPO](https://arxiv.org/abs/1707.06347) and
substantially reducing memory overhead.
[DeepSeek R1](https://arxiv.org/abs/2501.12948) then demonstrated
at scale that RLVR alone, with no supervised chain-of-thought, can teach models
to reason step by step. We apply the same idea to biomedical relation
classification.

Because GRPO generates many rollouts per training step (up to 1,024 completions
in our experiments), the dominant cost is inference, not gradient computation.
Full post-training a 26B parameter MoE under these conditions is
prohibitively expensive: every parameter update must be applied after each batch
of thousands of sampled completions, requiring enormous GPU memory and wall-clock
time. We therefore use
[LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685)
, which inserts small trainable rank
decomposition matrices into the frozen base model, reducing trainable parameters
by several orders of magnitude while keeping the base weights intact.
Crucially, this is not a performance compromise:
[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
shows that LoRA fully matches full post-training in policy
gradient RLVR even at rank 1, because RLVR absorbs far less information per
update than supervised learning. We run the full pipeline on
[Prime Intellect](https://www.primeintellect.ai)’s
training infrastructure.

We ran 12 experiments sweeping LoRA alpha, learning rate, batch size, generation
budget, temperature, and rollout count. The question is deliberately practical:
**Which hyperparameters actually move the needle** when you RLVR-tune
an open-weight sparse MoE for biomedical relation extraction?

## The task & the model 

### DrugProt: the RL environment

[DrugProt](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vii/track-1/)
(BioCreative VII Track 1) is a corpus of PubMed abstracts annotated with 13 types
of drug-protein interactions: INHIBITOR, SUBSTRATE, INDIRECT-DOWNREGULATOR,
INDIRECT-UPREGULATOR, AGONIST, ANTAGONIST, ACTIVATOR, PRODUCT-OF, and five others.
Given an abstract and a specific (chemical, gene/protein) pair, the model must
classify the relationship.

We use the
[`maziyar/OpenMed_DrugProt@0.3.0`](https://app.primeintellect.ai/dashboard/environments/maziyar/openmed_drugprot)
environment, which frames DrugProt as a **13-way multiple choice question**:
each relation type maps to a letter A through M. The model sees the full PubMed abstract,
the highlighted entities, and must output its answer inside `\boxed{}` tags.

Here is a representative example from the dataset:

```text
Example input

Abstract:
Na+/Ca2+ exchange inhibitors: a new class of calcium regulators. The Na(+)/Ca(2+) exchanger (NCX) is a bidirectional transporter that normally extrudes Ca(2+) from the cell (forward mode), but also brings Ca(2+) into the cell (reverse mode) under special conditions such as intracellular Na(+) accumulation or membrane depolarization. There are three mammalian NCX isoforms: NCX1 is widely expressed in the heart, kidney, brain, blood vessels, and so on; whereas the expression of NCX2 and NCX3 is limited mainly to the brain and skeletal muscle. […]

Drug/Chemical: Ca(2+)
Gene/Protein: Na(+)/Ca(2+) exchanger

Question:
What is the relation type between Ca(2+) and Na(+)/Ca(2+) exchanger?

Options:
A. INDIRECT-DOWNREGULATOR
B. INDIRECT-UPREGULATOR
C. DIRECT-REGULATOR
D. ACTIVATOR
E. INHIBITOR
F. AGONIST
G. AGONIST-ACTIVATOR
H. AGONIST-INHIBITOR
I. ANTAGONIST
J. PRODUCT-OF
K. SUBSTRATE
L. SUBSTRATE_PRODUCT-OF
M. PART-OF

Correct answer:
K - SUBSTRATE (Ca(2+) is transported by the NCX, making it a substrate of the exchanger)
```

### Reward structure

The environment defines a composite reward with three components:

```
reward = 0.70 * accuracy + 0.20 * reasoning + 0.10 * format
```

- **Accuracy (weight: 0.70):** Binary, 1.0 if the boxed letter
  matches the gold label, 0.0 otherwise. This is the signal that matters.
- **Reasoning (weight: 0.20):** Heuristic score composed of three
  components: (1) length of the `<think></think>` block
  (0.1–0.4 depending on whether it exceeds 100 / 200 / 400 characters),
  (2) keyword density from a fixed list of 37 biomedical terms — agonist,
  inhibit, substrate, kinase, transporter, etc. — capped at 0.3, and
  (3) presence of the words “chemical” or “drug”
  (+0.1) and “protein” or “gene” (+0.1).
  Crucially, this reward is **task-agnostic**: it does not check
  whether the reasoning is correct or relevant to the specific entity pair.
  A model writing generic biomedical prose scores well regardless of whether
  its analysis of *this* relationship is right. This explains why
  reasoning reward improves similarly across all alpha values — it measures
  format and vocabulary, not depth of analysis. It also makes the reasoning
  component the noisiest of the three signals.
- **Format (weight: 0.10):** Binary, 1.0 if the output contains
  a valid `\boxed{A..M}`, 0.0 otherwise.

This structure is intentional: accuracy dominates, but the shaping signals from
reasoning and format help bootstrap learning in early steps when the model has not
yet learned the task. A practical consequence: total reward can improve early via
format compliance and longer reasoning traces before accuracy actually moves.

### The model: Arcee Trinity Mini

[Trinity Mini](https://www.arxiv.org/pdf/2602.17004) is a sparse
mixture-of-experts (MoE) model from Arcee AI: 26B total parameters,
3B active per forward pass. Sparse MoE is a natural fit for RLVR post-training
because GRPO is dominated by forward passes. With
`batch_size=128` and `rollouts_per_example=8`, each training
step generates 1,024 sampled completions. A model that is cheap at inference time
but has high total capacity is exactly what you want.

We adapt Trinity Mini using
[LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685).
LoRA freezes the base model and learns two low-rank matrices *A* and
*B* per targeted layer (rank *r*), scaled by *α*.

Δ*W* = (*α*/*r*) · *B**A*

The key hyperparameter we ablate is **α**: it controls how
much the LoRA updates influence the model’s outputs independently of the
number of trainable parameters. Higher α means stronger adaptation per
step; lower α is more conservative. Note that α acts as a multiplier
on the effective learning rate for the adapter layers; raising α and
raising the global LR have similar effects on training dynamics.

**Important Distinction:** We sweep LoRA *alpha* (the
scaling factor), not LoRA *rank*. Alpha controls update magnitude without
changing the number of trainable parameters. This makes it a pure “how
aggressively do we adapt?” knob.

## Experiments & results 

Our ablation strategy is sequential. We sweep one variable at a time, lock in the
best setting, and move to the next. This gives us a clear cause-and-effect story
at the cost of not exploring the full combinatorial space. For each comparison we
show learning curves (accuracy reward over training steps) with a 5-step moving
average for readability, plus snapshot bar charts at key steps.

One parameter we did not sweep is LoRA rank. Prime Intellect’s training
infrastructure fixes rank at `r=16`, applied across all projection
layers (`q_proj`, `k_proj`, `v_proj`,
`o_proj`, `gate_proj`, `up_proj`,
`down_proj`, and expert layers). We are comfortable leaving rank
fixed because
[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
establishes that policy gradient algorithms absorb roughly
one bit of information per episode independent of model size, meaning a
rank-1 LoRA (3M parameters for an 8B model) vastly exceeds the
capacity needed for typical RL training sets. Rank is therefore not the binding
constraint in our setting. The same work also demonstrates that applying LoRA to
all layers (attention *and* MLPs) substantially outperforms
attention-only LoRA — a finding consistent with our all-layer setup.
On Trinity Mini this is especially relevant: the model’s G1 gated
attention uses a dedicated `gate_proj` weight matrix in every
attention layer that is both a LoRA target and architecturally load-bearing
(it controls attention output magnitudes via a learned sigmoid gate), so
leaving it frozen would mean skipping a significant set of parameters.

All 12 experiments (click to expand)

Unless noted otherwise, defaults are:
`batch_size=128`, `rollouts_per_example=8`,
`temperature=0.7`, `max_tokens=512`.

| Run | α | LR | Batch | Rollouts | Max tokens | Temp |
| --- | --- | --- | --- | --- | --- | --- |
| `w1-alpha16-baseline` | 16 | 1e-6 | 128 | 8 | 512 | 0.7 |
| `w1-alpha32-baseline` | 32 | 1e-6 | 128 | 8 | 512 | 0.7 |
| `w1-alpha64-baseline` | 64 | 1e-6 | 128 | 8 | 512 | 0.7 |
| `w1-alpha128-baseline` | 128 | 1e-6 | 128 | 8 | 512 | 0.7 |
| `w1-alpha64-lr1e-5` | 64 | 1e-5 | 128 | 8 | 512 | 0.7 |
| `w1-alpha64-lr5e-6` | 64 | 5e-6 | 128 | 8 | 512 | 0.7 |
| `w1-alpha64-lr3e-6` | 64 | 3e-6 | 128 | 8 | 512 | 0.7 |
| `w1-alpha64-lr3e-6-b64-r16` | 64 | 3e-6 | 64 | 16 | 512 | 0.7 |
| `w1-alpha64-lr3e-6-1024` | 64 | 3e-6 | 128 | 8 | 1024 | 0.7 |
| `w1-alpha64-lr3e-6-2048` | 64 | 3e-6 | 128 | 8 | 2048 | 0.7 |
| `w2-2048-t0.15` | 64 | 3e-6 | 128 | 8 | 2048 | 0.15 |
| `w2-2048-r4` | 64 | 3e-6 | 128 | 4 | 2048 | 0.7 |

### 1) LoRA alpha ablation (16 / 32 / 64 / 128)

We started with the most fundamental LoRA knob: the alpha scaling factor.
Keeping everything else fixed (LR 1e-6, batch 128, rollouts 8,
max\_tokens 512), we trained four runs at α=16, α=32, α=64,
and α=128. The general picture is clear: up to α=64, higher alpha
produces both faster learning and higher accuracy. α=64 is the best
overall setting, reaching ~0.75 accuracy reward at step 100 versus ~0.70
for α=32 and ~0.41 for α=16.

![Figure 1 - LoRA alpha sweep](/assets/images/drug-think/figure1.png)

Figure 1. LoRA alpha sweep (steps 1–100). Accuracy reward vs training
step for α=16, 32, 64, and 128, all at LR 1e-6. α=128
is terminated at step 70 due to eval degradation.

The exception is α=128. On the training metric it looks competitive
, even outperforming α=64 at matching steps, but the
held-out eval tells the opposite story. Figure 2 makes this impossible
to miss: α=128 started with the highest eval score of any run and ended
with the lowest, dropping steadily from step 0. The model was terminated
at step 70. Training reward is healthy because aggressive updates adapt
quickly to each batch; the eval exposes that generalization is collapsing at
the same rate.

![Figure 2 - Held-out eval score by LoRA alpha](/assets/images/drug-think/figure2.png)

Figure 2. Held-out eval score (`avg@2`) over training steps for
each LoRA alpha. α=64 climbs steadily; α=128 declines from the
very first checkpoint and is terminated at step 70. α=16 and
α=32 stay roughly flat.

![Figure 3 - Reasoning reward by LoRA alpha](/assets/images/drug-think/figure3.png)

Figure 3. Reasoning reward by LoRA alpha (steps 1–100). Unlike
accuracy, reasoning reward improves similarly across all alphas. The
model learns to produce structured thinking traces regardless of update
magnitude.

Reasoning reward tells a different story from accuracy. While α=64
clearly outperforms α=16 on accuracy, all four alphas produce similarly
improving reasoning traces over time. The model learns to write structured
pharmacological analysis regardless of how aggressively the LoRA layers are
being updated. This suggests that reasoning quality, as measured by our
heuristic, is more about training duration than update magnitude, a
contrast worth keeping in mind when interpreting the max tokens results below.

Alpha is not a dial where higher is always better. The LoRA scaling factor is
`α / rank`, so a higher alpha directly amplifies the
magnitude of every adapter update — effectively inflating the learning rate for
the LoRA layers independently of the global LR. The consequences are threefold.
First, the adapted weights receive disproportionately large updates, producing
loss spikes and erratic training dynamics. Second,
the model overfits aggressively to each batch: training accuracy climbs fast,
but the pretrained feature space gets warped at the same rate, eroding the
general capabilities inherited from pretraining. Third, and most visible in
Figure 3, generalization collapses on held-out data while the training
signal stays deceptively healthy.

The key practical insight is that alpha and learning rate are not independent
knobs: raising alpha has the same effect as raising the LR for the adapter
layers. If instability appears, lowering either will help. α=64 sat in
the stable, productive middle of our sweep, and we locked it in for all
subsequent experiments.

### 2) Learning rate sweep at α=64 (1e-5 / 1e-6 / 3e-6 / 5e-6)

With alpha fixed at 64, we swept the learning rate across four values:
1e-6 (our alpha baseline), 3e-6, 5e-6, and the aggressive 1e-5. This
is where the story gets interesting.

The three moderate learning rates (1e-6, 3e-6, 5e-6) all produced healthy
learning curves with different speed/stability tradeoffs. 5e-6 learned fastest
early on. 3e-6 was steady and reliable. 1e-6 was conservative but safe.

**1e-5, however, was catastrophic.** The run started promisingly; it
actually climbed faster than the others in the first 20 steps. But by step 50
the reward began oscillating wildly, and by step 77 accuracy collapsed to
zero. The run was terminated at step 82 with no recovery in sight. This is
not a gradual degradation; it is a cliff. The model essentially forgot how to
produce valid outputs.

![Figure 4 - Learning rate sweep at alpha 64](/assets/images/drug-think/figure4.png)

Figure 4. Learning rate sweep at α=64 (steps 1–100). The 1e-5 run
(red) climbs early but collapses catastrophically around step 50–77. The
remaining three learning rates show the classic speed-vs-stability tradeoff.

![Figure 5 - Accuracy reward at step 82 by learning rate](/assets/images/drug-think/figure5.png)

Figure 5. Accuracy reward at step 82 vs learning rate (log scale). The cliff
between 5e-6 and 1e-5 is dramatic: a 2× increase in LR produces
a collapse from ~0.83 to near zero.

**Why 1e-5 collapses and what LoRA Without Regret says about it:**
[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
finds that the optimal LoRA learning rate is roughly
**10× the optimal full post-training LR** across a wide
range of models and tasks (15× for short runs of ~100 steps). On first
reading, this suggests we should be running at 1e-5 or higher. So why did
1e-5 destroy our model?

The key is the interaction between LR and α. LoRA’s weight update
is ΔW = (α/r)·BA. At α=64, r=16 the
scaling factor is 4; at the α=32 used in LoRA Without Regret it is 2.
Our effective per-step update magnitude at LR 1e-5 is therefore
equivalent to running at LR 2e-5 under their parameterization,
already at the edge of their reported stability range. Additionally, LoRA
Without Regret’s experiments use dense Llama-class architectures; sparse
MoE models route tokens through only a subset of experts per forward pass,
which can amplify gradient variance and narrow the stable LR window further.
The collapse at 1e-5 is therefore consistent with the theory once the
α-scaling and MoE architecture are both accounted for: we were above
the stability boundary for *this specific combination* of adapter
scaling and model class.

Two Trinity Mini-specific factors are worth flagging. First, Trinity Mini
was pre-trained with [μP](https://arxiv.org/abs/2203.03466) input
scaling (embedding outputs multiplied by √2048 ≈ 45 before
each transformer layer), a different pre-training regime from the standard
architectures used in LoRA
Without Regret’s experiments; their empirical LR bounds were not
established under equivalent conditions. Second, MoE models face a
documented instability problem under GRPO specifically: top-k discrete
routing causes the set of activated experts to drift between inference and
training, [inflating importance-sampling
variance](https://arxiv.org/abs/2510.23027) in a way that dense models do not experience.
These two factors make empirical LR validation more important here than on
a standard dense model.

The practical takeaway: when porting the LoRA Without Regret 10× rule
to a new setup, adjust for your actual α/r ratio and verify stability
empirically at the high end before committing to a full run.

At step 82 (the last step before 1e-5 data disappears): 5e-6 leads with
0.83 accuracy, followed by 3e-6 at 0.72, 1e-6 at 0.70, and 1e-5 at 0.00.
However, 5e-6 showed signs of instability in later steps. We chose
**3e-6** as the safest default that still converges quickly.

### 3) Batch size: 128×8 vs 64×16

Next we tested whether the same total completions per step (1,024) should come
from more examples with fewer rollouts or fewer examples with more rollouts.
We compared our default (`batch=128, rollouts=8`) against
(`batch=64, rollouts=16`), both at α=64, LR 3e-6.

![Figure 6 - Batch size comparison](/assets/images/drug-think/figure6.png)

Figure 6. Batch size comparison (steps 1–100). Accuracy reward for
128×8 (more examples, fewer rollouts) vs 64×16 (fewer examples,
more rollouts per example).

The two configurations performed comparably. At step 100, accuracy was
~0.74 for 128×8 and ~0.75 for 64×16. The key difference was cost:
64×16 generated notably longer completions on average (~222 vs ~167
tokens), meaning more inference compute per step for similar accuracy. We kept
**128×8** as the default.

### 4) Max generation tokens (512 / 1024 / 2048)

With α=64, LR 3e-6, and batch 128×8 locked in, we swept the
maximum generation length. This knob changes both compute cost and the model’s
behavior, as longer caps allow more elaborate reasoning traces.

![Figure 7 - Max tokens sweep](/assets/images/drug-think/figure7.png)

Figure 7. Max tokens sweep (steps 1–100). Accuracy reward for 512, 1024,
and 2048 max tokens, all at α=64, LR 3e-6, batch 128×8.

![Figure 8 - Composite reward at step 100 by max tokens](/assets/images/drug-think/figure8.png)

Figure 8. Mean composite reward at step 100 by max tokens. 2048 achieves the
highest overall reward, driven by both accuracy and richer reasoning traces.

![Figure 9 - Reasoning reward by max tokens](/assets/images/drug-think/figure9.png)

Figure 9. Reasoning reward by max tokens (steps 1–100). The gap is
structural and visible from step 1: 2048 tokens starts at ~0.87,
1024 at ~0.81, and 512 at ~0.62, before any learning has occurred.
The reasoning heuristic rewards think blocks of 400+ characters, which a
512-token budget cannot reliably accommodate alongside the answer.

The 2048 configuration stood out as the strongest overall performer, and
Figure 9 makes the mechanism explicit. This is not a learned behaviour; it is a structural constraint. With 512 max tokens, the model must
fit both its reasoning trace and its boxed answer into a tight budget; reliably
hitting the 400+ character threshold for the reasoning reward becomes
structurally difficult. With 2048 tokens, the model has room to reason from
step 1, and that head-start compounds throughout training. Since reasoning
carries a 0.20 weight in the composite reward, the advantage is immediate and
persistent.

The cost, however, is real: at step 100, mean completion length is ~167
tokens at 512, ~334 at 1024, and ~605 at 2048. That is a 3.6× increase in
inference tokens per completion. We treat **2048** as the quality
setting for the final ablations.

### 5) Final ablations: temperature and rollout budget

Our final two experiments started from the best configuration found so far
(α=64, LR 3e-6, batch 128, max\_tokens 2048) and probed
two remaining knobs:

- **Temperature 0.15:** Does lower sampling entropy improve
  accuracy by concentrating probability on the model’s best guesses?
- **Rollout 4:** Can we halve the rollout budget (and thus
  inference cost) per step without losing accuracy?

![Figure 10 - Final ablations](/assets/images/drug-think/figure10.png)

Figure 10. Final ablations (steps 1–100). Accuracy reward for the 2048
baseline, the low-temperature variant (t=0.15), and the reduced-rollout variant
(r=4).

The rollout=4 variant matched or exceeded the baseline’s accuracy
throughout training, suggesting that for this task, 4 rollouts per example
provide sufficient signal for GRPO updates. This is significant because it
halves inference cost per training step.

The low-temperature variant (t=0.15) told a more nuanced story. It started
with very high accuracy in early steps (the model’s existing knowledge,
sampled deterministically, was already decent). But over the course of training,
the reduced exploration appeared to limit improvement. Lower temperature means
less diversity in rollouts, which means less informative gradient signal for GRPO.

## Key findings & best configuration 

After 12 experiments, here is what we learned:

**1. LoRA alpha is the single most important knob.**
The difference between α=16 and α=64 was larger than any other
hyperparameter change we tested. This is not about adding more parameters (rank
was fixed); it is purely about update magnitude. If you are running LoRA
with RLVR, tune alpha before anything else.

**2. Learning rate 1e-5 is a hard boundary for LoRA + RLVR.**
The collapse is not gradual. The model goes from learning to producing garbage
within ~30 steps. This is a direct consequence of LoRA’s alpha-scaled
updates interacting with GRPO’s policy gradient: when the effective step
size is too large, the policy moves so far from the reference that the KL penalty
cannot recover. Anyone running RLVR with LoRA adapters should treat 1e-5 as a
red line, not a starting point.

**3. More thinking room helps, but you pay for it.**
Going from 512 to 2048 max tokens improved composite reward meaningfully. The
model produced richer, more pharmacologically grounded reasoning. But completion
length tripled, and the accuracy gains themselves were noisy. The right framing is:
max\_tokens is a cost-quality dial, not a free lunch.

**4. You can cut rollouts without losing accuracy.**
The rollout=4 experiment was our most practically useful finding. Halving rollouts
per example halves inference cost per training step, and on this task the model
maintained strong accuracy. This suggests that for well-structured classification
tasks with clear reward signals, you do not need 8 rollouts to get good GRPO
gradients.

**5. Low temperature hurts RLVR exploration.**
Temperature 0.15 gave a strong start (the model’s existing knowledge was
sampled more deterministically) but underperformed over the full 100 steps. RLVR
needs diversity in rollouts to compute meaningful advantages. Squeezing exploration
too early is counterproductive.

**Best configuration:**
α=64, LR 3e-6, batch 128, rollouts 8 (or 4 for cost
savings), max\_tokens 2048, temperature 0.7.
This combination produced the highest overall reward and maintained stable
learning throughout training.

### The output contrast: best model vs catastrophic 1e-5

To make the impact concrete, here is a side-by-side comparison of model outputs
on the same DrugProt example. The best configuration produces structured
pharmacological reasoning; the collapsed 1e-5 model produces incoherent output.

Best model (α=64, lr=3e-6, 2048 tokens)

```text
<think>
The user is asking about the relation between “Ser” (serine, an amino acid) and
“GXSXG motif” (a conserved sequence motif in proteins). The options are
various types of interactions.

Key points from the biomedical text: Calnuc is a serine protease, and the text
mentions “Ser-378 of GXSXG motif”. This indicates that serine (Ser) is part
of the GXSXG motif, specifically at position 378. The GXSXG motif is a catalytic
motif in serine proteases, where serine acts as the catalytic residue. The text
states that calnuc has two Zn²+-binding sites and is inactive as a peptidase.
Zn²+ binding negatively regulates serine protease activity by causing a
conformational change. G protein α subunit also inhibits enzymatic activity.

Analyzing the options: A–I are about regulation (inhibition, activation, etc.),
but the relationship here is that Ser is part of the motif, not regulating it.
J–M are about the chemical’s role in relation to an enzyme’s process.
M is PART-OF, which fits because Ser is a component of the GXSXG motif. Other
options don’t fit: J–L are about the chemical being a product, substrate,
or both; A–I are about regulation. Ser here is a residue in a protein motif,
not a product, substrate, or external regulator.

Final answer: M.
</think>

The text explicitly states that Ser-378 is part of the GXSXG motif in calnuc.
Serine is a component residue within the motif, making it a structural constituent
rather than an external regulator or product/substrate.

\boxed{M}
```

Collapsed model (lr=1e-5, step ~80)

```text
<think>
the the the the the the the the the the the the the the the
the the the the the the the the the the the the the the the
the the the the the the the the the the the the the the the
</think>

\boxed{A}
```

## Next steps 

- **Longer training runs.** Our best configuration (α=64,
  LR 3e-6, 2048 tokens) was capped at ~100 steps and its eval curve had
  not flattened. Running it for 500+ steps is the clearest path to a stronger
  model, though diminishing returns and eval divergence both remain risks.
- **Evaluation on the official DrugProt test set.** Our held-out
  metric (`avg@2`) is computed on the training split during RLVR. A
  proper evaluation on the official test set would tell us whether the gains
  reflect genuine generalization or a more narrow adaptation to the training
  distribution.
- **Trying different base models.** Trinity Mini is one choice.
  Running the same ablation protocol on other open-weight models would reveal
  whether the alpha and LR sensitivities we found are MoE-specific or general
  properties of LoRA under RLVR.
- **Better reward signals.** The reasoning reward is heuristic
  (keyword matching and length thresholds). Replacing it with a learned reward
  model or NLI-based evaluation could change the learning dynamics substantially,
  especially for the reasoning component which was the noisiest signal.

## Conclusion 

The recipe works. α=64, LR 3e-6, 2048 max tokens: a 3B-active-parameter
open-weight MoE that learns to reason about drug-protein interactions from reward
signals alone, no labeled chain-of-thought required. The leverage is almost
entirely in two numbers; everything else is secondary.

If you want to use the fine-tuned model, you do not need to re-run the training.
We have published the adapter weights ([**lokahq/Trinity-Mini-DrugProt-Think**](https://huggingface.co/lokahq/Trinity-Mini-DrugProt-Think)) and written a step-by-step deployment guide
using the **AWS SageMaker SDK v3**, covering how to serve the merged
model as a real-time endpoint:
[deployment guide](https://medium.com/loka-engineering/deploying-trinity-mini-drugprot-think-on-amazon-sagemaker-ai-9e1c1c430ce9).
SageMaker’s managed inference handles scaling and hardware allocation;
the guide walks through container selection, endpoint configuration, and a
sample inference call against the DrugProt relation types.

We have also started running the same RLVR+LoRA recipe on
**Trinity Nano** (6B parameters), across both medical and
general reasoning tasks. Nano is small enough to serve cheaply on a single GPU
instance, which matters for inference in production use cases. We expect the same
hyperparameter sensitivities to hold; we will publish a follow-up when we have
results worth sharing.

## Acknowledgements 

- **Model:** [Arcee AI](https://www.arcee.ai) (with [Prime Intellect](https://www.primeintellect.ai) and [Datalogy](https://datalogy.ai)) for releasing the Trinity family.
- **Training:** [Prime Intellect](https://www.primeintellect.ai) for hosted training infrastructure.
- **Environment:** [OpenMed](https://huggingface.co/datasets/OpenMed/drugprot-parquet) for DrugProt dataset packaging.
- **Deployment:** [AWS](https://aws.amazon.com/) for deployment and hosting.

## Citation 

If you find this work useful, please cite:

```
@misc{jakimovski2026drugprotrl,
  title        = {Post-Training an Open MoE Model to Extract Drug-Protein Relations: Trinity-Mini-DrugProt-Think},
  author       = {Jakimovski, Bojan and Kalinovski, Petar},
  year         = {2026},
  month        = feb,
  howpublished = {Blog post},
  url          = {https://github.com/LokaHQ/Trinity-Mini-DrugProt-Think}
}
```

## Further reading 

1. MIT Technology Review (2026).
   *What’s next for Chinese open-source AI.*
   Reports Qwen overtaking Llama in Hugging Face downloads and
   Qwen-derived models reaching 40%+ of new language model uploads.
   [technologyreview.com](https://www.technologyreview.com/2026/02/12/1132811/whats-next-for-chinese-open-source-ai/)
2. Epoch AI (2026).
   *Chinese AI models have lagged the US frontier by 7 months on average since 2023.*
   Analysis of the Epoch Capabilities Index showing the US-China frontier gap
   ranged from 4 to 14 months.
   [epoch.ai](https://epoch.ai/data-insights/us-vs-china-eci)
3. IEEE ComSoc Technology Blog (2026).
   *China’s open source AI models to capture a larger share of 2026 global AI market.*
   Cites Goldman Sachs and UBS data on the ~10× gap in compute
   infrastructure investment between U.S. hyperscalers ($400B+) and
   Chinese platforms (~$57B).
   [techblog.comsoc.org](https://techblog.comsoc.org/2026/01/27/chinas-open-source-ai-models-to-capture-a-larger-share-of-2026-global-ai-market/)
4. Latent Space (2026).
   *[AINews] Z.ai GLM-5: New SOTA Open Weights LLM.*
   Covers the February 2026 releases of GLM-5 (744B) and MiniMax M2.5 (230B).
   [latent.space](https://www.latent.space/p/ainews-zai-glm-5-new-sota-open-weights)
5. Z.ai (2026).
   *GLM-5.*
   Official open-weights model card.
   [huggingface.co/zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5)
6. MoonshotAI (2026).
   *Kimi K2.5.*
   Official model repository and technical report.
   [github.com/MoonshotAI/Kimi-K2.5](https://github.com/MoonshotAI/Kimi-K2.5)
7. MiniMax (2026).
   *MiniMax-M2.5.*
   Official open-weights model card.
   [huggingface.co/MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
8. MiniMax (2026).
   *MiniMax M2.5: The official release announcement.*
   [minimaxi.com](https://www.minimaxi.com/news/minimax-m25)
9. Schulman, J. et al. (2017).
   *Proximal Policy Optimization Algorithms.*
   Introduces PPO.
   [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
10. Schulman, J. (2025).
    *LoRA Without Regret.*
    Thinking Machines Lab: Connectionism.
    DOI: 10.64434/tml.20250929.
    Establishes that LoRA fully matches full post-training in policy gradient RL
    even at rank 1. Derives an empirical LR scaling rule (optimal LoRA LR
    ≈ 10× full post-training LR), shows attention-only LoRA substantially
    underperforms all-layer LoRA, and characterizes the α/r parameterization
    invariance.
    [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/)
11. Hu, E. J. et al. (2022).
    *LoRA: Low-Rank Adaptation of Large Language Models.*
    International Conference on Learning Representations (ICLR).
    Introduces low-rank decomposition adapters that reduce trainable parameters
    by up to 10,000× compared to full post-training, with no inference
    latency overhead.
    [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
12. Shao, Z. et al. (2024).
    *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open
    Language Models.*
    Introduces GRPO (Group Relative Policy Optimization), which estimates policy
    gradient baselines from groups of sampled completions, eliminating the critic
    network required by [PPO](https://arxiv.org/abs/1707.06347) and
    substantially reducing memory cost.
    [arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
13. DeepSeek-AI (2025).
    *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
    Reinforcement Learning.*
    Demonstrates that RLVR alone — with no supervised chain-of-thought
    data — can teach models to reason step by step, achieving
    AIME 2024 accuracy of 71.0% (86.7% with majority voting), matching
    OpenAI o1.
    [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
14. Arcee AI (2025).
    *The Trinity Manifesto.*
    Blog post introducing Trinity Mini and Trinity Nano, including benchmark
    comparisons against Qwen3-30B-A3B Thinking, OLMo3-32B, and Magistral Small.
    [arcee.ai/blog/the-trinity-manifesto](https://www.arcee.ai/blog/the-trinity-manifesto)
15. Arcee AI (2025).
    *Trinity Large.*
    Blog post presenting Trinity Large benchmark comparisons against
    GLM-4.5, GLM-4.7, DeepSeek-V3, Llama 4 Maverick, and MiniMax M2.1.
    [arcee.ai/blog/trinity-large](https://www.arcee.ai/blog/trinity-large)
16. Team OLMo (2025).
    *OLMo3.*
    Allen Institute for AI.
    A fully open foundation model family (weights, training data, code) released
    by AI2 as a contribution to transparent Western open-source AI research.
    [arxiv.org/abs/2512.13961](https://arxiv.org/abs/2512.13961)
17. Mistral AI (2025).
    *Magistral.*
    Mistral AI.
    Mistral's first reasoning model family, trained with RL on top of Mistral
    Medium 3. Magistral Small (24B, Apache 2.0) is open-weight.
    [arxiv.org/abs/2506.10910](https://arxiv.org/abs/2506.10910)
18. Arcee AI (2026).
    *Arcee Trinity Large Technical Report.*
    Describes the Trinity family of sparse mixture-of-experts models, including
    Trinity Mini (26B total / 3B active) used in this study.
    [arxiv.org/pdf/2602.17004](https://www.arxiv.org/pdf/2602.17004)
19. maziyar/OpenMed\_DrugProt (2025).
    *DrugProt RL environment for Prime Intellect.*
    The verifiers environment used for all experiments in this study.
    [app.primeintellect.ai](https://app.primeintellect.ai/dashboard/environments/maziyar/openmed_drugprot)
20. Miranda, A. et al. (2021).
    *Overview of DrugProt BioCreative VII Track 1: Advanced Biomedical Named
    Entity Recognition and Relation Extraction.*
    BioCreative VII Workshop. Introduces the DrugProt corpus of PubMed abstracts
    annotated with 13 drug-protein relation types.
21. OpenMed (2025).
    *drugprot-parquet dataset.*
    Parquet version of DrugProt used by the environment.
    [huggingface.co/datasets/OpenMed/drugprot-parquet](https://huggingface.co/datasets/OpenMed/drugprot-parquet)
22. Qwen Team (2025).
    *Qwen3 Technical Report.*
    Alibaba Cloud. Dense and MoE models from 0.6B to 235B parameters, with
    unified thinking/non-thinking modes. Qwen3-30B-A3B is the MoE variant
    used for comparison in this study.
    [arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)
23. Zhang, D. et al. (2025).
    *Towards Stable and Effective Reinforcement Learning for Mixture-of-Experts.*
    Identifies top-k routing drift between inference and training as the root
    cause of MoE instability under GRPO, and proposes RSPO to address it via
    router-aware importance sampling.
    [arxiv.org/abs/2510.23027](https://arxiv.org/abs/2510.23027)
24. Yang, G. et al. (2022).
    *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot
    Hyperparameter Transfer.*
    NeurIPS 2022. Introduces Maximal Update Parametrization (μP), enabling
    hyperparameter transfer across model widths during pre-training.
    [arxiv.org/abs/2203.03466](https://arxiv.org/abs/2203.03466)
