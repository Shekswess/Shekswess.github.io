---
layout: post
title: "Teaching an Open Model to Do Science"
description: "Post-training an open model for tool use, biological reasoning, and auditable research workflows"
author: shekswess
date: 2026-07-30 00:00:00 +0200
permalink: /teaching-an-open-model-to-do-science.html
categories: [AI, Research]
tags: [AI Scientist, Trinity Mini, RLVR, GRPO, LoRA, Prime RL, Biomedical AI]
image: https://cdn.sanity.io/images/qmhg7q8j/prod/13ad8c6cd6741aeec2bfc338a4d488e7ec686995-2400x1200.png?w=2048&q=80&auto=format
source_title: "Teaching an Open Model to Do Science"
source_url: "https://www.arcee.ai/blog/teaching-an-open-model-to-do-science"
source_authors: ["Bojan Jakimovski", "Sara Kovachovska", "Maziyar Panahi"]
---

_**Original Source of the blog post: [Teaching an Open Model to Do Science](https://www.arcee.ai/blog/teaching-an-open-model-to-do-science)**_

_**Blog was written with my colleagues from Loka & Arcee: Sara Kovachovska & Maziyar Panahi**_

Our earlier Trinity Mini work taught a compact model to classify drug–protein relations from biomedical abstracts. This project asked more of the same open model: investigate a scientific question with tools, reason across incomplete evidence, and return work another researcher can check.

Loka, AWS, Arcee, and Prime Intellect built two reinforcement-learning environments around that goal. One teaches the model to seek evidence across biomedical tools. The other teaches it to infer Gene Ontology annotations from protein evidence and return strict JSON. We then ran 21 controlled post-training experiments and promoted the configuration that performed well on both tasks.

### Results at a glance

Run 120 was the selected checkpoint.

- On the held-out Drug Tool evaluation, its score rose from 70.8% to 81.2%, increasing at every evaluation point.
- On held-out BioReason, it reached 86.3% on the reported combined score, which includes GO term F1, GO tree similarity, aspect coverage, and JSON validity.
- Before reinforcement learning, one GEPA prompt-search pass improved base-model validation by about 84% on BioReason and 7.7% on Drug Tool. These figures measure prompt-search validation before post-training and are separate from the final results.

Selection also depended on tool traces, verifier outputs, citations, identifiers, structured responses, and behavior inside a working scientific application.

Scientific work rarely begins as a clean classification problem. A researcher may start with a disease and a target rather than an answer set. The first query may fail. A protein may have several identifiers. One paper may support a hypothesis while another weakens it. A useful system has to decide what evidence to seek next, distinguish missing evidence from negative evidence, and leave a record of how it reached its conclusion.

### From prompting to learned scientific behavior

A prompt can describe a procedure, but it does not make that procedure reliable across a multi-step run. Instructions to prefer primary evidence, resolve identifiers, use tools, and return valid JSON compete with partial results and formatting constraints. Models also tend to produce a helpful-looking answer when the evidence is weak.

That failure mode is especially costly in science. Invalid JSON is easy to catch. A polished synthesis about the wrong protein can pass quietly into the next stage of a workflow.

We therefore focused the research on two questions:

1. What feedback would reward the scientific behavior we wanted?
2. What held-out evaluations would show whether that behavior improved?

Reward design followed from those questions. Drug Tool evaluated tool choice, arguments, retrieved facts, completion, efficiency, and concision alongside the final prose. BioReason evaluated biological content, output structure, and JSON validity.

There is also an operational reason to specialize an open model. A compact model and a small LoRA adapter can run inside an organization’s cloud boundary, remain pinned to a known policy version, and be updated against that organization’s evidence and standards. The team controls the data path, reward, evaluation, serving cost, and incident response. This operational case complements the scientific case for better feedback.

### Two environments, two kinds of scientific work

The training mixture used two published Prime RL environments in equal proportion. Together they cover investigation and conclusion.

### Drug Tool RL: learning to investigate

The Loka Drug Tool SFT dataset came from the prompt bank used by Loka’s drug-discovery application. Arcee Trinity Large Thinking generated assistant and tool trajectories through OpenRouter’s OpenAI-compatible interface. Each example preserves the biomedical question, tool schemas, assistant actions, and serialized tool results.

The dataset contains 800 training prompts and 200 held-out prompts across 17 workflow categories, with 5,049 structured tool calls in total. The categories include disease–pathway evidence, literature triage, contradiction checking, GEO biomarker screening, KEGG resolution, protein records, interaction networks, ortholog comparison, molecular generation, protein folding, and docking planning.

The `lokahq/drug-tool-rl@3` environment presents a biomedical question, a tool catalog, and a completion contract. The model must choose and parameterize tools, recover when retrieval fails, and finish with a concise synthesis. Seven retrieval tools cover PubMed, GEO, KEGG, UniProt, and STRING. When enabled, NVIDIA NIM-backed tools add protein folding, ligand docking, and molecular generation. Identical calls are cached within a rollout, and structural artifacts can move between stateful tools.

Its reward combines grounded facts, successful tool use, appropriate tool choice and arguments, completion, efficiency, concision, and final-answer quality. Diagnostics record hallucinated identifiers, tool errors, rate limits, duplicate calls, and evidence overlap. A fluent answer after failed retrieval should score differently from a synthesis grounded in retrieved evidence. Calling many tools without reaching a useful conclusion should also lose credit.

### BioReason: learning to conclude

The BioReason RL corpus contains 8,630 curated records across training and held-out splits. It builds on BioReason-Pro and combines protein metadata and sequence with InterPro, interaction, and subcellular-location evidence. Each record also includes GO-GPT candidate terms. The model must evaluate these noisy hypotheses rather than copy them as labels.

In `lokahq/bioreason-go-rl@1`, the model must infer a functional summary and lists of Gene Ontology identifiers for Molecular Function, Biological Process, and Cellular Component. It must return exactly one JSON object.

The verifier scores GO F1 by aspect, mean GO F1, tree similarity, aspect presence, and strict JSON validity. The final objective emphasized accuracy for aspects present in the evidence while retaining a format check.

The evaluation protocol ran 200 held-out Drug Tool examples and 512 held-out BioReason examples every 20 training steps. These results measure behavior under specified environments and verifiers. They are not claims about clinical validity, wet-lab performance, or success across an entire drug program.

### Improve the task before training the policy

The system prompt is part of an RL environment. It changes the actions a policy considers and the errors a verifier can observe. Before GRPO post-training, we ran one GEPA prompt-optimization pass on each environment.

Trinity Mini produced the base-model rollouts. A Claude Sonnet 5 reflection model used those rollouts and verifier feedback to propose revised task instructions. The selected prompts improved base-model validation by about 84% on BioReason and 7.7% on Drug Tool.

![GEPA prompt search before RL](/assets/images/teaching-an-open-model-to-do-science/image.png)

*Figure 1. GEPA prompt search before RL. Starting from the same base-model validation index of 100, the selected prompts raised BioReason to 184 and Drug Tool to 107.7. The dashed line marks the shared starting point; each bar is the lift the reflection loop found before GRPO ever ran.*

We then retrained with the selected prompts and repeated the full held-out evaluation. The environments responded differently: one retained a post-training benefit, while the other largely absorbed the prompt gain during RL. That result shaped how we used GEPA.

We ran prompt search once, before RL, to improve the task definition. We did not keep rewriting prompts around an already trained checkpoint. The rest of the research budget went to data, reward, and optimization changes.

![Dataset map for Drug Tool and BioReason](/assets/images/teaching-an-open-model-to-do-science/image_2.png)

*Figure 2. Dataset map. Drug Tool and BioReason turn distinct scientific records into verifiable learning signals. One path teaches evidence-seeking behavior; the other teaches precise biological annotation.*

### Model and training setup

Trinity Mini is a 26-billion-parameter mixture-of-experts model with 3 billion active parameters and a 128k native context window. It has 128 experts; eight routed experts and one shared expert are active for each token. Every experiment began from the same original `arcee-ai/Trinity-Mini` checkpoint.

Training used GRPO with LoRA adapters. Prime Intellect’s `prime-rl` separated policy optimization from rollout inference and connected both to the two verifiable environments. Training and rollout workers ran separately, while the environments returned rewards and diagnostics.

The active training context was 16k tokens. Rollouts allowed up to 7,168 completion tokens. Drug Tool training allowed 12 turns, while held-out Drug Tool evaluation allowed eight. Across the series, we kept the original model checkpoint, Muon optimizer, held-out protocol, and four-plus-four GPU topology fixed.

Those controls made each run easier to interpret. A configuration change could be compared with a known baseline instead of a moving system.

![AWS training infrastructure showing the prime-rl training and rollout path](/assets/images/teaching-an-open-model-to-do-science/report/training-loop-aws-infrastructure.png)

*Figure 3. One AWS path from training to deployment. prime-rl builds GRPO batches across separate trainer and rollout-generation workers on Amazon EC2 P5. Loka's environments compute rewards, Amazon S3 preserves checkpoints and adapters, and the promoted LoRA adapter moves into the serving and agentic harness path. Click the figure to zoom into the full-resolution view.*

### Twenty-one runs, one change at a time

A promotion bar, a bounded search space, and persistent experiment records guided the 21 runs. The final recipe emerged from that process.

`GOAL.md` defined the threshold across both environments. `CONTEXT.md` stored the fixed protocol, known failure modes, and important measurements. `PLAN.md` enforced a one-hypothesis-per-run rule. Each launch had a versioned configuration under `rl/`. `runs.jsonl` recorded settings, metrics, hypotheses, and decisions, while reports under `analysis/` captured curves, component metrics, and representative rollouts. `THREAD.md` carried the current state and next hypothesis between runs.

The loop was simple:

1. Inspect held-out curves, component metrics, and representative rollouts.
2. State one hypothesis about a failure mode or opportunity.
3. Change a bounded part of the configuration or environment.
4. Re-run both evaluations and separate model behavior from infrastructure failures.
5. Keep, reject, rerun, or retune the change, then record the decision.

Each run left enough evidence for another person to understand what changed and why. The final ledger entry for run 120 recorded:

```json
{
  "id": "120",
  "research_question": "Which complete configuration is ready for workflow testing?",
  "evidence": {
    "drug_tool_avg1": 0.812,
    "bioreason_avg1": 0.863,
    "held_out_protocol": "fixed environment splits"
  },
  "artifacts": [
    "configuration",
    "evaluation curves",
    "trace review"
  ],
  "decision": "promote"
}
```

This record mattered during handoffs and restarts. The next run began from written evidence instead of memory, and rejected directions remained visible.

### Why run 120 was promoted

Run 120 met the promotion bar on both environments. Drug Tool increased from 70.8% to 81.2% during training. BioReason reached 86.3% on its combined held-out metric.

![Drug Tool held-out evaluation](/assets/images/teaching-an-open-model-to-do-science/image_3.png)

*Figure 6. Drug Tool held-out evaluation.*

![BioReason held-out evaluation](/assets/images/teaching-an-open-model-to-do-science/image_4.png)

*Figure 7. BioReason held-out evaluation.*

![Drug Tool reward during run 120](/assets/images/teaching-an-open-model-to-do-science/image_5.png)

*Figure 8. Drug Tool reward during run 120.*

![BioReason reward during run 120](/assets/images/teaching-an-open-model-to-do-science/image_6.png)

*Figure 9. BioReason reward during run 120.*

![Combined reward during run 120](/assets/images/teaching-an-open-model-to-do-science/image_7.png)

*Figure 10. Combined reward during run 120.*

This is also where run 120 shines. It reached a BioReason accuracy of about 86%, a combined score built from GO term F1, tree similarity, and JSON structure validity together, while Drug Tool accuracy climb

The BioReason number combines several checks. It measures overlap with reference GO terms, similarity within the GO hierarchy, coverage of the three ontology aspects, and valid JSON structure. It is a composite measure of biological annotation and output compliance.

Scores narrowed the candidates, but deployment required three checks.

First, held-out evaluation measured the behaviors encoded by the environments. Second, trace and verifier review tested whether the scores came from grounded tool use and valid structured outputs. Third, candidates ran inside the scientific application for qualitative workflow testing.

That last stage checked behavior the aggregate metrics could miss: whether evidence seeking was purposeful, whether citations and identifiers were usable, whether the model handled ambiguity, and whether its synthesis acknowledged uncertainty. The team called this “vibe testing,” but the criteria were concrete.

Run 120 passed all three stages. That is why it became the promoted checkpoint.

### Putting the adapter in a scientific application

The promoted adapter runs in a companion AI Scientist application built with Strands, FastAPI, and React. Specialist tasks route to the run 120 adapter. Base Trinity Mini remains available for orchestration and routing that benefit from its larger context window.

The top-level orchestrator delegates to six named specialists. Two of those specialists coordinate narrower sub-specialists. Counting the three flat specialists, the critic, and four nested specialists gives eight agents below the top-level orchestrator. Each has a limited job and its own visible tools, sources, and artifact identifiers.

Four runtime choices make that delegation easier to inspect:

- `ask_user` lets only the top-level orchestrator pause for clarification. Specialist delegations are headless and complete in one pass.
- `execute_python_code` runs in an isolated AWS Bedrock AgentCore sandbox rather than in the host process.
- `/plan`, `/report`, and `/hypothesize` invoke fixed flows instead of free-form chat. They produce downloadable plans, cited reports, or competing evidence-based hypotheses.
- Domain skills load only when needed. Examples include scientific critical thinking and brainstorming at the orchestrator, literature rigor for the literature specialist, ADMET reasoning for biomedical intelligence, and peer review for the critic.

The repository includes an ECS Fargate topology with Cognito, an Application Load Balancer, managed model serving with LoRA support, session artifacts, secrets, and observability. The serving layer can host Trinity Mini adapters without tying the application to a single model family.

![AWS architecture for the AI Scientist agentic harness](/assets/images/teaching-an-open-model-to-do-science/report/agentic-harness-aws-architecture.png)

*Figure 16. Agentic harness deployment. The React frontend and FastAPI/Strands backend run on Amazon ECS Fargate. The backend coordinates the post-trained Trinity Mini endpoint, biomedical APIs, NVIDIA scientific endpoints, session artifacts, secrets, authentication, and observability. Click the figure to zoom into the full-resolution view.*

### A reusable post-training method

This project offers a practical route for teams that need specialized model behavior but do not want to pretrain a foundation model.

Start with a workflow that matters. Define what useful behavior looks like in that workflow. Build held-out tests and inspectable rewards around it. Then train a small adapter, review the traces, and deploy it inside an operating boundary the organization controls.

For a life-sciences team, the workflow might be target evidence synthesis, biological reasoning over an internal knowledge base, or tool-assisted analysis. The same method can apply to other regulated or technical work where an organization already has its own data, procedures, and standards for judging evidence.

The stack used here remains open at each layer: base weights, training code, environments, reward, and deployment. That gives the team control over policy versions, data flow, evaluation, serving, and future updates.

A 3-billion-active-parameter open model, trained through a 21-run program on two scientific environments, reached 81.2% on the held-out Drug Tool evaluation and 86.3% on the held-out BioReason composite. Trace review and workflow testing supported the decision to promote that checkpoint.

The method is the reusable part: encode a scientific workflow as actions and verifiable outcomes, keep the experiments reproducible, and promote a model only when the scores and the behavior agree.

### Collaboration

Arcee supplied Trinity Mini, an adaptable open model with a practical active-parameter footprint. Prime Intellect supplied the open-source `prime-rl` training system. AWS supplied infrastructure for accelerated training, artifact storage, model serving, application services, security, and observability. Loka built the biomedical datasets, environments, verifiers, research process, and scientific application.

Petar Kalinovski’s earlier [Trinity Mini DrugProt-Think](https://lokahq.github.io/tech-blog/blog/trinity-drugprot/) project showed that a compact open model could learn a scientific specialty and set the direction for this collaboration.


### Citation

If you find this work useful, please cite it below.

```json
@misc{jakimovski2026teachingopenmodel,
title        = {Teaching an Open Model to Do Science},
author       = {Jakimovski, Bojan and Kovachovska, Sara and Panahi, Maziyar},
year         = {2026},
month        = jul,
howpublished = {Blog post},
url          = {https://github.com/LokaHQ/prime-rl-drug-discovery-copilot}
}
```

#### Further reading

1. [Amazon Web Services](https://aws.amazon.com/). Infrastructure foundation for accelerated training, artifact storage, operations, model serving, and the production-shaped agentic harness architecture described in this report.
2. [Arcee Trinity Mini model card](https://huggingface.co/arcee-ai/Trinity-Mini). Architecture, context length, license, inference settings, and serving examples.
3. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). Introduces Group Relative Policy Optimization (GRPO).
4. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). The adapter method used for parameter-efficient post-training.
5. [Prime RL](https://github.com/PrimeIntellect-ai/prime-rl). The open-source RL training system underlying the experiment pack.
6. [BioReason-Pro paper](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1). Biological-reasoning benchmark and data source.
7. [BioReason-Pro RL reasoning data](https://huggingface.co/datasets/wanglab/bioreason-pro-rl-reasoning-data) and [test data](https://huggingface.co/datasets/wanglab/bioreason-pro-test-data).
8. [Loka BioReason RL dataset](https://huggingface.co/datasets/lokahq/bioreason-rl). Curated dataset used by the final configurations.
9. [Loka Drug Tool SFT dataset](https://huggingface.co/datasets/lokahq/drug-tool-sft). Reference prompts and tool-use traces.
10. [Prime RL Drug Tool environment](https://app.primeintellect.ai/dashboard/environments/lokahq/drug-tool-rl) and [BioReason environment](https://app.primeintellect.ai/dashboard/environments/lokahq/bioreason-go-rl).
11. [Introducing GeneBench-Pro](https://openai.com/index/introducing-genebench-pro/). Research-level evaluation of judgment-heavy computational biology.
12. [GPT-Rosalind](https://openai.com/gpt-rosalind/). OpenAI's purpose-built model for life-sciences reasoning, evidence synthesis, scientific tool use, and multi-step research workflows.
13. [Claude Science](https://www.anthropic.com/news/claude-science-ai-workbench). Anthropic's scientific workbench for connected tools, compute, code, and auditable research artifacts.
14. [Build an AI Scientist with NVIDIA BioNeMo Agent Toolkit](https://developer.nvidia.com/blog/build-an-ai-scientist-for-life-science-discovery-with-nvidia-bionemo-agent-toolkit/). A tool-connected life-science agent workflow.
15. [Learning to Replicate Expert Judgment in Financial Tasks](https://thinkingmachines.ai/news/learning-to-replicate-expert-judgment-in-financial-tasks/). Task-specific post-training and expert judgment.
16. [Autonomous AI research for nanoGPT speedrun](https://www.primeintellect.ai/auto-nanogpt). Prime Intellect's durable-harness auto-research loop.
17. [Anthropic statement on Fable and Mythos access](https://www.anthropic.com/news/fable-mythos-access). A concrete example of frontier-model access changing through external policy.
18. [GPT-5.6 availability](https://help.openai.com/en/articles/20001354-gpt-56-in-chatgpt/). Official OpenAI guidance on plan, product, workspace, regional, and safeguard constraints.
19. [Trinity Mini DrugProt-Think](https://lokahq.github.io/tech-blog/blog/trinity-drugprot/). The earlier project that led into this collaboration.
20. [GEPA](https://github.com/gepa-ai/gepa). Reflective prompt evolution for optimizing text-based components against task-specific evaluation.
