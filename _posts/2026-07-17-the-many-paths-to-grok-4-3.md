---
layout: post
title: "The Many Paths to Grok 4.3"
description: "A benchmark of Grok 4.3 through Amazon Bedrock Mantle and the xAI API, covering accuracy, latency, cost, reliability, and enterprise deployment tradeoffs."
author: shekswess
date: 2026-07-17 00:00:00 +0200
permalink: /the-many-paths-to-grok-4-3.html
categories: [AI, LLM]
tags: [xAI, Grok 4.3, Amazon Bedrock, Bedrock Mantle, Benchmarking, AWS]
image: /assets/images/grok-benchmark/background.png
source_title: "The Many Paths to Grok 4.3"
source_url: "https://lokahq.github.io/tech-blog/blog/grok-bedrock-xai/"
source_authors: ["Nina Cvetkovska", "Bojan Jakimovski"]
---

_**Original Source of the blog post: [Benchmarking Grok 4.3 on Amazon Bedrock Mantle vs. the xAI API: The Engineering Walkthrough with Loka](https://lokahq.github.io/tech-blog/blog/grok-bedrock-xai/)**_

_**Blog was written with my colleague from Loka & AWS: Nina Cvetkovska**_

## Why this matters

In June 2026, [xAI joined Amazon Bedrock as a model provider](https://aws.amazon.com/about-aws/whats-new/2026/06/grok-amazon-bedrock/), and Grok 4.3 became the first model to run on [Mantle, Bedrock's new OpenAI-compatible inference engine](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-xai-grok-4-3.html). For teams that already live inside AWS, that's a bigger deal than it sounds: it means access to xAI's reasoning-first model without a new vendor relationship, a new billing pipeline, or a security review that starts from zero.

But availability is only half the story. The question we kept hearing from clients wasn't whether Grok 4.3 is good; [xAI's own benchmarks](https://x.ai/news/grok-amazon-bedrock) make that case. The question was more practical: *does anything change when the same model runs through a different door?* Does the AWS-native path affect accuracy? Latency? Does it fail differently under load? Nobody had published a straight answer, so we ran the experiment ourselves.

This isn't a leaderboard post, and it isn't an implementation walkthrough. We took the same workload, the same evaluation, and pointed it at both routes to Grok 4.3, Amazon Bedrock Mantle and the first-party xAI API, and measured what actually differed.

> The short version: once reasoning was enabled, final accuracy was nearly identical on both paths. Where the paths genuinely diverged was latency, reliability behavior, and the operational machinery around the model. In our experience, that is exactly where enterprise deployments succeed or stall.

## What we wanted to learn

If your organization has standardized on AWS (procurement, security review, IAM, regional architecture, cost allocation), then Bedrock Mantle is the path of least resistance to Grok 4.3. The question is whether "least resistance" comes with tradeoffs you'd want to know about before committing.

So we framed the benchmark around the questions a technical decision-maker would ask before recommending one path over the other:

- Does Grok 4.3 deliver the same final-answer quality through Bedrock Mantle as through the xAI API?
- How much latency does each step of reasoning effort really add?
- Where does cost start to climb, and does accuracy keep pace with it?
- Do the two paths fail in different ways?
- Which reasoning-effort setting is the sensible default?

A quick note on scope: Mantle itself deserves more than a passing mention, and we give it one [later in the post](#bigger-picture). For now, the relevant point is that it puts Grok 4.3 behind AWS billing, governance, and regional deployment patterns your organization has likely already approved.

## What we tested

We ran Grok 4.3 against the full [GSM8K](https://arxiv.org/abs/2110.14168) test split: **1,319** grade-school math word problems. GSM8K is a narrow benchmark, and we chose it deliberately for that reason. Because the task is so well-defined (read a short problem, produce one correct number), it gives us a clean, controlled way to compare final-answer accuracy, reasoning behavior, latency, and cost across the two paths without application-specific prompt design muddying the signal. (The dataset is [openly available](https://github.com/openai/grade-school-math) if you want to reproduce the setup.)

Both paths got identical treatment: the same prompt shape, streaming enabled, one request at a time, and all four reasoning-effort settings that Grok 4.3 exposes (`none`, `low`, `medium`, and `high`). For each run we measured final-answer accuracy, p50 and p95 end-to-end latency, time to first visible token, median reasoning-token usage, and estimated run cost.

Scoring was deliberately unforgiving: the final numeric answer had to match the reference exactly. A beautifully reasoned explanation ending in the wrong number scored zero. Strict scoring keeps the comparison honest and easy to interpret.

## What we got

The most useful finding is also the simplest: **reasoning effort mattered far more than provider path** for final accuracy. With reasoning turned on at any level, both paths landed in the same **96–97%** band. Whichever door you walk through, you get essentially the same model.

Turning reasoning off entirely was the fastest and cheapest configuration, and also the least accurate. The single biggest jump in the whole benchmark came from switching to `low` effort. On Bedrock, accuracy climbed from **93.10%** to **96.51%**. On the xAI API, from **94.16%** to **96.89%**.

![GSM8K accuracy by provider and reasoning effort](/assets/images/grok-benchmark/image.png)

*Figure 1. GSM8K accuracy by provider and reasoning effort. Low effort captured most of the accuracy gain.*


After that, the curve went flat. `medium` and `high` bought essentially nothing on GSM8K: Bedrock reached **96.59%** at medium and actually dipped to **96.13%** at high, while xAI reached **96.97%** at medium and **96.89%** at high. Those differences are noise, not signal.

The clearest way to see it is to plot each step up in reasoning effort against the extra median latency it costs. The move from `none` to `low` sits in the part of the chart you want to be in: a large accuracy gain for a few additional seconds. Every step after that clusters near zero accuracy gain while latency keeps climbing.

![Accuracy gain versus added median latency by reasoning-effort step](/assets/images/grok-benchmark/image_2.png)

*Figure 2. Accuracy gain versus added median latency for each reasoning-effort step. The first step to low effort delivered the meaningful payoff.*


To be clear, this doesn't mean higher effort is never worth it; harder workloads may well need the extra thinking budget. It means that for structured arithmetic like GSM8K, the model had all the room it needed at `low`. Beyond that point, you're mostly buying tokens, not accuracy.

## Reading the tradeoff

Latency is where the two paths stopped looking identical. Enabling `low` effort raised median latency on both: Bedrock moved from **0.79s** to **3.56s**, and xAI from **1.05s** to **2.81s**. That is the expected price of reasoning. The interesting part is the tail.

Anyone who has shipped a user-facing product knows the tail is what users remember. A workflow can feel perfectly fine at the median and still feel broken when p95 spikes. In this run, the xAI API held a meaningfully tighter tail at low effort: **5.45s** at p95 versus **9.70s** on Bedrock.

![End-to-end p50 and p95 latency by provider](/assets/images/grok-benchmark/image_3.png)

*Figure 3. End-to-end p50 and p95 latency. The provider path showed up more clearly in latency than in final accuracy.*


The token story told the same tale from another angle. `low` effort used roughly **260–290** median reasoning tokens per question. `medium` and `high` pushed that past **400** on both paths while, as we saw above, accuracy barely moved.

![Median reasoning tokens by reasoning effort](/assets/images/grok-benchmark/image_4.png)

*Figure 4. Median reasoning tokens by effort. Higher effort increased reasoning-token usage more clearly than it improved GSM8K accuracy.*


Cost tracked the reasoning tokens almost exactly, which makes the recommendation straightforward: `low` effort was the sweet spot in this benchmark. It captured most of the quality improvement without paying the full latency and token bill of medium or high.

### Full benchmark view

Here's everything in one place: accuracy, latency, throughput, reasoning tokens, and cost for all eight runs.

| Run | Provider | Effort | Accuracy | p50 | p95 | TTFT p50 | Tok/s p50 | Reason p50 | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bedrock-grok43-none` | Bedrock | none | 93.10% | 0.79 | 1.72 | 0.62 | 88.3 | 0 | $0.4418 |
| `bedrock-grok43-low` | Bedrock | low | 96.51% | 3.56 | 9.70 | 3.14 | 107.6 | 289 | $1.5347 |
| `bedrock-grok43-medium` | Bedrock | medium | 96.59% | 3.81 | 11.24 | 3.46 | 159.0 | 467 | $2.4169 |
| `bedrock-grok43-high` | Bedrock | high | 96.13% | 3.61 | 10.41 | 3.32 | 165.8 | 459 | $2.4400 |
| `xai-grok43-none` | xAI | none | 94.16% | 1.05 | 1.56 | 0.67 | 54.8 | 0 | $0.5883 |
| `xai-grok43-low` | xAI | low | 96.89% | 2.81 | 5.45 | 0.71 | 111.1 | 264 | $1.5323 |
| `xai-grok43-medium` | xAI | medium | 96.97% | 3.87 | 9.66 | 0.71 | 116.3 | 399 | $2.1449 |
| `xai-grok43-high` | xAI | high | 96.89% | 3.78 | 9.96 | 0.67 | 118.2 | 412 | $2.2431 |

*p50 and p95 are measured in seconds. Reasoning tokens are medians. Cost is the estimated cost for the full 1,319-question run.*

## Reliability behavior

Here's the part that never shows up on a leaderboard. In one early Bedrock low-effort pass, two of the first ten prompts were stopped by an automated prompt-safety check before the model produced any answer at all. The prompts were ordinary GSM8K math problems, questions about apples and train schedules. That's not a model-quality issue. It's a property of the path the request travels through.

And that distinction matters enormously for production planning. A blocked request, a timeout, and a wrong answer are three different failure modes with three different fixes, and collapsing them into one accuracy score hides the information you need most. Accuracy tells you whether the model solved the task. Reliability tells you whether the path consistently delivered a scoreable answer in the first place.

> This is why we always recommend benchmarking through your actual deployment path, not a proxy for it. Governance layers, safety filters, streaming behavior, and regional infrastructure all shape the user experience, even when the model underneath is identical.

## The bigger picture: why this pairing matters

Before the takeaways, it's worth zooming out. We've spent this post measuring one workload through two doors, and being picky about tail latencies and safety filters along the way. None of that picking should obscure the larger point: both sides of this partnership shipped something genuinely useful.

### What AWS built with Mantle

Mantle is not a wrapper around a third-party API. It is a [new inference engine inside Amazon Bedrock](https://aws.amazon.com/about-aws/whats-new/2026/06/grok-amazon-bedrock/), designed for price performance and shipped with the full production feature set from day one: response streaming, structured outputs, tool calling, and Standard and Priority service tiers.

The design choice that matters most for adoption speed is API compatibility. Mantle speaks the OpenAI API specification, so a team already built on OpenAI SDKs can point `base_url` at the [Mantle endpoint](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-xai-grok-4-3.html), set the model to `xai.grok-4.3`, and be running inside Bedrock with minimal code changes. Meanwhile everything an enterprise actually worries about (billing, IAM, governance, procurement, regional controls) stays on rails the organization has already approved.

> As an AWS partner, we read Mantle as infrastructure, not a one-off integration. AWS built an OpenAI-compatible runtime into Bedrock, which positions the platform to onboard future third-party models the same way. Model choice without vendor sprawl is the whole promise of Bedrock, and Mantle extends it.

### What xAI brings with Grok 4.3

On the model side, Grok 4.3 earned its place in the catalog. It is a reasoning-first design: reasoning is always active rather than optional, with effort configurable from `none` to `high`, which is exactly the lever our benchmark shows paying off. It ships a [1-million-token context window](https://x.ai/news/grok-amazon-bedrock), and xAI reports top marks on independent evaluations, including the lowest hallucination rate among frontier models on Artificial Analysis's Omniscience benchmark and leading results on real-world tool-calling tests.

That profile maps cleanly onto the workloads enterprises are trying to ship right now: contract review, case law research, financial document Q&A, and customer-support agents that need reliable tool calling over long context. Combined with Bedrock's on-demand pricing for the model, it makes Grok 4.3 one of the most cost-effective frontier reasoning options on AWS today.

Our results add the practical footnote: the combination holds up under measurement. Same model quality through the AWS route, with the operational conveniences that come with it.

## Takeaways for decision-makers

If you're a CTO, Head of AI, or solutions architect weighing this decision, the framing that helps most is this: you're not choosing between two models. You're choosing between two operational environments around the same model. Here's what our data says about that choice:

- **The strategic value of Bedrock Mantle is AWS-native access.** Grok 4.3 arrives inside your existing billing, governance, IAM, and deployment patterns, with no new vendor onboarding and no parallel operational stack.
- **Model quality is effectively path-independent.** With reasoning enabled, both paths landed in the same 96–97% accuracy band on GSM8K. You're not sacrificing capability for convenience.
- **`low` reasoning effort was the best default in this benchmark.** It delivered nearly all of the accuracy lift at a fraction of the token and latency cost of medium or high.
- **Medium and high effort need to earn their keep.** On this workload they increased reasoning tokens and cost far more than they improved accuracy. Justify them with your own workload data, not intuition.
- **The paths diverge in latency shape and reliability behavior, not accuracy.** Tail latency (p95) and pre-model safety filtering are where the differences live, and where your production architecture review should focus.

## Conclusion

Grok 4.3 landing on Bedrock Mantle changes the adoption conversation for AWS-centered organizations. Governance, procurement, regional planning, streaming, structured outputs, and tool calling stop being separate integration projects and become part of the same motion as adopting the model itself. That's the real headline; the benchmark just tells you what it costs.

And what it costs turns out to be encouragingly little. Our practical recommendation from 10,552 scored requests: start with `low` reasoning effort for structured, GSM8K-like workloads. It captured most of the accuracy gain while keeping latency and cost under control. Reach for `medium` or `high` only when your own evaluation proves the extra reasoning budget pays for itself, because on this workload it didn't.

Finally, evaluate the path for what it actually changes: latency shape, tail behavior, reliability under safety filtering, governance fit, and day-two operations. When final accuracy is this close, the right decision stops being about the model alone and starts being about the system you wrap around it.

*If you're evaluating Grok 4.3, or any foundation model, for a production workload on AWS, this is the kind of benchmarking [Loka's AI engineering team](https://www.loka.com/) runs for clients every week. We're happy to compare notes.*


## Citation

If you find this work useful, please cite:

```
@misc{loka2026grok43bedrock,
  title        = {The Many Paths to Grok 4.3},
  author       = {Cvetkovska, Nina and Jakimovski, Bojan},
  year         = {2026},
  month        = {jul},
  howpublished = {Blog post},
  url          = {https://lokahq.github.io/tech-blog/blog/grok-bedrock-xai/}
}
```

<a id="references"></a>
## Further reading

1. AWS (2026). *Grok 4.3 from xAI now available in Amazon Bedrock.* Official AWS announcement covering regional availability and Bedrock Mantle integration. [aws.amazon.com](https://aws.amazon.com/about-aws/whats-new/2026/06/grok-amazon-bedrock/)
2. xAI (2026). *Grok on Amazon Bedrock.* xAI's launch post, including model positioning and benchmark claims. [x.ai](https://x.ai/news/grok-amazon-bedrock)
3. AWS (2026). *Grok 4.3 model card.* Amazon Bedrock User Guide covering endpoints, reasoning-effort configuration, service tiers, and quotas for the Mantle path. [docs.aws.amazon.com](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-xai-grok-4-3.html)
4. xAI (2026). *API documentation.* Reference for the first-party path used in this comparison. [docs.x.ai](https://docs.x.ai/)
5. Cobbe et al. (2021). *Training Verifiers to Solve Math Word Problems.* The paper that introduced GSM8K. [arxiv.org](https://arxiv.org/abs/2110.14168)
6. OpenAI (2021). *GSM8K dataset.* The openly available test split used in this benchmark. [github.com](https://github.com/openai/grade-school-math)
