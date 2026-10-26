---
layout: "post"
title: "Celebrating the release of TorchNeuron"
description: "Bojan Jakimovski shares Loka's journey from BEVFusion on AWS silicon to native PyTorch with TorchNeuron, ESMC on Trainium2, and the AWS Trainium Frontier Competition."
author: "shekswess"
date: "2026-10-13 00:00:00 +0000"
permalink: "/celebrating-the-release-of-torchneuron.html"
categories: ["AWS", "PyTorch"]
tags: ["TorchNeuron", "AWS Trainium", "PyTorch", "Annapurna Labs"]
image: "/assets/images/celebrating-the-release-of-torchneuron/celebrating-the-release-of-torchneuron.webp"
source_title: "Celebrating the release of TorchNeuron"
source_url: "https://lokahq.github.io/tech-blog/celebrating-the-release-of-torchneuron/"
source_authors: ["Bojan Jakimovski"]
---

_**Original Source of the blog post: [Celebrating the release of TorchNeuron](https://lokahq.github.io/tech-blog/celebrating-the-release-of-torchneuron/)**_

![TorchNeuron: Native PyTorch on AWS Silicon. An orange PyTorch emblem floats above a stylized blue chip labeled AWS Silicon, connected by blue and yellow circuit traces on a deep navy background.](/assets/images/celebrating-the-release-of-torchneuron/celebrating-the-release-of-torchneuron.webp)

TorchNeuron is here, and we’re excited about what we can build with it. At Loka, we’re already bringing its native PyTorch workflow into our research and client work. Our journey on AWS Silicon started with real customer workloads and took us through custom kernels, applied research, and conversations with Annapurna Labs about the tools we wanted to use. A native PyTorch backend was high on that list, and now it’s finally here.

## Getting started with AWS Silicon

We got started at the end of 2024 with two client projects on AWS Silicon. [Sebastian Gonzalez Aseretto](https://www.linkedin.com/in/sebastian-gonzalez-aseretto/) and [Nicolás Roldán Fajardo](https://www.linkedin.com/in/nicolas-roldan-fajardo/) worked on one. On the other project, I worked with [Zafir Stojanovski](https://www.linkedin.com/in/zafstojano/) and [Henrique Ribeiro Delgado da Silva](https://www.linkedin.com/in/henrique-ribeiro-silva/). Real customer workloads gave us a practical way to learn what these chips could do and where we needed to dig deeper.

Zafir, Henrique, and I focused on migrating BEVFusion to AWS Neuron. The Neuron Kernel Interface (NKI) had just become available, so we experimented with NKI kernels and wrote custom C++ operations too. I enjoyed working through those details and getting closer to the hardware.

We put what we’d learned into a report and presentation for Annapurna Labs, covering what worked well and where the developer experience could improve. A native PyTorch backend was high on our list: we wanted to bring our models and training code over using the tools we already knew. We sent the deck over, and it started making its way around the team.

One moment at AWS Summit New York in 2025 caught me completely off guard. We were talking with the Annapurna Labs team at their booth about our work when they recognized us as the Loka team behind the presentation. Then one of them pulled out his phone and brought up our deck. I was stunned. It was the same presentation Zafir, Henrique, and I had put together, now being shown back to us in the middle of the conversation. I hadn’t expected them to recognize our work, let alone pull up the deck right there. We even have a photo from that moment.

<figure>
<div style="display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem;">
<a href="/assets/images/celebrating-the-release-of-torchneuron/annapurna-labs-aws-summit-2025.png" style="flex: 3 1 15rem; min-width: 0;">
<img alt="An Annapurna Labs team member holds up a phone displaying Loka's BEVFusion presentation at the AWS Summit New York booth." decoding="async" height="640" loading="lazy" src="/assets/images/celebrating-the-release-of-torchneuron/annapurna-labs-aws-summit-2025.webp" style="height: auto;" width="480"/>
</a>
<a href="/assets/images/celebrating-the-release-of-torchneuron/aws-summit-2025-group.png" style="flex: 4 1 20rem; min-width: 0;">
<img alt="Bojan Jakimovski takes a group selfie with two AWS team members at AWS Summit New York 2025." decoding="async" height="800" loading="lazy" src="/assets/images/celebrating-the-release-of-torchneuron/aws-summit-2025-group.webp" style="height: auto;" width="800"/>
</a>
</div>
<figcaption>Our BEVFusion presentation on a phone at the Annapurna Labs booth, AWS Summit New York 2025, alongside Bojan Jakimovski with the AWS team. Select either photo to view the original.</figcaption>
</figure>

## The native PyTorch workflow we wanted

We stayed in touch after New York, and Annapurna Labs gave us early access to the first TorchNeuron beta. And trust me, the appeal was immediate. With [TorchNeuron’s native backend](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/pytorch-native-overview.html) installed, we can move a supported model and its input tensors onto the Neuron device with `.to("neuron")` and start experimenting in eager mode. For compiled workloads, we keep that device placement and use `torch.compile(model, backend="neuron")`. We’re working through our training and inference code with familiar PyTorch tooling, without an XLA-specific execution loop. After our conversations in 2024 and 2025, getting to run our own code this way felt especially satisfying.

That extends to distributed training. TorchNeuron supports the [PyTorch distributed APIs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/index.html), including DDP, FSDP, and DTensor, with sharding configured for the NeuronCores we use.

<figure>
<img alt="A workflow diagram connects PyTorch code and tensors to TorchNeuron using .to('neuron'), then to AWS Trainium NeuronCores, with eager mode, torch.compile, and distributed training using DDP, FSDP, and DTensor." decoding="async" height="1353" loading="lazy" src="/assets/images/celebrating-the-release-of-torchneuron/native-pytorch-workflow.webp" style="height: auto;" width="2048"/>
<figcaption>A sketch of the native TorchNeuron workflow, from device placement to eager execution, compilation, and distributed training.</figcaption>
</figure>

## Putting TorchNeuron to work

We’ve been putting that workflow to use in our applied research. One experiment I especially enjoyed was post-training Tiny ESMC in pure PyTorch, which worked like a charm. We’ve also been experimenting with SAM3, GENA-LM, and Boltz 2.2.1.

ESMC-300M gave us a concrete example of that workflow. For zero-shot protein-variant scoring, we kept the model in PyTorch, switched off Flash Attention to use the standard attention path, and compiled the transformer blocks with `torch.compile`. No model rewrite was needed. A single Trainium2 instance reached 490.8 variants per second, enough to score roughly 42.4 million a day at that rate.

At batch size 16 and sequence length 512, that was 18.6% higher throughput than our H100 baseline, with nearly identical ROC-AUC. The cost per variant was about 60% lower at the Capacity Block reservation rates we used in August 2026. For the code, setup, and full benchmarks, take a look at our [ESMC-on-Trainium2 post](https://lokahq.github.io/tech-blog/esmc-trainium2/). [João Correia](https://www.linkedin.com/in/joao-diogo-correia/), [Telmo Felgueira](https://www.linkedin.com/in/telmo-felgueira/), [Tiago Gonçalves](https://www.linkedin.com/in/tiagofilipegoncalves/), and I wrote it with [Jim Burtoft](https://www.linkedin.com/in/jimburtoft/) and [Louise Ping](https://www.linkedin.com/in/louiseping/), our friends at AWS’s Annapurna Labs.

Tiago took that ESMC work to Sword AI Summit 2026 in Portugal. In his talk, “Foundation Models on AWS Trainium,” he shared our experience running the model on Trainium2.

<figure>
<img alt="Tiago Gonçalves presents Foundation Models on AWS Trainium to an audience at Sword AI Summit 2026 in Portugal." decoding="async" height="2048" loading="lazy" src="/assets/images/celebrating-the-release-of-torchneuron/tiago-sword-ai-summit-2026.webp" style="height: auto;" width="1536"/>
<figcaption>Tiago Gonçalves presenting “Foundation Models on AWS Trainium” at Sword AI Summit 2026, Portugal.</figcaption>
</figure>

## What’s next

And we’re still cooking. Alongside our applied research, we’re working with two major clients, one in food delivery and one in streaming entertainment. We’re stress-testing ranking models in native PyTorch with one and preparing to work on LLMs with the other. The early ranking-model results have us excited. Our next steps are to keep testing the inference code, compare serving configurations, and get the LLM work underway.

Tiago and I are now competing in Phase 2 of the [AWS Trainium Frontier Competition](https://trainium-frontier.devpost.com/) with The Optimal Batch @ Loka. As of October 19, 2026, we’re third out of ten teams on the [public leaderboard](https://d3n62br6wsx257.cloudfront.net/). It’s exciting to see how far we’ve come, and we’re still pushing to improve our results.

<figure class="figure-wide">

<a href="/assets/images/celebrating-the-release-of-torchneuron/phase-2-leaderboard-2026-10-19.png" style="display: block; position: relative; overflow: hidden; aspect-ratio: 912 / 492; border-radius: 8px;">
<img alt="AWS Trainium Frontier Competition Phase 2 leaderboard showing The Optimal Batch @ Loka in third place among ten teams, with a composite score of 0.4035." decoding="async" height="690" loading="lazy" src="/assets/images/celebrating-the-release-of-torchneuron/phase-2-leaderboard-2026-10-19.webp" style="position: absolute; width: calc(100% * 1909 / 912); max-width: none; height: auto; left: calc(-100% * 502 / 912); top: calc(-100% * 87 / 492);" width="1909"/>
</a>
<figcaption>Phase 2 leaderboard as of October 19, 2026: The Optimal Batch @ Loka is third out of ten teams. Select the image to view the full-size screenshot.</figcaption>
</figure>

What excites me now is how much more we can explore with the PyTorch tools we already know, whether we’re testing a research idea or building for a client. There’s still hardware-specific work to do, from checking compatibility to tuning performance, and we’re eager to get into it. I’m grateful to everyone at Loka and Annapurna Labs who’s been part of this journey since 2024. We’ll keep sharing what we learn as the client work and the competition progress.

Give [TorchNeuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/pytorch-native-overview.html) a try on AWS Silicon. And if you’d like a hand bringing your own models and workloads over, [reach out to us at Loka](https://www.loka.com/contact). We’d love to help you get them running. Let’s put TorchNeuron to work.
