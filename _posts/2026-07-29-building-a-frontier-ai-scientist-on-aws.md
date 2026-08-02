---
layout: post
title: "Building a Frontier AI Scientist on AWS"
description: "How Loka and Arcee AI post-trained an open-weight AI scientist agent and deployed it on AWS."
author: shekswess
date: 2026-07-29 00:00:00 +0200
permalink: /building-a-frontier-ai-scientist-on-aws.html
categories: [AI, AWS]
tags: [AI Scientist, Trinity Mini, RLVR, GRPO, LoRA, Strands Agents, Amazon Trainium, SageMaker, Biomedical AI]
image: https://cdn.prod.website-files.com/649228879157da0fb68c3bee/6a675d9d0603dff7ebafe223_banner_Arcee_Jul26_2026_V1.png
source_title: "Building a Frontier AI Scientist on AWS"
source_url: "https://www.loka.com/blog/building-a-frontier-ai-scientist-copilot-on-aws"
source_authors: ["Bojan Jakimovski", "Sara Kovachovska", "Maziyar Panahi"]
---

_**Original Source of the blog post: [Building a Frontier AI Scientist on AWS](https://www.loka.com/blog/building-a-frontier-ai-scientist-copilot-on-aws)**_

_**Blog was written with my colleagues from Loka & Arcee: Sara Kovachovska & Maziyar Panahi**_

Scientific research like drug discovery lives in the gap between dozens of specialized databases and tools. A single question like "What does this protein do, and can we design a molecule against it?" can mean touching UniProt, KEGG, PubMed, STRING, a structure predictor, and a molecule generator, then reasoning over everything that comes back. An agentic harness closes that gap by planning, calling the right tools, reading the evidence, and answering without inventing certainty.

For quite some time, the standard approach has been to rely on large frontier models. This strategy, however, often limits ownership, precludes the ability to post-train on specific behaviors or task rewards, and incurs significant costs with every API call.

<figure class="post-video">
<iframe title="vimeo-player" src="https://player.vimeo.com/video/1213577217?h=f1fa5aac78" width="640" height="360" frameborder="0" referrerpolicy="strict-origin-when-cross-origin" allow="autoplay; fullscreen; picture-in-picture; clipboard-write; encrypted-media; web-share" allowfullscreen></iframe>
</figure>

Loka, partnering with Arcee AI and AWS, took the other path. We post-trained **Arcee Trinity Mini**, a small, open-weight frontier Mixture-of-Experts model (\~26B total, \~3B active per token) into a Biomedical AI Scientist with reinforcement learning from verifiable rewards (RLVR). The wager is that a small, post-trained model can reach frontier-level quality on a focused domain without the large-frontier-model bill, and the defining property of this work is that the **entire lifecycle runs on AWS**: training on Amazon EC2, the harness on AWS-native services with Strands Agents, and serving on AWS Trainium and Amazon SageMaker AI.

This is the engineering story behind that.

## **Post-Training on Amazon EC2** 

So we post-train Trinity Mini with GRPO and a LoRA adapter on the frozen base, against two custom-made verifiable environments from the Prime Intellect Environments Hub — [lokahq/drug-tool-rl](https://app.primeintellect.ai/dashboard/environments/lokahq/drug-tool-rl) for multi-turn evidence-seeking tool use, and [lokahq/bioreason-go-rl](https://app.primeintellect.ai/dashboard/environments/lokahq/bioreason-go-rl) for structured Gene Ontology function prediction. Both reward checkable behavior, so RL moves real quality rather than overfitting a judge, and a run is promoted only when it improves the held-out evaluation for both.

The infrastructure underneath is where AWS does the heavy lifting. Training runs on an **Amazon EC2** `p5e.48xlarge` instance with **8× NVIDIA H200 GPUs** was the right home for three concrete reasons:

1. **Capacity on demand**: H200-class instances are available when an experiment is ready to launch, instead of waiting on a fixed cluster.  
2. **Reproducibility:** the entire training box is provisioned as infrastructure-as-code with  Terraform, so the environment is rebuildable rather than a hand-tuned snowflake.  
3. **Operational simplicity:** day-to-day launching and monitoring goes through **AWS Systems Manager (SSM),** which is headless, secure, and keyless access with no SSH or VPN to manage, while every checkpoint and rollout streams to **Amazon S3**, so the large local NVMe scratch is never a single point of failure, and any run can be restored on demand.

![][image2]

*Figure 1: The GRPO training pipeline on a single Amazon EC2 p5e.48xlarge (8× H200). A trainer updates a LoRA policy on Trinity Mini while a co-located inference server generates rollouts against the drug and BioReason environments; rewards drive the GRPO update, and checkpoints, adapters, and rollouts stream to Amazon S3, with runs controlled through AWS Systems Manager.*

Post-training delivered steady, stable gains across the run. Starting from the raw Trinity Mini checkpoint, held-out Drug Tool accuracy climbed cleanly and monotonically **from** **70.8% to 81.2%**, while held-out BioReason accuracy climbed **from 31.7% to 86.3%**, nearly tripling, with no instability along the way. The promoted configuration is published as a reproduction package: [trinity-mini-ai-scientist](https://github.com/LokaHQ/trinity-mini-ai-scientist) ships the exact `rl/config.toml` behind this run, alongside the post-trained model on Hugging Face.

## **The Agentic Harness**

A checkpoint that scores well on held-out evals still has to prove itself in the real agent environment, the Agentic Harness, where a researcher asks a question and the system delegates, calls tools, and streams a grounded answer. The harness is built with **Strands Agents**, AWS's open-source agent framework, and deployed entirely on AWS-native services.  


![][image3]


*Figure 2: Agent Application Architecture. AI Scientist's AWS-native deployment: a Strands multi-agent backend on AWS Fargate, served by Amazon SageMaker AI, behind Amazon ECS, Cognito, and an Application Load Balancer — with session state in Amazon S3 and observability via Amazon Bedrock AgentCore.*  

### **The Key Architectural Decision**

There are some important decisions on why we have built the whole harness on AWS:

1. **Routing & Auth (Amazon Route 53, Application Load Balancer, Amazon Cognito):** We front the harness with Route 53 and an Application Load Balancer, and handle authentication through a Cognito-hosted OAuth user pool, so access control is managed rather than bolted on.  
2. **Compute (AWS Fargate on Amazon ECS):** Two containerized services run in one ECS cluster, a Vite/React frontend and a FastAPI backend, each scaling independently without managing servers.  
3. **The Agent (Strands Agents):** The backend service runs a Strands orchestrator that delegates to six named specialists rather than calling tools itself, two of them mini-orchestrators over their own sub-specialists. Every specialist's tool calls, sources, and artifacts stay inspectable, with session and artifact state persisted in **Amazon S3**, where oversized tool results are also offloaded so a long multi-turn session never blows the context window.  
4. **The Brain (Amazon SageMaker AI):** We serve the post-trained Trinity Mini behind a SageMaker AI endpoint. Because the harness is model-agnostic, we can swap a single setting to compare it head-to-head against a hosted frontier model.  
5. **Tools (NVIDIA Endpoints, Biomedical APIs and AWS-native tooling):** The specialists call the same tools the model trained against: NVIDIA-hosted endpoints for ESMFold, MolMIM, and DiffDock, plus KEGG, GEO, STRING, PubMed, UniProt, PDB, and PubChem. A biomedical-intelligence specialist discovers the right call at runtime over a ToolUniverse catalog spanning OpenTargets, ChEMBL, clinical trials, FDA drug labels and ADMET/pharmacogenomics data instead of one hard-wired tool per API, and when a curated source comes up empty the orchestrator falls back to a web-search tool served through Amazon Bedrock AgentCore Gateway \- always labeled as web-sourced. Alongside these, the harness ships its own tools for molecule depiction, SMILES validation, protein domain maps, and charting, with `execute_python_code` running inside an isolated Amazon Bedrock AgentCore Code Interpreter sandbox. A gain in the training environment translates directly into better product behavior.  
6. **Delivery & Observability (Amazon ECR, AWS Secrets Manager, Amazon Bedrock AgentCore, Amazon CloudWatch):** We build images in CI and push them to Amazon ECR, keep secrets in AWS Secrets Manager, and trace every agent step through Amazon Bedrock AgentCore Observability with logs in Amazon CloudWatch.

   
Because the model trains against the exact tools it will use in production, swapping in a new checkpoint is frictionless — and that is how a promotion candidate is judged: on real tasks like folding a protein, generating a molecule, and recovering gracefully from a failed tool result.  


## **Inference and Hosting on AWS Trainium** 

For a domain specialized agentic harness accessed frequently, serving economics decides whether it ships *\-* and this is where a small, open-weight model pays off on AWS. We host the post-trained Trinity Mini on **AWS Trainium2 (Trn2)**, AWS's purpose-built AI silicon, through the AWS Neuron SDK (`neuronx-distributed-inference`). On a `trn2.3xlarge` instance *—* tensor-parallel across four NeuronCores, BF16, Neuron SDK 2.28 Trinity Mini serves at **109.5 tokens per second** for single-stream generation and up to **295 tokens per second** at batch size 4, with a 371 ms time-to-first-token and 9.1 ms per output token comfortably within interactive range for a streaming agent, with a fused Mixture-of-Experts kernel adding a further 29% throughput and 22.5% latency improvement.

The reason this matters is cost. AWS positions Trn2 at **up to 30–40% better price-performance than comparable NVIDIA P5e/P5en GPU instances**, with more readily available capacity — so the same workload runs for materially less. The model is exposed behind a standard, OpenAI-compatible Amazon SageMaker AI endpoint. Hence, the harness consumes it through configuration with no changes: Trainium provides the economical compute, SageMaker the managed serving surface.

|  | AWS Trainium2 Instance | Comparable NVIDIA GPU Instance |
| :---- | :---- | :---- |
| **Representative instance** | trn2.3xlarge | p5.4xlarge |
| **Price / hr** | \~$2.24     | \~$4.72 |
| **Relative serving cost** | \~52.5% lower  | baseline |

*Table 1: Cost comparison of Trainium2 based Instance vs. H100 GPU based Instance for hosting Arcee Trinity Mini model.*


## **Conclusion**

Building a strong frontier domain specialized agent is not just a modeling problem; it is also an infrastructure problem. Post-training, wrapping the model in a production agentic harness, and serving it economically are three different jobs, and running all three on AWS is what makes the loop hold together. 

This is why Trinity Mini hits the engineering sweet spot for a domain agent: 

1. **Ownable & post-trainable:** an open-weight model you can shape with verifiable rewards and keep — no API lock-in.  
2. **Verifiable agentic gains:** training against the same tools the product uses, so eval improvements translate into real workflow behavior.  
3. **AWS-native and cost-effective**: the whole lifecycle — training, harness, and serving — runs on AWS, with Trainium making production serving economical at scale.

By post-training a small open-weight model and serving it on AWS Trainium, the AI Scientist moves beyond a frontier-API demo to an owned, cost-controlled production grade system a team can run and keep improving — a blueprint that generalizes to any domain where a small model must reason over verifiable tools.

A special thanks to Petar Kalinovski, whose earlier [Trinity Mini DrugProt-Think](https://lokahq.github.io/Trinity-Mini-DrugProt-Think) project first showed that a compact open model could learn a real scientific specialty, and kicked off this whole effort.

[image1]: /assets/images/frontier-ai-scientist-aws/image1.png

[image2]: /assets/images/frontier-ai-scientist-aws/image2.png

[image3]: /assets/images/frontier-ai-scientist-aws/image3.png
