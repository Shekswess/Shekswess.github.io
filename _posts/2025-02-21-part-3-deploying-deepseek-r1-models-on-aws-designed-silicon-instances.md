---
title: "Part 3: Deploying DeepSeek-R1 Models on AWS- designed Silicon Instances" 
description: "Harnessing Open Source AI on AWS"
author: shekswess
date: 2025-02-21 00:00:00 +0800
categories: [AI, LLM]
tags: [DeepSeek, DeepSeek-R1, Generative AI, Large Language Models, LLMs, AI, Machine Learning, Open Source]
image: https://miro.medium.com/v2/resize:fit:2800/format:webp/1*0lr_BYOYcsuXOa2N-ZZNoA.png
---

_**Original Source of the blog post: [Part 3: Deploying DeepSeek-R1 Models on AWS- designed Silicon Instances](https://medium.com/loka-engineering/part-3-deploying-deepseek-r1-models-on-aws-designed-silicon-instances-0aef410d0617)**_

_**Blog was written with my colleague from Loka: Crhistian Cardona**_


In the previous parts of this series, we explored how easily we can deploy distilled DeepSeek-R1 models on AWS Bedrock and on AWS SageMaker AI. While these managed services abstract infrastructure complexities, they can come with some limitations if we want to deploy something extremely custom.

That’s why in this post, we adopt a more hands-on methodology by deploying DeepSeek-R1 distilled models on Amazon EC2 instances equipped with Inferentia and Trainium chips. These AI accelerators, designed by AWS, offer substantial cost savings and enhanced performance for inference tasks compared to traditional GPU instances.

## Why Inferentia & Trainium (Silicon Chips)?

Instances powered by Inferentia and Trainium chips deliver more economical inference solutions than their GPU-based counterparts. Notably, Inferentia-based instances can offer up to 40% cost savings, while Trainium-based instances have demonstrated up to 50% price-performance benefits in certain workloads. The AWS Neuron SDK is engineered to accelerate AI workloads by optimizing hardware utilization, ensuring efficient processing of machine learning tasks. These instances seamlessly integrate with EC2 Auto Scaling and Load Balancers, facilitating effortless scaling to meet production demands. Unlike managed services, deploying on EC2 instances grants complete control over model deployment and optimization processes, allowing for tailored configurations to meet specific requirements.

## Prerequisites

Before getting started, ensure you have the following:

- [AWS Account](https://aws.amazon.com/account/) with permissions for SageMaker AI
- [AWS CLI](https://aws.amazon.com/cli/) installed and configured
- [IAM Role](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html) with sufficient permissions to access EC2 and other necessary services.
- Security group with the necessary inbound and outbound rules(port 22, 80, 443)
- Key Pair, to access the EC2 instance.
- [uv](https://docs.astral.sh/uv/), a fast Python package manager

## Architecture Overview

![image](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*parnLnHjdRlxTT2X)
_**Architectural Diagram for Deployment of DeepSeek-R1 Distill Models (or Open-Source Models) on Amazon EC2 Silicon Based Instances**_

The deployment consists of only one step:

1. Deploying the EC2 instance using Hugging Face Deep Learning AMI and launch Hugging Face TGI

For our purposes we’ll cover one deployment method:

- Deployment via Code (with boto3)

## Deploy via Code (with boto3)

This method is a typical preferred method for deployment of EC2s (besides using cdk, terraform, pulumi or others). The code can be found on this [repository](https://github.com/LokaHQ/deepseek-distilled-on-ec2-inferentia-trainium).

We have seven triggering process steps to trigger the Deployment via Code method.

**Step 1: Set Up Environment**

Clone the repository and install dependencies:

```bash
git clone https://github.com/LokaHQ/deepseek-distilled-on-ec2-inferentia-trainium.git
pip install uv
uv sync
```

**Step 2: Configure AWS Credentials**

```bash
aws configure
```

Or use AWS Vault for secure credential management:

```bash
aws-vault add <profile-name>
```

**Step 3: Set Environment Variables**

Create an .env file and update it with the necessary configurations:

```plaintext
region_info=us-east-1
instance_type=inf2.8xlarge
ami_id=ami-xxxxxxxx
security_group_id=sg-xxxxxxxx
key_name=your-key-pair
endpoint=http://your-ec2-public-dns:8080/generate
subnet_id=subnet-xxxxxxxx
instance_id=i-xxxxxxxx
```

**Step 4: Deploy Model Using Python Script**

Run the deployment script:

```bash
uv run python ./scripts/deploy.py \
    --region <AWS-REGION> \
    --instance_type <INSTANCE-TYPE> \
    --ami_id <AMI-ID> \
    --key_name <KEY-NAME> \
    --security_group_ids <SECURITY-GROUP-IDS> \
    --subnet_id <SUBNET-ID>
```

**Step 5: Run Inference**

Once deployed, you can run inference using the following script:

```bash
uv run python ./scripts/inference.py \
    --endpoint_url <ENDPOINT-URL> \
    --prompt "<PROMPT>" \
    --temperature <TEMPERATURE> \
    --max_tokens <MAX-TOKENS> \
    --top_p <TOP-P>
```

**Step 6: Benchmarking the Model (Optional)**

If you want to evaluate the deployed model by calculating some statistics about the costs and the latency you can execute the following script:

```bash
uv run python ./benchmark/benchmark.py \
    --endpoint <ENDPOINT-URL> \
    --output_dir <OUTPUT-DIR> \
    --temperature <TEMPERATURE> \
    --max_tokens <MAX-TOKENS> \
    --top_p <TOP-P> \
    --max_retries <MAX-RETRIES> \
    --cold_start_loops <COLD-START-LOOPS> \
    --stat_loops <STAT-LOOPS>
```

**Step 7: Removing the Model**

After you experiment with your model, it is advisable to remove it (delete it) from the custom model imports. You can do that by running the following script:

```bash
uv run python ./scripts/delete.py — region <AWS-REGION> — instance_id <INSTANCE-ID>
```

## Best Practices

To ensure a smooth deployment and optimal performance of your DeepSeek-R1 model on SageMaker AI, consider these best practices:

- Inf2 instances are available in 13 AWS regions across 4 continents: Asia Pacific: Mumbai (ap-south-1), Tokyo (ap-northeast-1), Singapore (ap-southeast-1), Sydney (ap-southeast-2).Europe: Stockholm (eu-north-1), Paris (eu-west-3), London (eu-west-2), Ireland (eu-west-1), Frankfurt (eu-central-1). North America: US East (N. Virginia) (us-east-1), US East (Ohio) (us-east-2), US West (Oregon) (us-west-2). South America: São Paulo (sa-east-1).
- Choose your instance type based on the model variant: 8B Version: Suitable for inf2.8xlarge, 70B Version: Requires a larger instance, for example, inf2.48xlarge.
- Each Inferentia2 chip has 2 NeuronCores-v2. The DeepSeek-R1-Distill-Llama-8B model, running on an inf2.8xlarge instance, requires only one NeuronCore (/dev/neuron0). The 70B model, running on an inf2.48xlarge, utilizes 12 NeuronCores, aligning with the script configuration.
- The current deploy.py script deploys the DeepSeek-R1-Distill-Llama-8B model. To deploy the 70B variant, update the user data parameters accordingly.

```bash
user_data = """#!/bin/bash
docker run -p 8080:80 \
-v $(pwd)/data:/data \
 - device=/dev/neuron0 \
 - device=/dev/neuron1 \
 - device=/dev/neuron2 \
 - device=/dev/neuron3 \
 - device=/dev/neuron4 \
 - device=/dev/neuron5 \
 - device=/dev/neuron6 \
 - device=/dev/neuron7 \
 - device=/dev/neuron8 \
 - device=/dev/neuron9 \
 - device=/dev/neuron10 \
 - device=/dev/neuron11 \
-e HF_BATCH_SIZE=4 \
-e HF_SEQUENCE_LENGTH=4096 \
-e HF_AUTO_CAST_TYPE="bf16" \
-e HF_NUM_CORES=24 \
ghcr.io/huggingface/neuronx-tgi:latest \
 - model-id deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
 - max-batch-size 4 \
 - max-total-tokens 4096
"""
```

Deploying distilled DeepSeek-R1 models on EC2 with Inferentia or Trainium offers an affordable, high-performance alternative to SageMaker. While it requires more manual setup, it provides greater control and efficiency for AI inference workloads. In Part 4, we’ll analyze the cost benefits of different deployment methods and discuss business considerations when choosing AWS Bedrock, SageMaker, or EC2 for AI workloads.