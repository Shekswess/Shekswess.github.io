---
title: "Part 1: Deploying Distilled DeepSeek-R1 Models on Amazon Bedrock" 
description: "Harnessing Open Source AI on AWS"
author: shekswess
date: 2025-02-11 00:00:00 +0800
categories: [AI, LLM]
tags: [DeepSeek, DeepSeek-R1, Generative AI, Large Language Models, LLMs, AI, Machine Learning, Open Source]
image: https://miro.medium.com/v2/resize:fit:2800/format:webp/1*ZipoYVSKO1CqVk8_bI2SYg.png
---

_**Original Source of the blog post: [Part 1: Deploying Distilled DeepSeek-R1 Models on Amazon Bedrock](https://medium.com/loka-engineering/harnessing-open-source-ai-on-aws-2f9b0fd9c42d)**_

_**Blog was written with my colleague from Loka: Crhistian Cardona**_


Advancements in generative AI are evolving as we speak. As we noted in the introduction to this series, AWS supports emerging GenAI innovations through features such as [Amazon Bedrock](https://aws.amazon.com/bedrock/), a fully managed service for building and scaling GenAI applications with a variety of foundation models.

Loka’s engineers prefer Bedrock because it offers several advantages for deploying generative AI models. First, it provides a fully managed, serverless environment that allows developers to focus on building applications without worrying about infrastructure management. Second, by leveraging AWS’s cloud infrastructure, Bedrock enables efficient scaling of AI models to meet varying demand levels, ensuring applications remain responsive under different workloads without incurring unnecessary costs. Bedrock also integrates robust security measures, ensuring data privacy and compliance. And perhaps most important of all, it enables seamless integration with other AWS services.

From here on out, the rest of this series is aimed at a technical audience of Software Engineers, Solution Architects, ML Engineers, MLOps Engineers and of course AI Engineers. We’ll walk through the technical practical steps required to deploy a distilled version of DeepSeek-R1 — or any other open-source model — on AWS Bedrock as a custom model import. We’ll also explain the reasons why we prefer one model over another.

Our goal is to walk you through a spectrum of deployment strategies for DeepSeek-R1 distilled models — or any other open-source model — on AWS. We’ll explore a range of approaches designed to meet different needs: Amazon Bedrock’s user‑friendly, serverless interface that’s perfect for both technical and non‑technical (click‑ops) teams; the more hands‑on, customizable environment provided by Amazon SageMaker for those who want deeper control over their deployments; and finally the ultra‑optimized performance on Amazon EC2 Inferentia for the most advanced, performance‑driven use cases.

No matter where you fall on the technical spectrum, this series aims to have something for everyone, empowering you to choose the deployment path that best fits for your use case.

## Prerequisites

Before getting started, you should have the following:

- [AWS Account](https://aws.amazon.com/account/) with permissions for Bedrock and S3
- [AWS CLI](https://aws.amazon.com/cli/) installed and configured
- [IAM Role](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html) with access to S3 and Bedrock services
- [S3 Bucket](https://aws.amazon.com/s3/) to store model files
- [uv](https://docs.astral.sh/uv/), a fast Python package manager
- [git-lfs](https://git-lfs.com/), which is required for handling large model files

## Architecture Overview

The deployment consists of three main steps:

1. Download the model from [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B).
2. Upload model artifacts to an S3 bucket.
3. Deploy the model as a custom model in Bedrock.

![Image](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*rEHqfnCLVXtLn2m5)
_**Architectural Diagram for Deployment of DeepSeek-R1 Distill Models (or Open-Source Models) on AWS Bedrock**_

This piece will cover two different deployment methods:

1. Deployment via AWS Management Console
2. Deployment via Code (with boto3)

### Deployment Methods

#### Method 1: Deploy via AWS Management Console

This method is ideal for users who prefer a click-and-configure approach using the AWS UI.

**Step 1: Download the model.**

Visit [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) and download the necessary model files (for example, DeepSeek-R1-Distill-Llama-70B).


![Image](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*pvYd1kas5bR-wb6U)
_**Hugging Face Model Card of DeepSeek-R1 Distill Llama 70B Model**_

![Image](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*Sy0g5q6pEShJ7zWL)
_**Hugging Face Files of DeepSeek-R1 Distill Llama 70B Model**_


**Step 2: Upload Model Files to S3.**

1.Log in to AWS and navigate to S3.

![Image](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*XyC2VreOfykTcZp0)
_**AWS Management Console**_

2.Create a new bucket.

![Image](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*sjzDb6XRHD-AhuAO)
_**Amazon S3 Console**_

![Image](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*qPLwNIT_VQswma9m)
_**AWS S3 Console for Creating S3 Bucket**_


3.Upload the downloaded model files.

![Image](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*3hDW0FiHPx0XtLTc)
_**UI for Uploading Files to S3 bucket**_


**Step 3: Create an IAM Role.**

1.Go to IAM in the AWS Console.

![Image](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*hMwBZu1SfSSTWMsd)
_**AWS IAM Console**_

2.Create a new role with permissions for S3 and Bedrock.

![Image](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*NJ48i-APcJG_UBGj)
_**AWS IAM Roles Console**_

![Image](https://miro.medium.com/v2/resize:fit:2000/format:webp/0*BBIUnf3hcKFhpwf7)
_**UI for Creating IAM Role**_

**Step 4: Import Model into Bedrock.**

1.Navigate to Amazon Bedrock.

![Image](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*J5BLKn7MtF8wFtb3)
_**AWS Bedrock Console**_

2.Click “Import Model” and provide the required S3 location and IAM role.

![Image](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*uoWZWEO8a1qLk_vR)
_**AWS Bedrock Console for Importing Models**_

3.Wait for the model to be imported and deployed.

![Image](https://miro.medium.com/v2/resize:fit:2000/format:webp/0*qZClzFoR8O4xdjxe)
_**UI for Creating Imported Model**_

4.Use the result for testing and inference.

![Image](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*vPwml68CIJwDsIN6)
_**AWS Bedrock Playground on which we test the Imported Model**_

5. After testing the model, you can delete it by clicking on the Delete button.

![Image](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*GDlR2dj0emOJly94)
_**UI for Deleting the Imported Model**_

#### Method 2: Deploy via Code (with boto3)

This method is ideal for users who prefer an automated code-oriented deployment by using AWS SDK (boto3) in Python. The code can be found on this repository.

**Step 1: Set up the environment.**

Clone the repository and install dependencies:

```bash
git clone https://github.com/LokaHQ/deepseek-distilled-on-bedrock-custom.git
pip install uv
git-lfs install
uv sync
```

****Step 2: Configure AWS credentials.**

```bash
aws configure
```

Or use AWS Vault for secure credential management:

```bash
aws-vault add <profile-name>
```

**Step 3: Set environment variables.**

Create an .env file and update it with the necessary configurations:

```plaintext
bucket_name=”your-s3-bucket”
s3_prefix="model-prefix"
local_directory="./models"
hf_model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
job_name="deepseek-job"
imported_model_name="deepseek-bedrock-model"
role_arn="your-iam-role-arn"
region_info="us-west-2"
```

**Step 4: Deploy Model Using Python Script.**

Run the deployment script:

```bash
uv run ./scripts/deploy.py \
  -hf_model_id <HF-MODEL-ID> \
  -bucket_name <S3-BUCKET-NAME> \
  -s3_prefix <S3-PREFIX> \
  -local_directory <LOCAL-DIRECTORY> \
  -job_name <CMI-JOB-NAME> \
  -imported_model_name <CMI-MODEL-NAME> \
  -role_arn <IAM-ROLE-ARN> \
  -region_info <AWS-REGION>
```

**Step 5: Run inference.**

Once deployed, you can run inference using the following script:

```bash
uv run ./scripts/inference.py \
  -model_id <MODEL-ID> \
  -hf_model_id <HF-MODEL-ID> \
  -region_info <AWS-REGION> \
  -temperature 0.3 \
  -max_tokens 4096 \
  -top_p 0.9 \
  -max_retries 10 \
  -prompt "<PROMPT>"
```

**Step 6 (Optional): Benchmark the model.**

If you want to evaluate the deployed model by calculating some statistics about the costs and the latency, execute the following script:

```bash
uv run ./benchmark/benchmark.py \
  -model_id <MODEL-ID> \
  -region_info <AWS-REGION> \
  -temperature 0.3 \
  -max_tokens 4096 \
  -top_p 0.9 \
  -max_retries 10 \
  -cold_start_loops 2 \
  -stat_loops 5 \
  -output_dir <OUTPUT-DIR>
```

**Step 7: Remove the Model**

After you experiment with your model, it is advisable to delete it from the custom model imports. To do so, run the following script:

```bash
uv run ./scripts/delete.py \
  -model_id <MODEL-ID> \
  -bucket_name <S3-BUCKET-NAME> \
  -s3_prefix <S3-PREFIX> \
  -region_info <AWS-REGION>
```

## Best Practices

- Ensure your IAM role has correct S3 and Bedrock permissions.
- Use uv for managing dependencies efficiently.
- Adjust parameters like temperature, max_tokens and top_p based on your application needs.
- Optimize S3 storage by removing unused model files after deployment.

## Implementing Guardrails in Amazon Bedrock

Guardrails allow us to implement safeguards across deployed AI models, aligning with responsible AI policies and security restrictions. With Bedrock teams can easily create, test and deploy guardrails during model inference and integrate them with agents and retrieval-augmented generation (RAG) solutions to enhance AI reliability and control. Best practices for implementing guardrails include configuring harmful content categories with strength levels, detecting and blocking prompt attacks through level-based filters and restricting topics based on company policies with predefined sample phrases. Additionally, teams can apply word and profanity filters, automatically detect and remove personally identifiable information (PII) and incorporate contextual grounding and relevance checks to ensure AI-generated responses are accurate and appropriate.

By leveraging these features within Bedrock’s fully managed environment, organizations can enforce AI policies, prevent security risks and build trust in AI applications while maintaining compliance with corporate and regulatory constraints.

Looking ahead, Part 2 will cover deploying DeepSeek-R1 distilled models on Amazon SageMaker, providing insights into model training, fine-tuning, and cost considerations. Stay tuned as we continue our deep dive into AI deployment strategies!