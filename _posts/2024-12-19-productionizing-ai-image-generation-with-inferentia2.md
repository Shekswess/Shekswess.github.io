---
title: "Productionizing AI Image Generation with Inferentia2"
description: "AWS’s new chip unlocks cost savings and scalability in Stable Diffusion."
author: shekswess
date: 2024-12-19 00:00:00 +0800
categories: [AI, Inferetia2]
tags: [Stable Diffusion, Inferentia2, AWS, Generative AI]
image: https://slack-imgs.com/?c=1&o1=ro&url=https%3A%2F%2Fcdn.prod.website-files.com%2F649228879157da0fb68c3bee%2F67632269d401d3a75f49f12a_ai-image-generation.png
---

_**Original Source of the blog post: [Productionizing AI Image Generation with Inferentia2](https://www.loka.com/blog/productionizing-ai-image-generation-with-inferentia2)**_

_**Blog was written with my colleagues from Loka: Zafir Stojanovski & Henrique Silva**_


In recent years, diffusion models have emerged as groundbreaking architectures in the field of generative AI, driving huge advancements in tasks like image synthesis, inpainting, text-to-image generation and image-to-image generation. At the forefront is Stable Diffusion, an open-source model that has democratized access to state-of-the-art generative capabilities, enabling researchers, artists and developers to create high-quality images based on textual prompts. 

Unlike traditional generative adversarial networks (GANs), diffusion models work by modeling the step-by-step denoising of data, learning to reverse a stochastic process that transforms data (i.e. images) into pure noise. This iterative nature provides them with unique advantages such as stable training dynamics and the ability to produce highly diverse and coherent outputs. Their applications span industries, from generating marketing visuals to aiding video game design and beyond. 

However, deploying diffusion models in production is far from trivial. By their very nature, diffusion models involve iterative sampling processes—a single output image can require hundreds or thousands of forward passes through a neural network. This makes inference significantly slower compared to other generative models like GANs. Additionally, these models often have millions if not billions of parameters, resulting in substantial compute and memory requirements that can strain even the most advanced GPUs.

Engineers and technical leaders hoping to optimize inference have to get creative with real-time applications such as interactive design tools and personalized content generation. Techniques like model distillation, reduced-step sampling algorithms, unique noise schedulers and quantization can accelerate sampling while maintaining image fidelity. Scaling these models across distributed systems adds another layer of complexity, as engineers must balance memory footprints, computational throughput and latency constraints to deliver seamless user experiences.

Despite these challenges, diffusion models remain popular. Pre-trained diffusion models are widely available on Hugging Face, with their “diffusers” python package, simplifying experimentation and application development. But achieving high-throughput, low-latency deployment remains a significant hurdle, requiring specialized knowledge in systems engineering and model optimization. 

Here at Loka, we’ve observed a growing demand among our clients for using cutting-edge generative AI architectures like Stable Diffusion while keeping operational costs as low as possible. Many clients are exploring innovative use cases such as generating images with specific effects or specific theme-based images, but they require a cost-effective solution for deploying and scaling these models in production. AWS has addressed this need by introducing Inferentia2 chips, purpose-built for inference workloads, which offer a significant reduction in cost while maintaining competitive performance.

We deployed a Stable Diffusion XL 1.0-based model to generate interior design images for one of our clients. The end goal was to create high-quality, modern interior design concepts at the lowest possible cost. To identify the most efficient deployment strategy, we conducted benchmarking between AWS EC2 instances powered by Inferentia2 chips and traditional NVIDIA GPU-based instances. From this process we discovered a powerful insight: AWS’s new Inferentia2 chips unlock inference that’s up to 17.5% less expensive than normal. 

## The Next Generation of Inference
AWS Inferentia2 is the next iteration of the AWS Silicon chips made for inference. When Inferentia1 debuted in 2019, Amazon EC2 Inf1 instances achieved remarkable performance gains, offering 25% higher throughput and 70% lower costs than G5 instances powered by NVIDIA A10G GPUs. Now, with Inferentia2, AWS raises the bar once again.

The Inferentia2 chip delivers a 4x boost in throughput and 10x lower latency compared to the first generation. Similarly, EC2 Inf2 instances deliver up to 2.6x better throughput, 8.1x lower latency, and 50% greater performance per watt than G5 instances. This makes Inferentia2 an outstanding choice for inference workloads, providing an ideal balance of cost-efficiency (high throughput) and low latency for real-time applications.

Inf2 instances come in a variety of sizes, featuring between 1 and 12 Inferentia2 chips, interconnected via ultra-fast direct links to handle distributed inference for different kinds of large model architectures. Best of all, thanks to the deep integration with PyTorch, developers can easily leverage the strong performance of compiling models with the AWS Neuron SDK without significant added complexity.

Speaking of leveraging the power of Inferentia2, let’s look at the benchmark setup for this use case.

## Benchmarking for Efficiency
To evaluate the performance and cost-efficiency of deploying the fine-tuned version of Stable Diffusion XL 1.0 on AWS Cloud, we conducted a comprehensive benchmark comparing two deployment scenarios: one leveraging the AWS Inferentia2-powered inf2.8xlarge instance and the other using the NVIDIA GPU-based g5.8xlarge  instance. For each instance type, we compiled the Stable Diffusion XL model using tools optimized for their respective hardware architectures. 

For the inf2.8xlarge instance, we used the diffusers-compatible implementation of Stable Diffusion XL, optimized with the Torch XLA compiler. On the g5.8xlarge  instance, the model was optimized using torch.compile, a relatively new feature in PyTorch 2.0 that leverages Just-In-Time (JIT) compilation to optimize PyTorch code for improved performance, per the recommendations of the official PyTorch guide. These tools ensured that the models were optimized for their respective platforms in order to obtain the most accurate benchmarking results. 

To reiterate, the goal of the benchmark has been to assess both the computational efficiency and the operational cost-effectiveness of these setups under real-world inference conditions. The evaluation process itself involved creating a diverse set of 25 text prompts, each representing a distinct user case or input scenario for the model. To ensure consistent results and account for any initialization overhead, we warmed up the models by executing each prompt five times before starting the actual measurements. Each prompt was then executed 50 times on both instances, providing a robust dataset comparing performance metrics such as mean execution time, standard deviation, various target percentiles and more. The results of these 50 runs were averaged for each prompt, giving us a clear view of per-prompt performance across the two instances. 

![image](https://cdn.prod.website-files.com/649228879157da0fb68c3bee/6763231ef1d6d81b55caf70f_AD_4nXc74Hf4p87cM6XPQ8r2GSmgkgMf_nSvzH1uKlhK2TdrREpunx5rBa6nTWml49KjDR4bcyNv6Ej2cDAl6iQL9IDECFkM8cURRPr04hitGU5_iYJEBQryWlONP17VRhaplMdMOu47ig.png)
_**Figure 1. (Left) Cost comparison between the NVIDIA and AWS Silicon instances across different workloads. (Right) Execution time for the same workloads.**_
‍

## The Bottom Line
To quantify the operational implications of these results, we calculated the average execution time per prompt for each instance type and extrapolated this data to estimate the total time and cost for handling varying workloads. This structured approach ensured that our benchmarking not only captured the raw performance capabilities of the hardware but also provided actionable insights into their cost-effectiveness for real-world applications. 

From the empirical analysis in Figure 1, we have concluded the following: while the total execution time on the inf2.8xlarge instance is merely slightly larger compared to the g5.8xlarge instance (as shown in the right plot), the total cost of running the workload is remarkably lower (as shown in the left plot). This cost reduction is attributed to the more affordable hourly pricing of the Inferentia2-powered instance. At large workloads of processing 10,000,000 images, with the Inferentia-powered instance we are looking at a reduction in cost of up to $14,353 compared to the NVIDIA-powered instance. Therefore, for workloads requiring substantial compute resources, the inf2.8xlarge instance demonstrates clear cost-efficiency advantages without compromising the feasibility of completing tasks within reasonable timeframes. 

![image](https://cdn.prod.website-files.com/649228879157da0fb68c3bee/6763231eff93d66233e75b68_AD_4nXcAwosuVPF-SmtmCcKokcZ26_8-E0pSQfFKUFcGoVk8xbSgeLFY8fr-Y83j5WzTXRorgA3rv1FCZSllWx3pa4M1nhKAB5PUfF3v6n97Id_9B8RlWb0CZ_IEie7HVDp8Wf4aHo7b.png)
_**Figure 2. (Left) Average cost (in cents) per generated image between the NVIDIA and AWS Silicon instances. (Right) Average time (in seconds) per image generation.**_
‍

To dive deeper into the time and cost comparison at a more granular level, let us take a look at Figure 2. We can observe that even though the inf2.8xlarge instance generates an image merely 0.31 seconds slower compared to the g5.8xlarge instance (right figure), the average cost per image is remarkably 17.5% lower due to the favorable pricing of the Inferentia-based instances (left figure). 

![image](https://cdn.prod.website-files.com/649228879157da0fb68c3bee/6763231e7e780feb36e566ab_AD_4nXdRKgcW_fuduZ4Z3gR0oejKW7eE9PuPWzNxCvASpqnHGKYocf1Wqzza_hjct2QG8LXkRa287nCnCGjVirtnix_9eXyJ7dRJ2xGxExFUplaGtmotNbiaa3nRbrpOUlbwlNlcnxOs.png)
_**Figure 3. Prompt: “Art collector's loft with gallery walls, track lighting, polished concrete floors.” Left g5.8xlarge: Right: inf2.8xlarge.**_


![image](https://cdn.prod.website-files.com/649228879157da0fb68c3bee/6763231e6777adb8c2e3de14_AD_4nXfFMJhcg4XVoGtauHbap7uRXPqfROcOddBei4eEt-D6i5NlPkEDFet4j3F3pnvUOwUQHYnklouv3n8IPqqVa_zct-bPKFqYISPmMk2tsVIgQzw2rBVEsqb2mLpSylxwayZlOl5vkA.png)
_**Figure 4. Prompt: “Scandinavian living space with rattan furniture, sheepskin throws, dried pampas grass, soft winter light, interior design magazine.” Left g5.8xlarge: Right: inf2.8xlarge.**_

![image](https://cdn.prod.website-files.com/649228879157da0fb68c3bee/6763231ed4449cc4b5f071e8_AD_4nXd6mCqWqDKKumQLkVdBXreXhME_euqZ8QwlqXXjPB-D9h6iagOW2vlCRTYEXWErIt1nKW7uMbXnSCG9_kOVj5xbdJiWvfE3dUgiGsIYxZ93d0aMjXcCgfV1t0TrOstYfqmY0c8lyA.png)
_**Figure 5. Prompt: “Contemporary farmhouse kitchen with industrial elements, subway tiles, copper accents, morning light.” Left g5.8xlarge: Right: inf2.8xlarge.**_
‍

Generative AI models like Stable Diffusion have revolutionized creative workflows across industries, but their computational demands pose significant challenges for cost-effective deployment at scale. AWS Inferentia2 chips are purpose-built for inference workloads, offering a compelling solution by balancing affordability with high performance. 

Our benchmarking analysis highlighted the cost advantages of using the Inferentia2-powered inf2.8xlarge instance for deploying Stable Diffusion XL. The execution time was only slightly longer compared to the g5.8xlarge instance with NVIDIA GPUs but the reduced per-hour cost resulted in major savings for large-scale workloads. 

Our evaluation demonstrates the importance of specialized hardware like Inferentia2 for solving problems of performance and cost in AI inference workloads, offering a scalable and affordable solution for businesses using generative AI. Maximizing resources is crucial for startups and enterprise operations alike, and Loka can help anyone seeking the latest techniques for fast and cheap inference at scale.