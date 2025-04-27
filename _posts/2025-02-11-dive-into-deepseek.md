---
title: "Dive into DeepSeek"
description: "How the open-recipe LLM is transforming GenAI"
author: shekswess
date: 2025-02-11 00:00:00 +0800
categories: [AI, LLM]
tags: [DeepSeek, DeepSeek-R1, Generative AI, Large Language Models, LLMs, AI, Machine Learning, Open Source]
image: https://cdn.prod.website-files.com/649228879157da0fb68c3bee/67ab9db15a2cb13713a99e1b_Deepseek%20Blog.png
---

_**Original Source of the blog post: [Dive into DeepSeek](https://www.loka.com/blog/dive-into-deepseek)**_

_**Blog was written with my colleague from Loka: Crhistian Cardona**_

Original Post: https://www.loka.com/blog/dive-into-deepseek

Just when the world had settled into a post-ChatGPT reality along comes DeepSeek, upending everything we thought we knew, sending stock markets plunging and pundits hyperventilating. The Chinese-made Large Language Model is the latest wave of paradigm-shifting AI, each arriving at what seems like ever-shorter intervals. 

Does DeepSeek matter? Yes. Will it matter in six months? Possibly. It is indeed important—and important to understand right now—but not for the reasons you might think. 

## GenAI and LLM 101
To set the stage, let’s review the basic principles behind Generative AI (GenAI) and Large Language Models (LLMs). GenAI models create new content based on input, usually text but also images and videos. LLMs use transformers, a model type that analyzes word relationships through self-attention and multi-head attention—the latter which allows the LLM to retain memory of its previous inputs and outputs—to generate natural text at any scale.

In these models, words are converted into embeddings that position similar words close together in a mathematical space. Each word and subword is treated as a token, and the model’s context window determines how much text it can consider at one time.

LLMs predict the next word based on learned patterns, and their output can be fine-tuned using parameters such as temperature (for creativity), top-p (for probability filtering), and top-k (for word selection limitation). With millions or even billions of parameters capturing vast amounts of knowledge, these models can generate coherent, human-like responses. The result is the kind of popular LLMs like ChatGPT and Claude that can answer complex questions and create astonishing, even lifelike visuals, all with a simple verbal prompt. 


![image](https://cdn.prod.website-files.com/649228879157da0fb68c3bee/67ab36bd5667cd8c1858930d_AD_4nXcBxh_rDLZH7b0l11HmxhRxDkl4XoAflU1b94g_5qpRy_Jub-bDkcVztsHZ29MHK3g6pksWWfh8z10lWZ0q7edLFZVXaetx6iyPh0x4q9uhez2HVcpKHTU_gnD_lKfykIuUS_8Llg.png)
_**Diagram sourced via poloclub.github**_
‍
## ‍Timeline of Advancements, aka the Hits Keep Coming
Since their inception just 18 months ago Claude and ChatGPT have been iterated and improved multiple times. One fulcrum for their advancement is a machine learning technique called mixture of experts (MoE). Imagine a big problem being solved by a team in which each member is really good at one thing. The team leader picks the best person for each task, making the whole team work more efficiently and effectively. With MoE, the LLM works similarly by having different "experts" for specific tasks.

Then there’s multi-head latent attention (MHLA). Think of MHLA like the model paying attention to different parts of the information it gets, all at the same time. This simultaneous dividing of focus helps the model understand and respond more quickly and accurately. In addition, multi-token prediction (MTP) further improves efficiency by allowing the model to predict several words at once rather than one at a time.

Another crucial advancement is group relative policy pptimization (GRPO). Imagine you're learning to play a new game, and instead of practicing alone, you join a group where everyone shares tips and strategies. By learning together, you all improve faster and make better decisions during the game. 

Finally, model distillation brings the power of stronger larger (teacher) models to smaller (student) models while significantly reducing computational costs, automatically making distilled models faster while requiring less hardware and resulting in better performance.

All of these advancements have arrived over the course of a single year. The culmination is DeepSeek-R1. 

You may have heard that R1 is open source, which is true to an extent; in the realm of AI the term is more nuanced than in traditional software development. R1 is actually open recipe, meaning the methods and weights by which DeepSeek created the model are available to the public but not the source code and the data. (Fully open-source models include SmolLM models from Hugging Face in which the code, data and weights are all available to the public.)

## And Now, DeepSeek-R1
DeepSeek-R1 employs a MoE design with 671 billion parameters, 37 billion of which are active during inference, enabling dynamic allocation of computational resources. Additionally, its impressive 128k-token context window, FP8 precision and use of multi-head latent attention and multi-token prediction collectively enhance its reasoning capabilities. These advancements let DeepSeek-R1 process and understand far more detailed information at once, so it can deliver smarter, more context-aware and naturally human-like responses than older models.

A notable distinction in its development is the use of reinforcement learning (RL) over traditional supervised fine-tuning (SFT). This approach allows the model to self-correct and refine its outputs without constant human supervision, leading to more robust performance. In fact this particular approach is the sole reason why DeepSeek-R1 is important at this moment: It’s the recipe for creating better reasoning models. DeepSeek points toward the future of AI.

DeepSeek-R1 is a cost-effective model because it employs many innovative techniques that lower the cost of inference. For the sake of comparison, OpenAI’s o1 model, which is similar on benchmarks (see the image below), costs $26.3 per 1 million tokens compared to DeepSeek- R1, which costs $3 per 1 million tokens. 

Finally, distilled models represent a significant advancement in AI training, enabling smaller models to inherit capabilities from larger, more powerful ones. In the case of DeepSeek-R1, the full model was used to curate 800,000 high-quality samples, effectively enhancing the performance of more compact models—six in this instance.

To achieve this efficiency, the team leveraged a range of models, including Qwen 2.5 Standard and Math (available in 1.5 billion, 7 billion, 14 billion and 32 billion parameters) and Llama-3.3 Standard and Instruct (8 billion and 70 billion parameters). Across all cases, the balance between model size and performance demonstrated the effectiveness of distillation.

This approach not only makes AI more accessible but also optimizes cost and efficiency, proving that smaller, well-trained models can perform exceptionally well without the need for massive infrastructure.

![image](https://cdn.prod.website-files.com/649228879157da0fb68c3bee/67abbfee95eab0b5c1f6f3b2_Image%201%20(1).png)
_**Price vs Speed vs Quality comparison of DeepSeek-R1 and other models**_

## ‍Yes, But
‍DeepSeek-R1 is not the ChatGPT killer it has been hyped as primarily because of significant data security and privacy concerns. Reports reveal that DeepSeek AI stores user data including conversations and uploaded files on servers in China, raising fears of unauthorized access and surveillance under strict local cybersecurity laws. As a result, US lawmakers have introduced policy to ban the technology and other governments such as Italy have restricted its use on official devices.

Deploying DeepSeek-R1 on trusted cloud platforms like Amazon Web Services (AWS) solves these issues. Hosting on AWS enables organizations to store data regionally, comply with local regulations and benefit from robust security measures like encryption and certifications.

Additionally, DeepSeek-R1 adheres to content guidelines imposed by Chinese authorities, resulting in censorship of sensitive topics like Tiananmen Square and Taiwan. This limitation challenges users seeking unbiased information and underscores the need for transparency in AI development. Balancing advanced AI capabilities with data integrity and information freedom remains crucial.…

## ‍And What Comes Next?
‍As we've seen, advancements in generative AI are evolving rapidly. DeepSeek marks a turning point, shifting LLM training and inference from high-cost, heavy-parameter models to lighter, more efficient alternatives. Thanks to the open-recipe approach new competitors have adopted this method, leading to the rapid emergence of smaller, more affordable models—and making AI tools increasingly accessible.

Meanwhile, platforms like AWS with services such as Amazon Bedrock and Amazon SageMaker support these advancements by enabling seamless deployment, fine-tuning and production use of a wide array of AI models. Their frameworks integrate best practices in security, data privacy and compliance while incorporating techniques like retrieval-augmented generation (RAG) to help businesses scale AI solutions efficiently.

Loka’s engineers believe that integrating new models into scalable, cost-effective pipelines can be the deciding factor between successfully launching transformative products and failing to launch at all. In our upcoming blog series, we’ll explore deploying DeepSeek-R1 distilled versions on AWS, analyzing its size, performance, cost and business applications. Our goal is to help the community push boundaries even further and deliver smarter, more affordable AI solutions.