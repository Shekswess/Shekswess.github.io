---
title: Small Language Model RAGs is All You Need
description: A comprehensive analysis of Retrieval-Augmented Generation (RAG) systems based on different language models, with a particular focus on the performance of small language models compared to their larger counterparts.
author: shekswess
date: 2024-08-29 14:10:00 +0800
categories: [AI, SLM]
tags: [LLM, RAG, LangChain, Langfuse, Ragas, Llama 3.1, Mistral, Mixtral, Gemma, Gemma2, Claude 3.5 Sonnet, Claude 3.5 Opus, Claude 3.5 Haiku, GPT-4o, GPT-4o Mini, GPT-4 Turbo]
image: /assets/images/slm_rag_blog/small_language_model_rags_clip.jpeg
---

## Introduction

In the current AI landscape, the quest for efficient and effective language models has led researchers and practitioners to explore various approaches. One such promising technique is Retrieval-Augmented Generation (RAG), which combines the power of language models with external knowledge retrieval. We are going to do a comprehensive analysis of RAG systems based on different language models, with a particular focus on the performance of smaller models compared to their larger counterparts.

As AI continues to permeate various industries and applications, there’s a growing need for systems that can deliver high-quality, faithful, and relevant responses while efficiently utilizing context. Traditional wisdom might suggest that larger language models would invariably outperform smaller ones. However, with our experiments, we are going to show surprising and promising results that challenge this assumption.

We will explore the following topics in our analysis:


- The competitive performance of smaller language models in RAG systems
- The effectiveness of innovative techniques like Mixture RAG pipelines inspired by the Mixture of Agents (MoA) technique
- The importance of model architecture and training strategies over sheer size
- The particular strengths of smaller models in context utilization
- The additional benefits of using smaller language models, including self-hosting capabilities, improved efficiency, and democratization of AI

By examining these aspects, we aim to shed light on the potential of smaller language models and sophisticated RAG approaches to deliver powerful AI capabilities while offering greater flexibility, control, and cost-effectiveness.

## Setup

In this section, we will provide an overview of the dataset used for the experiments and the RAG pipeline setup. In total there were 28 experiments conducted in total.

### Dataset

The dataset used for the experiments is a collection of research papers in the field of Natural Language Processing (NLP), specifically focusing on Large Language Models (LLMs). The dataset consists of the 15 most cited papers in the field of NLP and LLMs. The papers included in the dataset are:

1. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
2. **The Claude 3 Model Family: Opus, Sonnet, Haiku**
3. **Gemma: Open Models Based on Gemini Research and Technology**
4. **Gemma 2: Improving Open Language Models at a Practical Size**
5. **Improving Language Understanding by Generative Pre-Training**
6. **Language Models are Few-Shot Learners**
7. **GPT-4 Technical Report**
8. **GPT-4o System Card**
9. **LLaMA: Open and Efficient Foundation Language Models**
10. **Llama 2: Open Foundation and Fine-Tuned Chat Models**
11. **The Llama 3 Herd of Models**
12. **Mistral 7B**
13. **Mixtral of Experts**
14. **Mixture-of-Agents Enhances Large Language Model Capabilities**
15. **Attention Is All You Need**

Questions from these papers were used as the evaluation dataset for the experiments. The questions included in the evaluation dataset are:

1. **"What are the two tasks in BERT?"**
2. **"Does Claude 3 models have vision capabilities?"**
3. **"On what architecture the Gemma model is based on?"**
4. **"What tokenizer is used in the Gemma2 model?"**
5. **"How many stages of training are in the GPT model?"**
6. **"On what architecture the GPT-3 model is based on?"**
7. **"Can the GPT-4 model accept both text and image inputs?"**
8. **"What is GPT-4o model?"**
9. **"What optimizer is used for LLaMA?"**
10. **"What is the difference between the Llama 2 and Llama 2-Chat?"**
11. **"How many stages are there in the development of the Llama 3 model?"**
12. **"What is sliding window attention?"**
13. **"Is Mixtral based on the idea of a mixture of experts?"**
14. **"What is Mixture of Agents?"**
15. **"How can attention be described in the Transformer?"**

The creation of the evaluation dataset (questions) was done by extracting the questions which capture the essence of the papers. The questions were selected based on their relevance to the content of the papers and their potential to evaluate the performance of the RAG systems. The questions were designed to test the faithfulness, answer relevancy, and context utilization of the RAG systems, providing a comprehensive evaluation of their capabilities.

### RAG Pipeline Setup

All the experimental pipelines share these common components:
- **Chunker**: The dataset is chunked into smaller parts to be used for the experiments. The chunk size is 1500 and the chunk overlap is 100.
- **Embedder**: The Amazon Titan Embed Text model is used to embed the chunks of the dataset, with 512 vector dimensions which are normalized.
- **Vector Store**: The embedded vectors are stored in a FAISS vector database for faster retrieval.
- **Retriever**: The retrieval of the most similar chunks is done using the FAISS vector database. The number of similar chunks retrieved is 5 and the search type is similarity.

The experimental pipelines differ in the LLMs used and the way of the LLMs are used/combined.

The LLM used in the pipelines are:
- **gemma2-9b-it** - Small LLM
- **gemma-7b-it** - Small LLM
- **mistral-7b-instruct** - Small LLM
- **mixtral-8x7b-instruct** - Small LLM
- **llama-3-8b-instruct** - Small LLM
- **llama-3.1-8b-instruct** - Small LLM
- **llama-3-70b-instruct** - Large LLM
- **llama-3.1-70b-instruct** - Large LLM
- **llama-3.1-405b-instruct** - Large LLM
- **claude-3-haiku** - Small LLM
- **claude-3-sonnet** - Large LLM
- **claude-3-opus** - Large LLM
- **claude-3-5-sonnet** - Large LLM
- **gpt-4o** - Large LLM
- **gpt-4o-mini** - Large LLM
- **gpt-4-turbo** - Large LLM


> The LLMs used in the experiments are from different providers like AWS Bedrock, Groq, and OpenAI. All the experiments were done on 27th and 28th of September 2024, so the results may vary if the experiments are run on different dates, because the versions of the models may change if they are trained again on newer data.
{: .prompt-warning }

Each of the LLMs have specific instruction prompt templates that are used for the experiments. Those templates can be found on:
- [Prompts Engineering Guide](https://www.promptingguide.ai/)
- [Ollama](https://ollama.ai/)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)

All of the LLMs that are used in the RAG pipelines have the same parameters:

- **temperature**: 0
  - In short, the lower the temperature, the more deterministic the results in the sense that the highest probable next token is always picked. Increasing temperature could lead to more randomness, which encourages more diverse or creative outputs. You are essentially increasing the weights of the other possible tokens. In terms of application, you might want to use a lower temperature value for tasks like fact-based QA to encourage more factual and concise responses. For poem generation or other creative tasks, it might be beneficial to increase the temperature value.

- **max_tokens**: 4096
  - This parameter sets the maximum number of tokens that the model can generate in a single response. It ensures that the output does not exceed a certain length, which is useful for controlling the verbosity of the responses.

- **top_p**: 0
  - A sampling technique with temperature, called nucleus sampling, where you can control how deterministic the model is. If you are looking for exact and factual answers keep this low. If you are looking for more diverse responses, increase to a higher value. If you use Top P it means that only the tokens comprising the top_p probability mass are considered for responses, so a low top_p value selects the most confident responses. This means that a high top_p value will enable the model to look at more possible words, including less likely ones, leading to more diverse outputs.

We utilized two different RAG pipeline configurations for the experiments:
- **Simple RAG Pipeline**: Uses a single LLM to generate the responses.
- **Mixture RAG Pipeline**: Uses multiple LLMs to generate the responses, which are then aggregated by another LLM.

Let's dive into the details of each pipeline configuration.

#### Simple RAG Pipeline

The Simple RAG pipeline uses a single LLM to generate the responses. This is how the Simple RAG looks:

![image](../assets/images/slm_rag_blog/simple.png)
_**Simple RAG Pipeline**_

For all the experiments the system and the user messages are the same:

```
system_message:
    "You are an assistant that has a lot of knowledge about Large Language Models.
    Answer the user's question in a way that is easy to understand and informative.
    Use the provided context to generate a response that is relevant and accurate.
    You are an assistant that has a lot of knowledge about Large Language Models.
    Answer the user's question in a way that is easy to understand and informative.
    Use the provided context to generate a response that is relevant and accurate."

user_message: "Please answer my question based on the provided context:"
```

#### Mixture RAG Pipeline

The Mixture RAG pipeline is mostly is like the Simple RAG pipeline, but in the Generator we basically trigger multiple LLMs (Simple RAGs with the same prompt system and user messages previously defined) to generate the responses, and those responses are the aggregated by another LLM. This is how the Mixture RAG looks:

![image](../assets/images/slm_rag_blog/mixture.png)
_**Mixture RAG Pipeline**_


There are three different system and user message combinations used for the experiments, for the aggregation LLM:

- One combination is really similar to the one used in the Mixture of Agents (MoA) implementation:

```
system_message: 
    You have been provided with a set of responses from various small language models to the latest user query. 
    Your task is to synthesize these responses into a single, high-quality response. 
    It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. 
    Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply. 
    Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
    
user_message: "Please synthesize the responses from the small language models and give me only the most accurate information."
```

- The second combination is a bit modified from the first one:

```
system_message: 
    You have been provided with a set of responses from various small language models to the latest user query. 
    The responses of the small language models are based on the context provided in the user query.
    Your task is to create a single, high-quality response based on the responses of the small language models.
    You should perform something like a ensemble model based on majority voting.
    Your response should be very accurate and informative, while keeping the faithfulness and relevance to the previous responses.

user_message: "Please generate a single response based on the provided responses:"
```

- The third combination is basically making the aggregator LLM to choose the best response from the generated responses (thought):

```
system_message: 
    You have been provided with a set of responses from various small language models to the latest user query. 
    The responses of the small language models are based on the context provided in the user query.
    Your task is to choose the best response from the provided responses.
    You should choose the response by analyzing all available responses and selecting the one you think is the most accurate and informative.
    Keep in mind the response must be a high-quality response, while getting the most faithful and relevant information from the provided responses.
    When you have made your choice, make that your final response and do not provide any additional responses, like explanations or clarifications why you chose that response.

user_message: "Please choose a single response based on the provided responses:"
```

## Methodology

All of the results are based on the evaluation of the experiments using the evaluation dataset. In this section, we will present the results of the experiments and analyze the performance of the RAG systems based on different language models. The evaluation metrics used for the analysis are faithfulness, answer relevancy, and context utilization. For calculating the metrics, the judge evaluator LLM and Embedder are used to generate the ground truth answers and to calculate the scores.

The evaluation was done using Ragas. Ragas is a framework that helps to evaluate  Retrieval Augmented Generation (RAG) pipelines. There are existing tools and frameworks that help you build these pipelines but evaluating it and quantifying your pipeline performance can be hard. This is where Ragas (RAG Assessment) comes in.

### Metrics

The metrics used for the evaluation of the experiments are:

- **Faithfulness**: This measures the factual consistency of the generated answer against the given context. It is calculated from answer and retrieved context. The answer is scaled to (0,1) range. Higher the better. The generated answer is regarded as faithful if all the claims made in the answer can be inferred from the given context. To calculate this, a set of claims from the generated answer is first identified. Then each of these claims is cross-checked with the given context to determine if it can be inferred from the context.

<div style="text-align: center;">
  <p>
    <span style="display: inline-block; vertical-align: middle;">
      <img src="https://latex.codecogs.com/svg.latex?\color{white}\text{Faithfulness}=\frac{\text{Number%20of%20claims%20in%20the%20generated%20answer%20that%20can%20be%20inferred%20from%20given%20context}}{\text{Total%20number%20of%20claims%20in%20the%20generated%20answer}}" alt="Faithfulness Formula">
    </span>
  </p>
</div>


- **Answer Relevancy**: The evaluation metric, Answer Relevancy, focuses on assessing how pertinent the generated answer is to the given prompt. A lower score is assigned to answers that are incomplete or contain redundant information, and higher scores indicate better relevancy. This metric is computed using the question, the context and the answer. The Answer Relevancy is defined as the mean cosine similarity of the original question to a number of artificial questions that were generated (reverse engineered) based on the answer:

<div style="text-align: center;">
  <p>
    <span style="display: inline-block; vertical-align: middle;">
      <img src="https://latex.codecogs.com/svg.latex?\color{white}\text{Answer%20Relevancy}%20=%20\frac{1}{N}%20\sum_{i=1}^{N}%20\cos(E_{g_i},%20E_o)" alt="Answer Relevancy Formula 1">
    </span>
  </p>

  <p>
    <span style="display: inline-block; vertical-align: middle;">
      <img src="https://latex.codecogs.com/svg.latex?\color{white}\text{Answer%20Relevancy}%20=%20\frac{1}{N}%20\sum_{i=1}^{N}%20\frac{E_{g_i}%20\cdot%20E_o}{\|E_{g_i}\|\%20\|E_o\|}" alt="Answer Relevancy Formula 2">
    </span>
  </p>
</div>

Where:

<div>
  <p>
    - <span style="display: inline-block; vertical-align: middle;">
      <img src="https://latex.codecogs.com/svg.latex?\color{white}E_{g_i}" alt="E_{g_i}"> is the embedding of the <img src="https://latex.codecogs.com/svg.latex?\color{white}i^{\text{th}}" alt="i^{th}"> artificial question generated from the answer.
    </span>
  </p>

  <p>
    - <span style="display: inline-block; vertical-align: middle;">
      <img src="https://latex.codecogs.com/svg.latex?\color{white}E_o" alt="E_o"> is the embedding of the original question.
    </span>
  </p>

  <p>
    - <span style="display: inline-block; vertical-align: middle;">
      <img src="https://latex.codecogs.com/svg.latex?\color{white}N" alt="N"> is the number of artificial questions generated from the answer.
    </span>
  </p>
</div>

> Even though in practice the score will range between 0 and 1 most of the time, this is not mathematically guaranteed, due to the nature of the cosine similarity, which ranges from -1 to 1.
{: .prompt-info }

- **Context Utilization**: Context utilization measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth. It is computed using question, ground truth and the retrieved context, and the values range between 0 and 1, with higher values indicating better performance. To estimate context utilization from the ground truth answer, each claim in the ground truth answer is analyzed to determine whether it can be attributed to the retrieved context or not. In an ideal scenario, all claims in the ground truth answer should be attributable to the retrieved context. If the ground truth is not provided, the judge evaluator LLM is used to generate the ground truth answer.

<div style="text-align: center;">
  <p>
    <span style="display: inline-block; vertical-align: middle;">
      <img src="https://latex.codecogs.com/svg.latex?\color{white}\text{Context%20Utilization}%20=%20\frac{\text{GT%20claims%20that%20can%20be%20attributed%20to%20context}}{\text{Number%20of%20claims%20in%20GT}}" alt="Context Utilization Formula">
    </span>
  </p>
</div>

### Judge LLM and Embedder

For the Judge LLM Evaluator, we utilized the Claude 3.5 Sonnet model with the model ID `anthropic.claude-3-5-sonnet-20240620-v1:0`. This model was configured with a maximum token limit of 4096 and a temperature setting of 0 to control the randomness of the output. Additionally, we employed the Amazon Titan Embed Text 2 model with the model ID `amazon.titan-embed-text-v2:0`, which operates with 512 dimensions and normalization enabled. 

### Results and Analysis

The initial exploration of the results focused on identifying problematic questions, specifically those with lower scores. The objective was to refine the experiments by excluding these less effective questions and concentrating on the 10 most relevant ones. This approach aims to enhance the overall quality and reliability of the experiments by ensuring that only the most pertinent questions and answers are considered.

To identify these problematic questions, the dataset was grouped by individual questions. For each question, the mean scores were calculated across three key metrics: faithfulness, answer relevancy, and context utilization. These mean scores provided a comprehensive view of each question’s performance. Subsequently, an overall average score was computed for each question by taking the basic average of the mean scores from the three metrics. This overall score was then used to rank the questions, allowing for an informed decision on which questions to exclude from the experiments.

### Monitoring the RAGs and Traces (Tracebility & Observability)

The experiments were monitored using the Langfuse, which provides detailed traces and metrics for each experiment. Langfuse is an open-source LLM engineering platform that helps teams collaboratively debug, analyze, and iterate on their LLM applications.

The traces include information about the execution of the RAG pipelines, the interactions with the LLMs, and the performance of the experiments. The traces are stored in the Langfuse Server and can be accessed through the Langfuse API. The traces provide valuable insights into the behavior of the RAG systems and help in identifying potential issues or areas for improvement. Additionally, the traces can be used to analyze the performance of the RAG systems and evaluate the quality of the generated responses. To each of the trace we attach the scores for the faithfulness, answer relevancy, and context utilization metrics, calculated using the Judge LLM and Embedder.

#### Questions with the lowest scores

<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>Question</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>How many stages are there in the development of the Llama 3 model?</td><td>0.932122</td></tr>
    <tr><td>2</td><td>Does Claude 3 models have vision capabilities?</td><td>0.929947</td></tr>
    <tr><td>3</td><td>What is the GPT-4o model?</td><td>0.881287</td></tr>
    <tr><td>4</td><td>On what architecture the Gemma model is based on?</td><td>0.856614</td></tr>
    <tr><td>5</td><td>How many stages of training are in the GPT model?</td><td>0.851652</td></tr>
    <tr><td>6</td><td>What is the difference between the Llama 2 and Llama 2-Chat?</td><td>0.840406</td></tr>
    <tr><td>7</td><td>Is Mixtral based on the idea of a mixture of experts?</td><td>0.838570</td></tr>
    <tr><td>8</td><td>Can the GPT-4 model accept both text and image inputs?</td><td>0.823327</td></tr>
    <tr><td>9</td><td>What tokenizer is used in the Gemma2 model?</td><td>0.782223</td></tr>
    <tr><td>10</td><td>What is Mixture of Agents?</td><td>0.767041</td></tr>
    <tr><td>11</td><td>What are the two tasks in BERT?</td><td>0.763584</td></tr>
    <tr><td>12</td><td>What is sliding window attention?</td><td>0.748468</td></tr>
    <tr><td>13</td><td>How can attention be described in the Transformer?</td><td>0.728414</td></tr>  
    <tr><td>14</td><td>What is optimizer is used for LLaMA?</td><td>0.614411</td></tr>
    <tr><td>15</td><td>On what architecture the GPT-3 model is based on?</td><td>0.570796</td></tr>
  </tbody>
</table>

From the table, we can observe which questions have the lowest scores. Specifically, the last five questions exhibit the lowest performance and are therefore excluded from the subsequent analysis. This exclusion helps to focus the analysis on the more reliable and relevant questions, ensuring that the results are not skewed by outliers or less effective queries.

The next step involves a detailed analysis of the results for each experiment. This analysis includes ranking the experiments based on the average scores for each metric: faithfulness, answer relevancy, and context utilization. For clarity and comprehensiveness, the top 14 experiments for each metric are highlighted and presented below. Additionally, an overall ranking is conducted by calculating the average of the average scores across all metrics. This comprehensive ranking provides a holistic view of the experiments' performance, facilitating a more informed evaluation and comparison.

#### Faithfulness

<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>Experiment</th>
      <th>Faithfulness</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>simple-rag-gpt-4o</td><td>0.950794</td></tr>
    <tr><td>2</td><td>simple-rag-llama-3.1-405b-instruct</td><td>0.935850</td></tr>
    <tr><td>3</td><td>simple-rag-llama-3.1-8b-instruct</td><td>0.913799</td></tr>
    <tr><td>4</td><td>simple-rag-llama-3.1-70b-instructed</td><td>0.913709</td></tr>
    <tr><td>5</td><td>simple-rag-mistral-7b-instruct</td><td>0.905000</td></tr>
    <tr><td>6</td><td>simple-rag-gemma-7b-it</td><td>0.902381</td></tr>
    <tr><td>7</td><td>simple-rag-gpt-4o-mini</td><td>0.872143</td></tr>
    <tr><td>8</td><td>simple-rag-llama-3-70b-instruct</td><td>0.869946</td></tr>
    <tr><td>9</td><td>mixture-rag-mixtral-8x7-instruct-modified</td><td>0.868546</td></tr>
    <tr><td>10</td><td>mixture-rag-llama3.1-8b-instruct-thought</td><td>0.866354</td></tr>
    <tr><td>11</td><td>simple-rag-llama-3-8b-instruct</td><td>0.862557</td></tr>
    <tr><td>12</td><td>simple-rag-mixtral-8x7b-instruct</td><td>0.862047</td></tr>
    <tr><td>13</td><td>simple-rag-claude-3-opus</td><td>0.861019</td></tr>
    <tr><td>14</td><td>simple-rag-gpt-4-turbo</td><td>0.860575</td></tr>
    <tr><td>15</td><td>simple-rag-claude-3-sonnet</td><td>0.950794</td></tr>
  </tbody>
</table>

The table above ranks various experiments based on their faithfulness scores, which measure how accurately the generated responses adhere to the source information. Based on the results from the table, it is evident that the scores of the RAG systems based on smaller language models are very close to, or in some cases even better than, those based on larger language models. For instance, in the top 10 scores, we have 5 RAG systems that are based on smaller language models: `simple-rag-llama-3.1-8b-instruct`, `simple-rag-mistral-7b-instruct`, `simple-rag-gemma-7b-it`, `mixture-rag-mixtral-8x7-instruct-modified`- which is a combination of multiple smaller language models and `mixture-rag-llama3.1-8b-instruct-thought`- also a combination of multiple smaller language models with specific prompt. These smaller models achieve faithfulness scores of 0.913799, 0.905000, 0.902381, 0.868546 and 0.866354 respectively, which are comparable to or even surpass the scores of some larger models. 

This observation suggests that smaller language models can perform nearly as well as, or sometimes better than, larger models in terms of faithfulness. The close scores among the top experiments indicate that model architecture and training strategies play a significant role in achieving high faithfulness, regardless of the model size. This insight is valuable for guiding future improvements and optimizations in model development, as it highlights the potential of smaller models to deliver high-quality results, results that are faithful to the context and source information provided.

#### Answer Relevancy

<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>Experiment</th>
      <th>Answer Relevancy</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>simple-rag-mistral-7b-instruct</td><td>0.903208</td></tr>
    <tr><td>2</td><td>simple-rag-gpt-4o-mini</td><td>0.902027</td></tr>
    <tr><td>3</td><td>simple-rag-gemma2-9b-it</td><td>0.898397</td></tr>
    <tr><td>4</td><td>simple-rag-llama-3.1-8b-instruct</td><td>0.889998</td></tr>
    <tr><td>5</td><td>simple-rag-claude-3.5-sonnet</td><td>0.887503</td></tr>
    <tr><td>6</td><td>mixture-rag-gemma2-9b-it-thought</td><td>0.880448</td></tr>
    <tr><td>7</td><td>mixture-rag-gemma2-9b-it-modified</td><td>0.875354</td></tr>
    <tr><td>8</td><td>simple-rag-mixtral-8x7b-instruct</td><td>0.871510</td></tr>
    <tr><td>9</td><td>simple-rag-claude-3-opus</td><td>0.869271</td></tr>
    <tr><td>10</td><td>simple-rag-claude-3-sonnet</td><td>0.868577</td></tr>
    <tr><td>11</td><td>mixture-rag-mixtral-8x7-instruct-thought</td><td>0.868344</td></tr>
    <tr><td>12</td><td>simple-rag-gpt-4o</td><td>0.868135</td></tr>
    <tr><td>13</td><td>simple-rag-gpt-4-turbo</td><td>0.866888</td></tr>
    <tr><td>14</td><td>simple-rag-claude-3-haiku</td><td>0.863664</td></tr>
    <tr><td>15</td><td>simple-rag-gemma-7b-it</td><td>0.863156</td></tr>
  </tbody>
</table>

The table above ranks various experiments based on their answer relevancy scores, which measure the relevance of the generated responses to the given prompts. The results show that in the top 10 experiments, 6 of them are again based on smaller language models with simple rag pipeline approach or with the smart technique of mixture rag pipeline approach. The experiments `simple-rag-mistral-7b-instruct`, `simple-rag-gemma2-9b-it`, `simple-rag-llama-3.1-8b-instruct`, `mixture-rag-gemma2-9b-it-thought`, `mixture-rag-gemma2-9b-it-modified` and `simple-rag-mixtral-8x7b-instruct` have really high answer relevancy scores of 0.903208, 0.898397, 0.889998, 0.880448, 0.875354 and 0.871510 respectively. 

This again indicates that smaller language models can generate highly relevant responses that are closely aligned with the given prompts. We can even see that the Mixture RAG pipeline approach with the smart technique of choosing the best response from the generated responses(thought) can achieve high answer relevancy scores. 


#### Context Utilization

<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>Experiment</th>
      <th>Context Utilization</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>mixture-rag-claude-3-haiku-thought</td><td>0.933333</td></tr>
    <tr><td>2</td><td>simple-rag-mistral-7b-instruct</td><td>0.925000</td></tr>
    <tr><td>3</td><td>mixture-rag-claude-3-haiku</td><td>0.916667</td></tr>
    <tr><td>4</td><td>mixture-rag-gemma2-9b-it-modified</td><td>0.916667</td></tr>
    <tr><td>5</td><td>mixture-rag-llama3.1-8b-instruct-modified</td><td>0.908333</td></tr>
    <tr><td>6</td><td>simple-rag-gpt-4o</td><td>0.905556</td></tr>
    <tr><td>7</td><td>mixture-rag-mixtral-8x7-instruct-thought</td><td>0.900000</td></tr>
    <tr><td>8</td><td>mixture-rag-llama3.1-8b-instruct-thought</td><td>0.897222</td></tr>
    <tr><td>9</td><td>simple-rag-llama-3-70b-instruct</td><td>0.897222</td></tr>
    <tr><td>10</td><td>mixture-rag-gemma2-9b-it-thought</td><td>0.897222</td></tr>
    <tr><td>11</td><td>simple-rag-gemma-7b-it</td><td>0.897222</td></tr>
    <tr><td>12</td><td>simple-rag-llama-3.1-70b-instructed</td><td>0.897222</td></tr>
    <tr><td>13</td><td>simple-rag-claude-3-haiku</td><td>0.894444</td></tr>
    <tr><td>14</td><td>simple-rag-gpt-4o-mini</td><td>0.888889</td></tr>
    <tr><td>15</td><td>simple-rag-gpt-4-turbo</td><td>0.888889</td></tr>
  </tbody>
</table>

The table above ranks various experiments based on their context utilization scores, which measure how effectively the retrieved context aligns with the annotated answers. Here really we can see how RAG systems based on smaller language models are performing really well in terms of context utilization. From the best 10 experiments, 6 of them are based on smaller language models. Another interesting thing is that Mixture RAG approaches are excellent in context utilization, with 2 of the top 5 experiments being based on the Mixture RAG approach. The experiments `mixture-rag-llama3.1-8b-instruct`, `mixture-rag-mixtral-8x7-instruct-modified`, and `mixture-rag-mixtral-8x7-instruct` have context utilization scores of 0.916667, 0.916667, and 0.913889 respectively.

#### Average of the Scores

<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>Experiment</th>
      <th>Average Score</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>simple-rag-mistral-7b-instruct</td><td>0.911069</td></tr>
    <tr><td>2</td><td>simple-rag-gpt-4o</td><td>0.908161</td></tr>
    <tr><td>3</td><td>simple-rag-llama-3.1-8b-instruct</td><td>0.890155</td></tr>
    <tr><td>4</td><td>simple-rag-gpt-4o-mini</td><td>0.887686</td></tr>
    <tr><td>5</td><td>simple-rag-gemma-7b-it</td><td>0.887586</td></tr>
    <tr><td>6</td><td>simple-rag-llama-3.1-70b-instruct</td><td>0.880033</td></tr>
    <tr><td>7</td><td>simple-rag-llama-3-70b-instruct</td><td>0.876048</td></tr>
    <tr><td>8</td><td>simple-rag-gemma2-9b-it</td><td>0.874335</td></tr>
    <tr><td>9</td><td>simple-rag-mixtral-8x7b-instruct</td><td>0.872297</td></tr>
    <tr><td>10</td><td>simple-rag-gpt-4-turbo</td><td>0.872117</td></tr>
    <tr><td>11</td><td>simple-rag-llama-3.1-405b-instruct</td><td>0.871293</td></tr>
    <tr><td>12</td><td>mixture-rag-gemma2-9b-it-modified</td><td>0.869955</td></tr>
    <tr><td>13</td><td>mixture-rag-llama3.1-8b-instruct-thought</td><td>0.869177</td></tr>
    <tr><td>14</td><td>simple-rag-claude-3-haiku-</td><td>0.860552</td></tr>
    <tr><td>15</td><td>mixture-rag-gemma2-9b-it-thought</td><td>0.860197</td></tr>
  </tbody>
</table>

The table above ranks various experiments based on their average scores, which provide a comprehensive view of the experiments' performance across all metrics. The results show the dominance of RAG systems based on smaller language models, with 9 of the top 15 experiments being based on smaller models.


### Conclusion

The analysis of various experiments comparing RAG (Retrieval-Augmented Generation) systems based on different language models yields several significant insights:

- Smaller language models perform competitively when used in RAG system, often achieving scores comparable to or even surpassing those of larger language models based RAG systems across multiple metrics (faithfulness, answer relevancy, and context utilization).

- The mixture RAG pipeline where the generator of the RAG system is inspired by the implementation of Mixture of Agents(MoA) technique like choosing the best response from generated output options, shows strong performance across metrics.

- The close scores among top experiments suggest that factors such as model architecture and training strategies may be more crucial than model size in achieving high-quality results.

- Smaller models and mixture RAG approaches demonstrate particular effectiveness in context utilization, indicating their ability to align retrieved information with annotated answers.

- Overall when considering average scores across all metrics, RAG systems based on smaller language models dominate the top rankings, occupying 9 out of the top 14 positions.

These findings highlight the potential of smaller language models and sophisticated RAG approaches to deliver high-quality, faithful, and relevant responses while efficiently utilizing context. 

Moreover, we do not need to work the additional benefits of the smaller language models, such as:

- Self-hosting and open-source capabilities: Smaller models are more amenable to self-hosting, allowing organizations to maintain control over their data and infrastructure. Many of these models are also open-source, fostering transparency, community-driven improvements, and customization.

- Improved efficiency and reduced costs: Smaller models require less computational resources, leading to lower energy consumption and reduced operational costs. This efficiency makes them more environmentally friendly and economically viable for a broader range of applications.

- Democratization of AI: The ability to run these models on less powerful hardware democratizes access to advanced AI capabilities. This allows individuals, small businesses, and organizations with limited resources to create and deploy sophisticated RAG systems, fostering innovation across diverse sectors.

- Faster inference times: Smaller models typically offer quicker response times, which is crucial for real-time applications and enhancing user experience in interactive systems.

- Privacy and compliance advantages: Self-hosted smaller models can help organizations better comply with data protection regulations and maintain stricter control over sensitive information.

- Flexibility and adaptability: Smaller models are often easier to fine-tune or adapt to specific domains or tasks, allowing for more tailored solutions without the need for extensive computational resources.

- Ability to run on edge devices: Smaller models can be deployed on edge devices, enabling AI applications to operate locally without relying on cloud services. This capability is essential for scenarios where low latency, privacy, or limited network connectivity are critical.

These insights and benefits could guide future developments in language model applications, potentially leading to more resource-efficient, accessible, and equally effective AI systems. By leveraging smaller language models in RAG systems, organizations and individuals can harness powerful AI capabilities while enjoying greater flexibility, control, and cost-effectiveness.

## Additional Resources

- [RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [What are Small Language Models with Examples](https://aisera.com/blog/small-language-models/)
- [Mixture of Agents Enhances Large Language Model Capabilities](https://arxiv.org/abs/2406.04692)
- [Code Repository for the Experiments](https://github.com/Shekswess/small-language-model-rags-is-all-you-need)
