---
title: Make JavaScript Great Again
description: Can JavaScript be great again?
author: shekswess
date: 2024-11-16 00:00:00 +0800
categories: [AI, JavaScript]
tags: [AI, deno, transformers.js, tensorflow.js,  Jupyter Notebook]
image: /assets/images/javascript_blog/javascript.jpeg
---

## JavaScript's Downfall

From what we been able to see from this year [GitHub's octoverse report](https://github.blog/news-insights/octoverse/octoverse-2024/#the-most-popular-programming-languages), Javascript's 10-year reign as the most popular programming language has come to an end. The reasons for this are many, but the most important is the rise of AI, for which Python rains supreme. 

![GitHub's octoverse report](https://github.blog/wp-content/uploads/2024/10/GitHub-Octoverse-2024-top-programming-languages.png?w=1000)
_**Source:** [GitHub's octoverse report](https://github.blog/news-insights/octoverse/octoverse-2024/#the-most-popular-programming-languages)_

Even though both JavaScript and Python are general-purpose & beginner-friendly programming languages, Python has been the go-to language for AI and data science. This is because Python has a rich ecosystem of libraries and frameworks that make it easy to build AI models and AI-powered applications. 

From my point of view, this is the main reason why JavaScript has lost its 'crown' to Python. It was just a matter of time before this happened, especially with the AI boom in the last few years.

**`Why Python is the go-to language for AI and data science ?`** is the question that we will need to answer in order to see how JavaScript can become great again.

The answer to this question is really simple: 

Python has a rich ecosystem of libraries and frameworks that make it easy to build AI models and AI-powered applications. Some of the most popular libraries and frameworks for AI and data science in Python are `TensorFlow`, `PyTorch`, `Transformers`, `scikit-learn`, `pandas`, `NumPy`, and `Matplotlib`. These libraries and frameworks provide a wide range of tools and functionalities that make it easy to build AI models and AI-powered applications. Other than that, experimentation with Python is extremely easy with Jupyter Notebook. The utilization of high computational power (GPUs) in Python is simple as including 1-2 lines of code. 

So, it is no wonder that Python is the go-to language for AI and data science. To make JavaScript great again, we need to build a rich ecosystem of libraries and frameworks that make it easy to build AI models and AI-powered applications.

## How to Make JavaScript Great Again

There were some pushes in the past to make packages like TensorFlow.js, NLP.js, Brain.js or ML5.js. But, these packages are not as popular as their Python counterparts. This is because they are not as mature as their Python counterparts. They lack the functionalities and tools that make it easy to build AI models and AI-powered applications. They also lack the community support that makes it easy to get help and support when building AI models and AI-powered applications. 

To make JavaScript great again, we need to build a rich ecosystem of libraries and frameworks that make it easy to build AI models and AI-powered applications. We also need to build a strong community that provides help and support when building AI models and AI-powered applications. Also the execution of JavaScript on the server side for training models is not as easy as Python. There isn't a simple way to utilize GPU power for training models in JavaScript. This is a big disadvantage for JavaScript. But not everything is lost for JavaScript. I can see big potential in two packages which can be essential for making JavaScript great again, these are:
- [Deno 2](https://deno.com/)
- [Transformers.js](https://huggingface.co/docs/transformers.js/en/index)


### Deno 2

`Deno 2` is the next iteration of Deno, a secure and modern runtime for JavaScript and TypeScript. Deno 2 improves on the first iteration of Deno for a lot of things. Firstly it improves the performance dramatically by doing some performance optimizations for cold-start times and memory management. Also the updated runtime is leaner, which translates to faster application start-up and more efficient resource utilization—great for serverless environments and large-scale apps which is really important for AI based apps. Other than that it has a lot of better native TypeScript support, it continues to use standardized Web APIs and the amazing security enhancements by providing a lot of higher level control over what it can be access with the code like file systems, network, environment variables, and more. But the two most important features are that Deno 2 has not a seamless integration with npm ecosystem of packages and the killer feature is that it has build a kernel for running JavaScript/TypeScript on Jupyter Notebook. This is a game changer for JavaScript.

Deno ships with a built-in Jupyter kernel that allows you to write JavaScript and TypeScript; use Web and Deno APIs and import npm packages straight in your interactive notebooks. 

To install the Deno 2 Jupyter kernel, you first need to have [Deno installed on your machine](https://docs.deno.com/runtime/getting_started/installation/). Once you have Deno 2 installed, you can install the [Deno 2 Jupyter kernel](https://docs.deno.com/runtime/reference/cli/jupyter/) very easily.

![Image](https://github.com/denoland/deno-docs/assets/836375/32f0ccc3-35f7-47e5-84f4-17c20a5b5732)
_**Running Deno 2 Jupyter Kernel**_

![Image](https://docs.deno.com/runtime/reference/images/jupyter-cli.gif)
_**Running Deno 2 Jupyter Kernel in CLI**_

This is a game changer for JavaScript because it allows you to quickly prototype and experiment, which on the other hand is really important for developing AI models and AI-powered applications. The usage of Jupyter Notebooks with Python is one of the main reasons why Python is the go-to language for AI and data science. With Deno 2, JavaScript can now compete with Python in this area.

### Transformers.js

The next package that can make JavaScript great again is `Transformers.js`. Transformers.js is a JavaScript library that allows you to run Hugging Face’s Transformers directly in your browser, with no need for a server. Transformers.js is designed to be functionally equivalent to Hugging Face’s Transformers Python library, meaning you can run the same pretrained models using a very similar API. These models support common tasks in different modalities, such as:

- **Natural Language Processing**: text classification, named entity recognition, question answering, language modeling, summarization, translation, multiple choice, and text generation.
- **Computer Vision**: image classification, object detection, segmentation, and depth estimation.
- **Audio**: automatic speech recognition, audio classification, and text-to-speech.
- **Multimodal**: embeddings, zero-shot audio classification, zero-shot image classification, and zero-shot object detection.

Transformers.js uses ONNX Runtime to run models in the browser. The best part about it, is that you can easily convert pretrained PyTorch, TensorFlow, or JAX models to ONNX using Optimum so it can be run in the browser.

By default models runned in the browser with Transformers.js use the CPU (via WASM), however there is the option to use the GPU on the machine on which you operate with using  WebGPU. This is an another game changer for JavaScript, because it allows you to utilize the GPU power for training or running AI models. 

## Can JavaScript be great again?

Absolutelly, JavaScript can be great again!

With the help of Deno 2 and Transformers.js, JavaScript can start to compete with Python in the AI and data science area. The Jupyter Notebook kernel for JavaScript provided by Deno 2, and the ability to run Hugging Face’s Transformers directly in the browser with Transformers.js, are big steps in the right direction. However there is still a lot of work to be done. JavaScript needs to build a rich ecosystem of libraries and frameworks that make it easy to build AI models and AI-powered applications. It also needs to build a strong community that provides help and support when building AI models and AI-powered applications.

Nothing is loss for JavaScript, it just needs to adapt to the new reality of AI and data science and so it can solve some new problems and challenges which it previously solved for web development.
