---
layout: post
title: Flexible AI Project Template
description: A comprehensive yet flexible AI project template that addresses the challenges of project organization in AI, ML, MLOps, and LLMOps.
author: shekswess
date: 2024-10-03 14:10:00 +0800
categories: [AI, uv]
tags: [uv, GitHub Actions, Docker Compose, pre-commit, pytest, MLflow, Langfuse, Jupyter Notebook]
image: /assets/images/ai_template_blog/flexible-ai-project-template.jpeg
---

## Introduction

In the rapidly evolving landscape of AI, ML, MLOps, and LLMOps, having a well-structured project template is not just a convenience, it's a necessity ! 

As a Machine Learning Engineer who has navigated the complexities of various projects, I found existing cookiecutter templates often overwhelming and sometimes lacking in crucial areas. On the other hand, having a project without any structure can lead to chaos and inefficiency.

This realization led me to develop a comprehensive yet flexible AI project template that addresses these challenges head-on.

## The Problem with Existing Templates

Many engineers, myself included, have struggled with project organization. Existing templates often suffer from several issues:

- Overcomplexity: They include unnecessary components that clutter the project structure.
- Rigidity: Many are too specific to certain types of AI projects, lacking flexibility.
- Outdated Tools: Some templates don't keep pace with the rapidly evolving AI tooling ecosystem.
- Lack of Best Practices: Important aspects like code quality checks, testing frameworks, or deployment configurations are often missing.

## The Solution: A Versatile AI Project Template

My AI project template aims to solve these issues by providing a clean, modular structure that accommodates various AI project types while incorporating modern best practices and tools. 

### Structure Overview

This is a high-level overview of the template structure:

```
.
├── config                    # Configuration files for your project
│   └── config.yaml           # Example configuration file
├── data                      # Folder to store raw and processed data
│   ├── database              # Databases or local data storage
│   ├── processed             # Preprocessed/cleaned data
│   └── raw                   # Raw data inputs
├── iac                       # Infrastructure as Code (IaC) scripts for cloud deployment
├── notebooks                 # Jupyter notebooks for exploratory data analysis, experiments
│   └── 00_example.ipynb      # Example notebook
├── results                   # Folder to store the results of experiments and models
├── src                       # Source code of your project
│   ├── constants             # Constants used in the project
│   ├── models                # Machine learning model scripts
│   ├── pipelines             # ML pipelines for preprocessing and modeling
│   ├── utils                 # Utility functions
│   └── execution.py          # Main execution script
├── tests                     # Unit and integration tests
│   └── test_example.py       # Example test file using pytest
├── .env                      # Environment variables file
├── .gitignore                # Standard .gitignore file
├── .pre-commit-config.yaml   # Configuration for pre-commit hooks
├── docker-compose.yaml       # Docker Compose setup for MLflow, Langfuse, and related services
├── pyproject.toml            # Configuration for formatting, linting, type-checking, and testing
├── README.md                 # Documentation for the project (you're reading it!)
└── uv.lock                   # Lock file for uv package manager
```

Let's dive into some key components and the rationale behind them:

#### 1. Directory Structure

The template includes directories for `config`, `data`, `iac`, `notebooks`, `results`, `src`, and `tests`. This structure isn't revolutionary, but it's purposefully designed to be intuitive and scalable, with having the option to include or exclude certain directories based on project needs. I will shortly explain the rationale behind some of these directories:

- **Why `config` directory?**

This directory is essential for storing configuration files that might change between environments or experiments. It promotes a separation of concerns and makes it easier to manage different configurations.

- **Why to have a `data` directory which is separated into `raw`, `processed`, and `database` directories?**

This separation enforces good data management practices. It maintains a clear distinction between original data (raw), transformed data ready for modeling (processed), and any database files for more complex data storage needs.

- **Why include an `iac` directory?**

Infrastructure as Code (IaC) is crucial for reproducible deployments. Including this directory from the start encourages thinking about deployment early in the project lifecycle and ensures that deployment scripts are version-controlled, because most of the AI projects today are deployed on cloud platforms. In this directory you can include scripts of Terraform, CloudFormation, Pulumi, CDK or any other IaC tool you prefer.

- **Why have a `notebooks` directory?**

Jupyter notebooks are a common tool for data exploration and experimentation. By having a dedicated directory for notebooks, it's easier to organize and share findings. The numbered prefix system encourages a logical flow of analysis and experimentation.

- **Why have a `results` directory?**

Even though the results of experiments can be stored in MLflow or other experiment tracking tools, having a dedicated directory for results makes it easier to access and share outputs. It also provides a backup in case the experiment tracking system fails.

- **Why include a `src` directory which is further divided into `constants`, `models`, `pipelines`, and `utils`?**

This `src` directory structure promotes modularity and maintainability. By having separate directories for constants, models, pipelines, and utility functions, it's easier to navigate the codebase and make changes without affecting other components.

- **Why include a `tests` directory?**

Testing is crucial for maintaining code quality and ensuring that changes don't introduce regressions. By including a dedicated tests directory, it encourages writing tests from the start and makes it easier to run tests using pytest.

#### 2. Modern Package Management with uv

Every Python based project (and most of the AI projects are Python based) needs a package manager and project management tool, that's why I chose `uv` for this template. 

I chose uv as the package manager for several reasons:

- Speed: uv is significantly faster than traditional tools like pip or conda, even though my initial option for this project was conda - in combination with pip.
- Reproducibility: It generates lock files `uv.lock` ensuring consistent environments across different machines.
- Simplicity: uv simplifies the management of virtual environments and package installations by utilizing of a `pyproject.toml` file.

#### 3. Configuration Management

The template uses a combination of `pyproject.toml`, `.env`, and a `config` directory:

- `pyproject.toml`: This file centralizes project metadata, dependencies (for uv), and tool configurations (like ruff for linting and formatting). It's becoming the standard for Python project configuration.
- `.env`: For environment-specific variables, promoting security by keeping sensitive information out of version control.
- `config` directory: For more complex configurations that might change between environments or experiments.

#### 4. Integrated DevOps

The template includes several DevOps tools and configurations to streamline development:

- GitHub Actions: GitHub Action workflows are essential for automating CI/CD pipelines. In this template, the idea is to utilize them for testing purposes and deployment automation.
- Docker Compose: Docker Compose is a powerful tool for managing multi-container applications. The template includes a `docker-compose.yaml` file for setting up services like MLflow and Langfuse for local development.
- Pre-commit Hooks: Pre-commit hooks are essential for maintaining code quality. The template includes a `.pre-commit-config.yaml` file with hooks for formatting, linting, type-checking, and more.


#### 5. Notebooks with Purpose

As we said in the directory structure, the `notebooks` directory is dedicated to Jupyter notebooks. These notebooks are not just for exploration—they serve a specific purpose in the project lifecycle:

- **Exploratory Data Analysis (EDA):** Initial data exploration and visualization to understand the dataset.
- **Experimentation:** Iterative model training and evaluation to find the best approach.
- **Documentation:** Detailed explanations of models, results, and insights for sharing with stakeholders.

#### 6. Experiment & Results Tracking

In addition to what we mentioned in the directory structure, the template includes MLflow for experiment tracking and model versioning. MLflow is a popular tool for managing the ML lifecycle, and it's integrated into the template to provide a seamless experience for tracking experiments and results. Also, Langfuse is included for monitoring, observing, tracing and managing Large Language Models (LLMs) or LLM based applications in the project.

## Practical Applications

This template shines in various AI scenarios:

- Machine Learning Projects: The structure supports the entire ML lifecycle, from data preprocessing to model deployment.
- MLOps Initiatives: With integrated tools like MLflow, it's ready for experiment tracking and model versioning out of the box.
- LLM Applications: The inclusion of Langfuse demonstrates its readiness for Large Language Model operations and monitoring.
- Research Projects: The notebook structure and results directory make it suitable for AI research and experimentation.

## Conclusion

This AI Project Template is more than just a directory structure—it's a thoughtfully crafted foundation that embodies best practices in AI development. By providing a flexible yet opinionated setup, it allows developers to focus on what matters most: **solving AI challenges**.

Whether you're building a complex MLOps pipeline, experimenting with cutting-edge LLMs, or conducting AI research, this template offers a solid starting point. It's designed to grow with your project, providing the scaffolding needed for both small experiments and large-scale AI applications.

By using this template, you're not just organizing files—you're adopting a methodology that promotes reproducibility, maintainability, and scalability in AI development.

In the future this template will be updated with more features and integrations to keep up with the latest trends in AI and ML.

Ready to revolutionize your AI development process? 
Check out the AI Project Template on GitHub and start your next project with a solid foundation!

## Additional Resources

- [AI Project Template on GitHub](https://github.com/Shekswess/ai-project-template)