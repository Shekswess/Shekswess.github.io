---
title: "Part 1: Agentic Patterns 101 with Loka & Strands-Agents"
description: "Agentic Patterns 101, with Strands-Agents"
author: shekswess
date: 2025-08-27 00:00:00 +0800
categories: [AI, LLM]
tags:
  [
    Agentic Patterns,
    Agents,
    Agenting AI,
    Generative AI,
    Large Language Models,
    LLMs,
    AI,
    Machine Learning,
    Open Source,
  ]
image: https://miro.medium.com/v2/resize:fit:1100/format:webp/1*6-WL3LSunbhXP-8QNFTD8A.png
---

_**Original Source of the blog post: [Part 1: Agentic Patterns 101 with Loka & Strands-Agents](https://medium.com/loka-engineering/part-1-agentic-patterns-101-with-loka-strands-agents-13f45ea62c70)**_

_**Blog was written with my colleagues from Loka: Nina Cvetkovska & Petar Kalinovski**_

These days, Agents are everywhere in daily conversation and it seems everyone wants one. We read or hear about AI Agents, Agentic AI, Agentic Architectures, and Agentic Patterns almost every day. But what do these terms really mean? How do we design Agentic Systems that are both reliable and powerful? And just as importantly, when are agents the wrong tool for the job?

In this short post series, we’ll explore these questions by examining nine common patterns for building AI-powered systems. Some patterns are rooted in structured workflows, while others embrace flexible, autonomous approaches.

To ground the discussion, we’ll use [Strands-Agents](https://strandsagents.com/latest/), an open-source SDK from AWS, as our reference framework. The main benefit of Strands is that it is model-agnostic, so you can define an agent using any LLM (cloud-based or local), a system prompt, and a set of tools, while the SDK handles reasoning, planning, and the execution loop. This makes it easy to experiment with both workflow-driven and agent-driven architectures, perfectly illustrating where each pattern excels.

### Why Do Agentic Patterns Matter?

Patterns provide structure to system design. They allow AI applications to grow in capability and complexity without becoming brittle and make it easier to adapt as requirements evolve. A pattern-driven, modular design is far easier to maintain and extend than a one-off build.

Patterns also tame the complexity of orchestrating multiple agents, tools, and workflows. Proven templates reduce trial-and-error risk, promote best practices, and establish a shared language among developers.

### When Should (and Shouldn’t) You Use Agents?

Before adding an agent, ask: **Do I really need one here?**

If the problem has a fixed, well-understood solution path, a simple workflow or even a short script will likely be faster, cheaper, and more reliable. Agents introduce flexibility, but that flexibility comes with trade-offs: higher latency, increased compute usage, and potential unpredictability.

Agents shine in messy, dynamic, or ambiguous environments, where adaptability and autonomous reasoning are genuine advantages.
Quick guide:

- **Workflows**: Best for predictability, repeatability, and efficiency.
- **Agents**: Best for open-ended, evolving, or uncertain tasks.

Even when using agents, keep designs as simple as possible. Overly complex setups can be fragile, hard to debug, and costly to maintain. Because agents can make mistakes, strong error handling, detailed logging, and retry mechanisms are essential - giving the system a chance to recover gracefully.

## Pattern Overview

We'll explore nine patterns, with some emphasizing the clarity of structured workflows, while others harnessing the autonomy of agentic design, showing how each fits into the broader landscape of AI system architecture.
We'll cover:

- **Sequential Workflow**
- **Parallel Workflow**
- **LLM Routing**
- **Reflection Pattern**
- Pure Python Tools
- Tools from MCP Servers
- Agents as Tools (Orchestrator-Workers)
- Swarm
- Graph Multi-Agent Pattern

In this part of the blog post series, we'll dive into the first four patterns in detail, leaving the remaining ones for later sections. All of the examples for the patterns can be found on the GitHub repository [here](https://github.com/LokaHQ/agentic-patterns-101).

### [Sequential Workflow](https://github.com/LokaHQ/agentic-patterns-101/blob/main/agentic-patterns/sequential-workflow.py)

![Sequential Workflow](https://cdn-images-1.medium.com/max/720/1*OnJOECYUzdiQHQEBCW-aFw.png)

In a Sequential Workflow (or sometimes called Prompt Chaining), tasks are completed step by step, where each agent's (LLM's) output becomes the input for the next. This pattern divides a larger task into a chain of smaller, well-defined steps, ensuring that each stage builds directly on the previous one. It is ideal for processes that follow a predictable, linear progression without branching or parallelization.

**Characteristics of Sequential Workflow**:

- Fixed, linear sequence of steps
- Each step depends on the output of the previous step
- Minimal branching or deviation from the sequence

**Use Cases & Industries**:

- **Structured document generation**: Agent 1 (LLM 1) creates an outline, Agent 2 (LLM 2) generates content, and Agent 3 (LLM 3) validates the content against predefined criteria - e.g., in Content Creation for blog posts, marketing copy, or reports.
- **Multi-step data processing**: Extract, transform, and summarize information from documents - e.g., in Intelligent Document Processing (IDP) for Healthcare records or Invoices.
- **Research workflow**: Agent 1 (LLM 1) collects research references, Agent 2 (LLM 2) synthesizes insights, Agent 3 (LLM 3) formats a report - e.g., in Academic Research or Market Analysis.

**When to Use / When Not to Use**:

- Use when: Tasks can be broken into discrete, ordered steps with predictable outcomes.
- Avoid when: Flexibility, parallel processing, or adaptive decision-making is required.

**Code Example:**

With strands agents to generate a sequential workflow there are two ways:

- Pythonic way where we define the sequential workflow as python function or class that resembles a pipeline:

```python
"""Example of a sequential workflow using three agents as a python function (or class) resembeling pipeline"""

from strands import Agent
from strands.models import BedrockModel

BLOG_PLANNER_INSTRUCTION = "You are a blog planner. Take the topic that is provided to you and create a detailed outline with 3 sections and key points for each section."
SECTION_WRITER_INSTRUCTION = "You are a blog writer. Take the outline provided by the blog planner and expand each section into a detailed paragraph."
EDITOR_INSTRUCTION = "You are a professional editor. Improve the following blog draft by fixing grammar, making sentences concise, and ensuring smooth flow."

def sequential_workflow(
    blog_planner_agent: Agent,
    section_writer_agent: Agent,
    editor_agent: Agent,
    topic: str,
) -> str:
    """
    A sequential workflow that involves three agents:
    1. Blog planner to create an outline.
    2. Section writer to write the blog post based on the outline.
    3. Editor to edit the blog post.

    Args:
        topic (str): The topic for the blog post.
        blog_planner_agent (Agent): The agent responsible for creating the blog outline.
        section_writer_agent (Agent): The agent responsible for writing the blog post.
        editor_agent (Agent): The agent responsible for editing the blog post.

    Returns:
        str: The final edited blog post.
    """

    print("Starting sequential workflow for topic:", topic)

    # Step 1: Create a blog outline
    outline = blog_planner_agent(topic)
    print("Blog outline created:", outline)

    # Step 2: Write the blog post based on the outline
    draft = section_writer_agent(outline)
    print("Blog draft created:", draft)

    # Step 3: Edit the blog post
    final_post = editor_agent(draft)
    print("Final blog post created:", final_post)

    return final_post

if __name__ == "__main__":
    # Define the model we use for the agents (we want creative response that's why temp is high)
    bedrock_model = BedrockModel(
        model_id="us.amazon.nova-premier-v1:0",
        temperature=0.9,
    )

    # Step 1: Agent that creates a blog outline
    blog_planner_agent = Agent(
        model=bedrock_model,
        system_prompt=BLOG_PLANNER_INSTRUCTION,
        callback_handler=None,  # Optional: You can provide a callback handler for logging or monitoring
    )

    # Step 2: Agent that writes the blog post
    section_writer_agent = Agent(
        model=bedrock_model,
        system_prompt=SECTION_WRITER_INSTRUCTION,
        callback_handler=None,  # Optional: You can provide a callback handler for logging or monitoring
    )

    # Step 3: Agent that edits the blog post
    editor_agent = Agent(
        system_prompt=EDITOR_INSTRUCTION,
        model=bedrock_model,
        callback_handler=None,  # Optional: You can provide a callback handler for logging or monitoring
    )

    # Define the topic for the blog post and create the sequential workflow
    topic = "The Future of AI in Content Creation"
    final_post = sequential_workflow(
        blog_planner_agent=blog_planner_agent,
        section_writer_agent=section_writer_agent,
        editor_agent=editor_agent,
        topic=topic,
    )
    print("Final blog post created:", final_post)
```

- A strands-agent tool way that easily defines sequential workflow with a predefined tool:

```python
"""Example of a sequential workflow using three agents as a workflow tool (strands-agents-tools)"""

from strands import Agent
from strands.models import BedrockModel
from strands_tools import workflow

BLOG_PLANNER_INSTRUCTION = "You are a blog planner. Take the topic that is provided to you and create a detailed outline with 3 sections and key points for each section."
SECTION_WRITER_INSTRUCTION = "You are a blog writer. Take the outline provided by the blog planner and expand each section into a detailed paragraph."
EDITOR_INSTRUCTION = "You are a professional editor. Improve the following blog draft by fixing grammar, making sentences concise, and ensuring smooth flow."

if __name__ == "__main__":
    # Define the model we use for the agents (we want creative response that's why temp is high)
    bedrock_model = BedrockModel(
        model_id="us.amazon.nova-premier-v1:0",
        temperature=0.9,
    )

    # Create one Agent with one workflow tool
    blog_agent = Agent(tools=[workflow])

    # Creating the workflow
    blog_agent.tool.workflow(
        action="create",
        workflow_id="blog_agent_workflow",
        tasks=[
            {
                "task_id": "blog_planner",  # The unique ID for the Blog Planner agent
                "description": 'Create a detailed outline for the blog post about "The Future of AI in Content Creation"',  # The description of the task
                "system_prompt": BLOG_PLANNER_INSTRUCTION,  # The system prompt for the Blog Planner agent
                "priority": 5,  # The priority of the task (higher numbers indicate higher priority)
            },
            {
                "task_id": "section_writer",
                "description": "Expand each section of the outline into a detailed paragraph",
                "dependencies": ["blog_planner"],
                "system_prompt": SECTION_WRITER_INSTRUCTION,
                "priority": 4,
            },
            {
                "task_id": "editor",
                "description": "Edit the blog draft for clarity and conciseness",
                "dependencies": ["section_writer"],
                "system_prompt": EDITOR_INSTRUCTION,
                "priority": 3,
            },
        ],
    )

    # Start the workflow
    blog_agent.tool.workflow(
        action="start",
        workflow_id="blog_agent_workflow",
    )

    # Monitor the workflow progress
    blog_agent.tool.workflow(
        action="monitor",
        workflow_id="blog_agent_workflow",
    )
```

### [Parallel Workflow](https://github.com/LokaHQ/agentic-patterns-101/blob/main/agentic-patterns/parallel-workflow-tool.py)

![Parallel Workflow](https://cdn-images-1.medium.com/max/720/1*QmMRftG8i1bFcd6JeQatDA.png)

In a Parallel Workflow, multiple agents (LLMs) execute tasks simultaneously, often starting from a shared input or from the output of a preceding step. This allows independent subtasks to run in parallel, reducing latency and improving efficiency. The results from parallel agents are typically aggregated or passed to a downstream step for integration.

**Characteristics of Parallel Workflow**:

- Multiple agents operate concurrently
- Steps can process shared or partitioned inputs
- Requires a mechanism to merge or coordinate results

**Use Cases & Industries**:

- **Content enrichment**: Agent 1 (LLM 1) drafts an article, while Agent 2 (LLM 2) in parallel checks grammar/style and Agent 3 (LLM 3) performs fact-checking - e.g., in Media & Publishing workflows for news or blog posts.
- **Multi-perspective analysis**: Agent 1 (LLM 1) structures a financial report, then Agent 2 (LLM 2) analyzes regulatory risks while Agent 3 (LLM 3) in parallel evaluates market sentiment - e.g., in Finance & Compliance domains.

**When to Use / When Not to Use**:

- **Use when**: Subtasks are independent, can safely be run simultaneously, and merging results is straightforward.
- **Avoid when**: Steps are highly dependent on each other's outputs, or when sequencing is critical to correctness.

**Code Example:**

- With strands-agents the preferred way of implementing Parallel Workflow is to use the workflow tool:

```python
"""Example of parallel workflow using three agents as a workflow tool (strands-agents-tools)"""

from strands import Agent
from strands.models import BedrockModel
from strands_tools import workflow

CEO_INSTRUCTION = "You are CEO. Your goal is to ensure the company's success and make high-level decisions."
FINANCE_INSTRUCTION = "You are finance expert. Always analyze the question and solution from a financial perspective."
LEGAL_INSTRUCTION = "You are legal expert. Always analyze the question and solution from a legal perspective."

if __name__ == "__main__":
   # Define the model we use for the agents
   bedrock_model = BedrockModel(
       model_id="us.amazon.nova-premier-v1:0",
       temperature=0.1,
   )

   # Create one Agent with one workflow tool
   solution_agent = Agent(tools=[workflow])

   # Creating the workflow
   solution_agent.tool.workflow(
       action="create",
       workflow_id="solution_agent_workflow",
       tasks=[
           {
               "task_id": "ceo_agent",  # The unique ID for the CEO Agent
               "description": "How to increase the efficiency of the team?",  # The description of the task
               "system_prompt": CEO_INSTRUCTION,  # The system prompt for the CEO Agent
               "priority": 5,  # The priority of the task (higher numbers indicate higher priority)
           },
           {
               "task_id": "finance_agent",
               "description": "Analyze the financial implications of the proposed solutions",
               "dependencies": ["ceo_agent"],
               "system_prompt": FINANCE_INSTRUCTION,
               "priority": 3,
           },
           {
               "task_id": "legal_agent",
               "description": "Ensure all proposed solutions comply with legal regulations",
               "dependencies": ["ceo_agent"],
               "system_prompt": LEGAL_INSTRUCTION,
               "priority": 3,
           },
       ],
   )

   # Start the workflow
   solution_agent.tool.workflow(
       action="start",
       workflow_id="solution_agent_workflow",
   )

   # Monitor the workflow progress
   solution_agent.tool.workflow(
       action="monitor",
       workflow_id="solution_agent_workflow",
   )
```

### [LLM Routing](https://github.com/LokaHQ/agentic-patterns-101/blob/main/agentic-patterns/llm-routing.py)

![LLM Routing](https://cdn-images-1.medium.com/max/720/1*Ny27jKW5vOwgwkT5jWrZ4w.png)

LLM Routing is used to automatically direct queries to the most appropriate model based on complexity, domain expertise, or computational requirements. Rather than using a single expensive, or inadequate model for all tasks, intelligent routing optimizes both cost and performance by matching query complexity and model capability.

Modern AI systems like GPT-5 (In the ChatGPT interface) use routing for automatically choosing between faster models for simple queries and reasoning models for complex problems. With strands-agents, we can implement similar routing patterns explicitly, giving us full control over cost optimization and performance tuning.

**Characteristics of LLM Routing**:

- Dynamic model selection based on query analysis
- Cost optimization through appropriate model sizing
- Performance optimization through complexity matching
- Centralized routing logic with distributed model execution
- Continuous learning from routing decisions and outcomes

**Use Cases & Industries**:

- **Software Development Support**: Route basic syntax questions to lightweight models (AWS Nova Lite) while directing complex architecture challenges to reasoning models (AWS Nova Pro/Claude - e.g., in Developer Tooling platforms, Code Assistants, or Technical Documentation systems.
- **Customer Support Systems**: Simple FAQ routing to fast models, complex troubleshooting to specialized domain experts - e.g., in SaaS platforms, Enterprise Software, or Technical Support workflows.
- **Financial Analysis**: Basic calculations and data retrieval to efficient models, complex risk analysis and regulatory compliance to advanced reasoning models - e.g., in Investment Banking, Risk Management, or Regulatory Reporting.

**When to Use / When Not to Use**:

- **Use when**: Query complexity varies significantly, cost optimization is important, or different models have genuinely different capabilities for the domain.
- **Avoid when**: All queries require similar complexity levels, routing overhead exceeds benefits, or a single model adequately handles the entire use case spectrum.

**Code Example:**

- We will use strands-agents for the router and the chosen LLMs. While using a strands-agent for the routing is not necessary, we will use it because it is one of the easiest ways to get structured output:

```python
"""Example of LLM Routing workflow for handling user queries based on complexity."""

from enum import Enum

from pydantic import BaseModel, Field
from strands import Agent
from strands.models import BedrockModel

ROUTER_INSTRUCTION = """
You are a helpful software development router that can route user queries to the appropriate agent.
Based on the user's question, determine the complexity of the query as:
simple, complex, or ambiguous.

- Simple queries can retrieve factual information but lack reasoning for complex system design.
- Complex queries require deeper understanding and reasoning about the system's architecture and design principles.
- Ambiguous queries are unclear and may require clarification before they can be properly routed.
"""

class RouteType(Enum):
    """
    Enumeration for the different types of routing that can be applied to a user query.
    """

    SIMPLE = "simple"
    COMPLEX = "complex"
    AMBIGUOUS = "ambiguous"

class RouteResponse(BaseModel):
    """
    Config Model representing the response from the routing process.
    """

    type: RouteType
    message: str = Field(..., description="The response message")

if __name__ == "__main__":
    """
    Main function to execute the routing process.
    """
    # User query to be routed
    user_query = "What is the difference between == and === in JavaScript?"

  # Other example queries:
    # complex_query = "Design a microservices architecture for a banking system with regulatory compliance"
    # ambiguous_query = "What is the best way to implement a feature?"

    # Initialize the routing model
    routing_model = BedrockModel(
        model_id="amazon.nova-pro-v1:0",
        temperature=0.6,
    )

    # Initialize the router agent
    router_agent = Agent(model=routing_model, system_prompt=ROUTER_INSTRUCTION)

    # Get structured response from user query
    response = router_agent.structured_output(RouteResponse, user_query)

    # Route the response based on its type

    # Handle simple response
    if response.type == RouteType.SIMPLE:
        print("Routing to simple response handler.")

        agent = Agent(
            model=BedrockModel(
                model_id="amazon.nova-lite-v1:0",
                temperature=0.9,
            ),
            system_prompt="You are a fast, efficient assistant for basic programming questions. Provide concise, accurate answers about syntax, commands, and simple explanations.",
        )

        agent_response = agent(user_query)

        print(f" Final response: {agent_response}")

    # Handle complex response
    elif response.type == RouteType.COMPLEX:
        print("Routing to complex response handler.")

        agent = Agent(
            model=BedrockModel(
                model_id="anthropic.claude-sonnet-4-20250514-v1:0",
                temperature=0.9,
            ),
            system_prompt="You are an expert software architect. Analyze complex system design problems, consider trade-offs, and provide detailed architectural guidance with reasoning.",
        )

        agent_response = agent(user_query)

        print(f" Final response: {agent_response}")

    # Handle ambiguous response
    elif response.type == RouteType.AMBIGUOUS:
        agent = Agent(
            model=BedrockModel(
                model_id="amazon.nova-pro-v1:0",
                temperature=0.9,
            ),
            system_prompt="You are an ambiguous query handler agent.",
        )

        agent_response = agent(user_query)

        print(f" Final response: {agent_response}")
```

### [Reflection Pattern](https://github.com/LokaHQ/agentic-patterns-101/blob/main/agentic-patterns/reflection_pattern.py)

![Reflection Pattern](https://cdn-images-1.medium.com/max/720/1*KfOfpZN-JSAwDcgsdmvxYw.png)

Reflection (or Self-Improvement Loop) is an iterative pattern where an agent generates initial output, receives feedback from a judge agent (can be the same agent depending on use case), and then refines its work based on that feedback. This pattern creates a feedback loop that continuously improves the quality of outputs through multiple iterations until the work meets specified criteria or reaches maximum iterations. The pattern mimics human creative processes where initial drafts are refined through critique and revision.

**Characteristics of Reflection Pattern**:

- Iterative improvement through feedback loops
- Structured feedback mechanisms with clear improvement criteria
- Quality enhancement through multiple refinement cycles

**Use Cases & Industries**:

- **Creative Content Generation**: Initial draft creation followed by critique and refinement for scripts, marketing copy, or creative writing - e.g., in Entertainment Industry, Advertising Agencies, or Content Creation platforms.
- **Code Review and Optimization**: Generate code solutions, analyze for bugs/improvements, then refactor based on feedback - e.g., in Software Development, DevOps pipelines, or Automated Code Review systems.
- **Document Review and Editing**: Create initial documents, evaluate for clarity/accuracy/compliance, then revise accordingly - e.g., in Legal Document Preparation, Technical Writing, or Regulatory Compliance reporting.
- **Research and Analysis**: Generate initial findings, critique methodology and conclusions, then strengthen analysis based on feedback - e.g., in Academic Research, Market Analysis, or Scientific Report generation.

**When to Use / When Not to Use**:

- **Use when**: Output quality is critical, iterative improvement adds value, or you need systematic refinement processes with clear evaluation criteria.
- **Avoid when**: Simple tasks don't benefit from iteration, feedback overhead exceeds improvement value, or single pass results are sufficient for the use case.

**Code Example**:

- For this example, we will demonstrate iterative content improvement through structured feedback. One single strands agent will play dual roles. A content creator and a judge providing feedback.

```python
"""Example of Reflection Pattern for iterative improvement of screenwriting."""

from pydantic import BaseModel, Field
from strands import Agent
from strands.models import BedrockModel

class JudgeOutput(BaseModel):
    """
    Model representing the feedback assessment from the judge agent.
    """

    needs_improvement: bool = Field(
        ..., description="Indicates if the screenwriting needs improvement."
    )
    feedback: str = Field(..., description="Feedback on the agent's screenwriting.")

if __name__ == "__main__":
    """
    Main entry point for the screenwriting assistant.
    """

    # Definining Agent
    agent = Agent(
        model=BedrockModel(
            model_id="amazon.nova-pro-v1:0",
            temperature=0.9,
        ),
        system_prompt="You are a screenwriting assistant. Help users create scripts, dialogues, and storylines. Critique existing works when asked to and provide suggestions for improvement.",
    )

    # User query
    user_query = "Write a dialogue between two characters. It needs to be a gripping thrilling scene."

    # Get initial screenwriting
    screenwriting = agent(user_query)

    # Run reflection_loop
    iteration = 1
    max_iterations = 3
    while iteration <= max_iterations:
        # Get feedback
        feedback_query = (
            "Please check this dialogue, and unless it's perfect, provide feedback: "
            + str(screenwriting)
        )
        feedback = agent.structured_output(JudgeOutput, feedback_query)

        # Check if feedback is needed
        if not feedback.needs_improvement:
            print("No feedback provided. Ending the loop.")
            break

        print("User feedback received. Processing...")

        # Adding feedback to screenwriting iteration
        screenwriting_feedback = (
            "Make this screenwriting better: "
            + str(screenwriting)
            + "\n With the following feedback: "
            + feedback.feedback
        )
        screenwriting = agent(screenwriting_feedback)

        iteration += 1
```

## Wrapping Up

So far, we've walked through the **first four patterns** : Sequential Workflow, Parallel Workflow, LLM Routing, and Reflection. Each of these highlights a different dimension of how we can structure AI systems, from simple ordered pipelines to adaptive feedback-driven designs.

What ties them together is that they provide **clarity and control**: workflows ensure determinism, routing adds adaptability, and reflection brings in self-improvement loops. Together, they form the foundation for building more advanced architectures.

In the next part of this series, we'll move beyond foundational workflows and look at **tool-based** and **multi-agent patterns**. These unlock even more flexibility, allowing systems to tap into external capabilities, collaborate through agents, and scale into complex, emergent behaviors.

Stay tuned - we'll explore how **Pure Python Tools**, **MCP Server Tools**, **Agents as Tools**, **Swarm**, **and the Graph Multi-Agent Pattern** push the boundaries of what AI systems can do !
