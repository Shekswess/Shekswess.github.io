---
layout: post
title: "Part 2: Agentic Patterns 101 with Loka & Strands-Agents"
description: "Agentic Patterns 101, with Strands-Agents"
author: shekswess
date: 2025-09-03 00:00:00 +0800
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

_**Original Source of the blog post: [Part 2: Agentic Patterns 101 with Loka & Strands-Agents](https://medium.com/loka-engineering/part-2-agentic-patterns-101-with-loka-strands-agents-86f6a1ad14e5)**_

_**Blog was written with my colleagues from Loka: Nina Cvetkovska & Petar Kalinovski**_

After exploring the foundations of Agentic Patterns in Part 1, where we looked at more structured workflows like Sequential and Parallel workflows, LLM Routing, and Reflection, this time we are going to push further into the more advanced side of agentic pattern designs.

In this second part, we’ll turn our attention to patterns that extend beyond predefined workflows and begin to showcase the real flexibility Strands-Agents,. Here, agents aren’t just decision-makers, they can become tools, orchestrators, collaborators, or even parts of larger graph-like systems. These patterns highlight how agentic systems can evolve from simple task runners into dynamic, interconnected ecosystems.

While Part 1 focused on how to structure work, Part 2 dives into how agents interact with their environment and with each other. This includes integrating with raw Python functions, leveraging external MCP servers, orchestrating worker agents, coordinating swarms, and managing complex graph-based interactions. Each of these patterns demonstrates a different way to scale capability, distribute reasoning, and balance autonomy with control.

As we said, all examples use Strands-Agents, the open-source SDK from AWS that makes it easy to test these approaches in a model-agnostic way. By the end, you’ll have a toolkit of patterns that span from single-agent efficiency to multi-agent collaboration, giving you the flexibility to choose the right design for your system’s needs.

## Pattern Overview

In this post, we’ll cover the remaining five patterns from the total of nine patterns:

- Sequential Workflow
- Parallel Workflow
- LLM Routing
- Reflection Pattern
- **Pure Python Tools**
- **Tools from MCP Servers**
- **Agents as Tools (Orchestrator-Workers)**
- **Swarm**
- **Graph Multi-Agent Pattern**

All of the examples for the patterns can be found on the GitHub repository [here](https://github.com/LokaHQ/agentic-patterns-101).

### [Pure Python Tools](https://github.com/LokaHQ/agentic-patterns-101/blob/main/agentic-patterns/pure-tools.py)

![Pure Python Tools](https://miro.medium.com/v2/resize:fit:640/format:webp/1*m4Hxid8AM6QudER3BycIag.png)

Pure Tools in Strands provide a straightforward and effective way to extend agent capabilities through specialized tools. In this pattern, each tool handles domain-specific logic while a single agent coordinates their usage through natural language interactions. This approach allows agents to work with multiple tools smoothly without requiring hardcoded workflows, keeping responsibilities modular and interactions intuitive. Pure Tools maintain clarity and ease of maintenance by separating tool functionality from agent reasoning. This lets the model focus on orchestration and decision-making rather than getting caught up in low-level task execution.

There are three main types of Pure Tools in Strands, categorized by how they are implemented and used within an agent:

- **Predefined Tools**: Built-in utilities provided by Strands for common tasks such as data retrieval, HTTP requests, or document searches.
- **Tools in Classes**: Custom tools implemented as methods inside a Python class, enabling stateful behavior or grouped domain-specific functionality.
- **Decorator Functions**: Python-decorated functions that define tools with simple input/output behavior, easy to register with agents, and reusable across different applications.

**Characteristics of Pure Python Tools**:

- Clean agent-tool architecture where tools handle specialized tasks
- Natural language coordination powered by intelligent models
- Clear separation between agent reasoning and tool execution
- Tools can be mixed, matched, and reused across different agents and projects

**Use Cases & Industries**:

- **E-commerce Operations**: Agents manage inventory, track orders, and handle fulfillment via Pure Tools — e.g., in online retail platforms.
- **Data Management & Analytics**: Agents perform CRUD operations, query databases, or generate reports — e.g., in Financial Services or Enterprise IT.
- **HR Systems**: Agents manage employee onboarding, leave requests, performance evaluations, and recruitment — e.g., in Corporate HR departments.
- **Enterprise Knowledge Management**: Agents access policies, compliance documents, or internal templates via Pure Tools — e.g., in Legal Tech or Enterprise Operations.

**When to Use / When Not to Use**:

- **Use when**: You have domain-specific tasks that can be encapsulated as independent tools and benefit from model-driven orchestration.
- **Avoid when**: Workflows are highly interdependent, require multi-step coordination across multiple specialized agents, or need complex multi-agent orchestration.

**Code Example:**

- This example demonstrates Pure Tools through a travel planning workflow. It features a TripPlanner class with tool-decorated methods for managing destinations and activities, plus a suggest_activity tool that uses agent intelligence for travel recommendations. The example also shows how the predefined tavily_search tool integrates to fetch destination data and activity ratings.

```python
"""Example of pure tools usage for Trip Planning."""

import os

from dotenv import load_dotenv
from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools.tavily import tavily_search

# Retrieve the Tavily API key from environment
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

SUGGESTION_INSTRUCTION = (
    "Suggest one specific, practical travel activity for {destination}."
    "Provide a concise recommendation (1-2 sentences) that includes:"
    "- A specific attraction, landmark, or experience"
    "- Brief context about why it's worth doing"
    "Example format: Visit [specific place] and [what you can do/see there]."
)


class TripPlanner:
    """Tracks destinations, travel dates, and planned activities."""

    def __init__(self):
        """Initialize an empty TripPlanner with no destinations or activities."""
        self.destinations = []
        self.itinerary = {}

    @tool
    def add_destination(self, city: str, start_date: str, end_date: str) -> str:
        """Add a destination to the trip with travel dates."""
        self.destinations.append(city)
        self.itinerary[city] = {
            "start_date": start_date,
            "end_date": end_date,
            "activities": [],
        }
        return f"{city} added to your trip from {start_date} to {end_date}."

    @tool
    def add_activity(self, city: str, activity: str) -> str:
        """Add an activity to a city's itinerary."""
        if city in self.itinerary:
            self.itinerary[city]["activities"].append(activity)
            return f"Added activity '{activity}' to {city} itinerary."
        return f"{city} is not in your trip."

    @tool
    def view_itinerary(self) -> str:
        """Generate a human-readable summary of the entire trip itinerary."""
        summary = ""
        for city, details in self.itinerary.items():
            summary += f"{city} ({details['start_date']} to {details['end_date']}):\n"
            for activity in details["activities"]:
                summary += f"  - {activity}\n"
        return summary if summary else "No destinations planned yet."


@tool
def suggest_activity(destination: str, agent: Agent) -> str:
    """Suggest a key activity for a given destination using the agent's model."""

    # Fill in the destination in the instruction
    prompt = SUGGESTION_INSTRUCTION.format(destination=destination)

    # Create a temporary agent with just the model for this generation
    temp_agent = Agent(model=agent.model)
    suggestion = temp_agent(prompt)

    return f"Suggestion for {destination}: {suggestion}"


if __name__ == "__main__":
    # Define the model for the agents
    model = BedrockModel(
        model_id="amazon.nova-lite-v1:0", temperature=0.7, max_tokens=3000
    )

    # Define planner and agent
    planner = TripPlanner()
    agent = Agent(
        tools=[
            suggest_activity,
            planner.add_destination,
            planner.add_activity,
            planner.view_itinerary,
            tavily_search,
        ],
        model=model,
    )

    # Example Usage
    response = agent("Add Tokyo as a destination from 2025-10-01 to 2025-10-07")
    response = agent("Suggest an activity for Tokyo and add it to my itinerary")
    response = agent("Search for ratings for each item in the itinerary.")
    response = agent("Show me my complete travel itinerary")
```

### [Tools from MCP Server](https://github.com/LokaHQ/agentic-patterns-101/blob/main/agentic-patterns/mcp-server-tools.py)

![Tools from MCP Server](https://miro.medium.com/v2/resize:fit:640/format:webp/1*RfutlFY02t93uMYSdosjBg.png)

The Model Context Protocol (MCP) is an open standard that defines how applications provide context to Large Language Models (LLMs). MCP enables communication between agents and MCP servers, which expose additional tools that agents can call to extend their capabilities. These tools can range from database queries to APIs or custom business logic, allowing agents to act beyond their base reasoning abilities.

There are three standardized types of MCP servers, categorized by transport protocol:

- **STDIO (Standard Input/Output)**, usually when the client and the server are on the same side
- **SSE (server-sent events)**, currently being deprecated
- **Streamable HTTP Events**, the modern, preferred approach

**Characteristics of Tools from MCP Servers:**:

- Provide structured, standardized communication between agents and external tools
- Model-agnostic: any LLM can leverage MCP-defined tools
- Extend agent functionality with external data, APIs, or services
- Enable interoperability between multiple systems and agents

**Use Cases & Industries**:

- **Enterprise Data Access**: Agents retrieve structured data from ERP/CRM systems via MCP tools — e.g., in Enterprise IT or Customer Relationship Management.
- **Knowledge Retrieval**: An MCP server provides search or vector database querying tools — e.g., in Legal Tech for case document retrieval.
- **Operational Automation**: Agents call MCP-exposed APIs (via MCP tools) for scheduling, ticketing, or reporting — e.g., in Customer Support or HR Systems.
- **Healthcare & Life Sciences**: MCP tools that can expose access to medical ontologies or EHR (Electronic Health Records) queries in a standardized way.

**When to Use / When Not to Use**:

- **Use when**: Agents need structured, standardized access to external tools, systems, or APIs in a way that is interoperable and vendor-neutral.
- **Avoid when**: The task is fully self-contained within the LLM (e.g., pure text summarization) or when a lightweight, direct API call without protocol overhead is sufficient.

**Code Example:**

- This example shows how we can use a STDIO MCP Server tools for AWS HealthOmics workflows.

```python
"""Example of tool use Agent with MCP server tools"""

# from mcp.client.sse import sse_client - for HTTP MCP servers that use Server-Sent Events (SSE)
# from mcp.client.streamable_http import streamablehttp_client - for HTTP MCP servers that use Streamable HTTP Events
from mcp import StdioServerParameters, stdio_client  # for MCP servers that use STDIO
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient

INSTRUCTION = "You are a Bioinformatics expert, that has a deep understanding of genomic data analysis and variant calling."

# Creating an STDIO client for the MCP server because that's the way to interact with this server
# https://awslabs.github.io/mcp/servers/aws-healthomics-mcp-server
mcp_client = MCPClient(
   lambda: stdio_client(
       StdioServerParameters(
           command="uvx", args=["awslabs.aws-healthomics-mcp-server"]
       )
   )
)

# If the MCP server uses Server-Sent Events (SSE)
# mcp_client = MCPClient(lambda: sse_client("http://localhost:8000/sse"))

# If the MCP server uses Streamable HTTP Events
# mcp_client = MCPClient(lambda: streamablehttp_client("http://localhost:8000/mcp"))


if __name__ == "__main__":
   # Define the model we use for the agents
   bedrock_model = BedrockModel(
       model_id="us.amazon.nova-premier-v1:0",
       temperature=0.1,
   )

   # Create an agent with MCP tools - must be within the context of the MCP client
   with mcp_client:
       # Get the tools from the MCP server
       tools = mcp_client.list_tools_sync()

       # Create an agent with these tools
       agent = Agent(tools=tools, system_prompt=INSTRUCTION, model=bedrock_model)

       # Define the query/question and execute the agent
       query = "Help me create a new genomic variant calling workflow"
       result = agent(query)
       print("The result is:", result)
```

### [Agent as Tools](https://github.com/LokaHQ/agentic-patterns-101/blob/main/agentic-patterns/agents-as-tools.py)

![Agent as Tools](https://miro.medium.com/v2/resize:fit:640/format:webp/1*PYORRGfq-pC7XB3RJL1Q4Q.png)

The Agents as Tools pattern is a multi-agent architecture where individual agents make their capabilities available as callable tools. A central orchestrator coordinates these specialized agents to tackle complex, multi-step problems. Each worker agent contains domain-specific knowledge and skills, letting the orchestrator mix and match their capabilities to build complete solutions. This approach promotes modularity, scalability, and clear responsibility divisions across agents.

**Characteristics of Agents as Tools**:

Domain-specialized agents handle specific tasks or knowledge areas
Orchestrator-managed workflows enable flexible combinations of worker agents
Modular architecture supports scalable deployments across large systems
Inter-agent communication allows efficient result sharing and coordination
New worker agents can be integrated without disrupting existing functionality

**Use Cases & Industries**:

- **Customer Support**: Orchestrator agent coordinates multiple chatbot agents to handle FAQs, ticket routing, and escalation — e.g., in Helpdesk or Customer Relationship Management.
- **Finance**: Worker agents perform loan processing, fraud detection, and credit assessment in a coordinated workflow — e.g., in Banking or FinTech applications.
- **Education**: Adaptive learning agents collaborate with curriculum planning agents to provide personalized recommendations — e.g., in EdTech platforms for student learning paths.
- **Healthcare & Life Sciences**: Specialist agents provide diagnostics, treatment suggestions, and scheduling assistance under orchestrator supervision — e.g., in Hospitals or Telemedicine platforms.

**When to Use / When Not to Use**:

- **Use when**: Tasks require multiple specialized agents working together, or when workflows are complex enough to benefit from modular architecture.
- **Avoid when**: The application is straightforward or single-domain, where one capable agent can handle all tasks effectively.

**Code Example:**

- This example shows how to use Strands agents as tools in an HR management workflow. Multiple specialized worker agents, each decorated as tools, handle employee data, leave management, performance reviews, and recruitment tasks. A central orchestrator coordinates these workers to respond to complex HR queries. The example also integrates predefined Strands tools like retrieve and http_request to access company knowledge bases and external data sources.

```python
"""Example of agents as tools usage for HR Management."""

import os

from dotenv import load_dotenv
from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import http_request, retrieve

# Define knowledge base
load_dotenv()
os.environ["KNOWLEDGE_BASE_ID"] = os.getenv("STRANDS_KNOWLEDGE_BASE_ID")
os.environ["AWS_REGION"] = os.getenv("STRANDS_KNOWLEDGE_BASE_REGION")

# Define prompts
EMPLOYEE_DATA_PROMPT = """
You are an Employee Data Specialist. You have access to employee records and company knowledge base.
Use the retrieve tool to search for relevant information from company documents, org charts,
job descriptions, and employee databases.

Always be professional and respect employee privacy - only share information that would
be appropriate for HR or management purposes.
"""

HR_ORCHESTRATOR_PROMPT = """
You are an HR Management Assistant that coordinates various HR functions. You have access to
specialized HR agents that can help with different aspects of human resources management:

1. Employee Data Assistant - Employee records, organizational charts, department information
2. Leave Management Assistant - PTO, vacation requests, sick leave, time off policies
3. Performance Review Assistant - Performance reviews, goal setting, career development
4. Recruitment Assistant - Hiring, job postings, candidate evaluation, recruitment strategy

When you receive a query, determine which specialist(s) can best help and route the request
appropriately. You can also coordinate between multiple specialists if needed for complex requests.

Always maintain professionalism and respect employee privacy and confidentiality.
"""

RECRUITMENT_PROMPT = """
You are a Recruitment Specialist. You help with hiring processes, candidate evaluation,
job posting creation, and recruitment strategy. You understand job requirements and
can match candidates to positions effectively.

Focus on:
- Job requirement analysis
- Candidate screening criteria
- Interview question preparation
- Recruitment process optimization
- Diversity and inclusion in hiring
"""

LEAVE_MANAGEMENT_PROMPT = """
You are a Leave Management Specialist. You handle all aspects of employee time off.
Use the retrieve tool to access company leave policies, holiday calendars, and approval workflows.
"""

PERFORMANCE_REVIEW_PROMPT = """
You are a Performance Management Specialist. You handle performance reviews, goal setting,
and career development using company resources and best practices.
Always search for the latest company guidelines and templates before providing advice.
Focus on objective, constructive feedback aligned with company values and processes.
"""


@tool
def employee_data_assistant(query: str, agent: Agent) -> str:
    """
    Handle employee data queries like finding employee information, department searches,
    organizational charts, and basic employee record management.
    """
    try:
        employee_agent = Agent(
            model=agent.model, system_prompt=EMPLOYEE_DATA_PROMPT, tools=[retrieve]
        )

        # Let the agent use retrieve to search company knowledge base
        enhanced_query = f"Search company knowledge base for: {query}"

        response = employee_agent(enhanced_query)
        return str(response)
    except Exception as e:
        return f"Error in employee data assistant: {str(e)}"


@tool
def leave_management_assistant(query: str, agent: Agent) -> str:
    """
    Handle leave and PTO related queries including checking balances, submitting requests,
    approving time off, and providing leave policy information.
    """
    try:
        leave_agent = Agent(
            model=agent.model, system_prompt=LEAVE_MANAGEMENT_PROMPT, tools=[retrieve]
        )

        # First search company knowledge base, then supplement with current data
        enhanced_query = f"First, search company knowledge base for leave policies and procedures related to: {query}"

        response = leave_agent(enhanced_query)
        return str(response)
    except Exception as e:
        return f"Error in leave management assistant: {str(e)}"


@tool
def performance_review_assistant(query: str, agent: Agent) -> str:
    """
    Handle performance review queries including review scheduling, performance analysis,
    goal setting, and career development planning.
    """
    try:
        performance_agent = Agent(
            model=agent.model, system_prompt=PERFORMANCE_REVIEW_PROMPT, tools=[retrieve]
        )

        # Search company knowledge base first, then use current data
        enhanced_query = f"Search company knowledge base for performance management resources related to: {query}"

        response = performance_agent(enhanced_query)
        return str(response)
    except Exception as e:
        return f"Error in performance review assistant: {str(e)}"


@tool
def recruitment_assistant(query: str, agent: Agent) -> str:
    """
    Handle recruitment and hiring queries including job postings, candidate evaluation,
    interview processes, and hiring decisions.
    """
    try:
        recruitment_agent = Agent(
            model=agent.model,
            system_prompt=RECRUITMENT_PROMPT,
            tools=[retrieve, http_request],
        )

        response = recruitment_agent(query)
        return str(response)
    except Exception as e:
        return f"Error in recruitment assistant: {str(e)}"


if __name__ == "__main__":
    # Define the model for the agents
    model = BedrockModel(
        model_id="amazon.nova-lite-v1:0", temperature=0.7, max_tokens=3000
    )

    # Create the main orchestrator agent
    hr_orchestrator = Agent(
        model=model,
        system_prompt=HR_ORCHESTRATOR_PROMPT,
        callback_handler=None,
        tools=[
            employee_data_assistant,
            leave_management_assistant,
            performance_review_assistant,
            recruitment_assistant,
        ],
    )

    # Example for employee lookup
    response = hr_orchestrator(
        "Can you find information about Sarah Johnson and tell me about her role and performance?"
    )

    # Example for leave request
    response = hr_orchestrator(
        "What is Mike Chen's current PTO balance and does he have any pending leave requests?"
    )

    # Example for performance review
    response = hr_orchestrator(
        "I need to prepare for Sarah Johnson's upcoming performance review. What are her current goals and recent achievements?"
    )

    # Example for recruitment query
    response = hr_orchestrator(
        "We need to hire a new software engineer for the Engineering team. What skills should we look for based on our current team composition?"
    )

    # Example with policy lookup
    response_policy = hr_orchestrator(
        "What is our current remote work policy and how many days per week can employees work from home?"
    )

    # Example with template search
    response_template = hr_orchestrator(
        "I need to conduct a performance review for a Software Engineer. Can you find our performance review template and suggest key areas to evaluate?"
    )

    # Example with compliance search
    response_compliance = hr_orchestrator(
        "What are the legal requirements for conducting layoffs in California, and do we have internal guidelines for this process?"
    )
```

### [Swarm](https://github.com/LokaHQ/agentic-patterns-101/blob/main/agentic-patterns/swarm.py)

![Swarm](https://miro.medium.com/v2/resize:fit:640/format:webp/1*oeKudoczZjnqYD3dCnYTyg.png)

Swarm is a collaborative agent orchestration pattern where multiple specialized agents work together as autonomous peers to solve complex tasks. Unlike hierarchical multi-agent systems with central orchestrators, Swarm enables self-organizing coordination between agents through shared working memory and autonomous handoffs. Each agent has access to the full task context, can see the history of which agents have worked on the task, and can decide when to transfer control to another agent with different expertise based on their discoveries.

**Characteristics of Swarm Pattern**:

- Autonomous coordination without central control
- Shared working memory accessible to all agents
- Dynamic task distribution based on agent capabilities and discoveries
- Peer-to-peer handoffs triggered by findings or needs
- Emergent intelligence through collective collaboration
- Self-organizing agent teams with flexible interaction patterns

**Use Cases & Industries**:

- **Viral Content Creation**: Autonomous collaboration between trend analysts, meme creators, video creators, copywriters, and community managers who hand off based on discoveries and creative insights — e.g., in Social Media Marketing, Influencer Platforms, or Digital Marketing Agencies.
- **Research and Investigation**: Collaborative teams where data scientists, domain experts, and analysts hand off based on findings that emerge during investigation — e.g., in Market Research, Scientific Research, or Investigative Journalism.
- **Crisis Response Teams**: Emergency response specialists, communications experts, and resource coordinators collaborating dynamically as situations evolve — e.g., in Emergency Management, Incident Response, or Crisis Communications.

**When to Use / When Not to Use**:

- **Use when**: Task outcomes are unpredictable, creative collaboration adds value, no single agent can determine the optimal sequence, or when discoveries by one agent should influence the entire approach.
- **Avoid when**: Tasks have clear hierarchies or sequences, central coordination is more efficient, or when agents don’t benefit from shared context and autonomous decision-making.

**Code Example**:

- In this example we are showing how a simple Swarm Agent can be used to create viral content for social media platforms.

```python
"""Example of a viral content creation swarm"""

from strands import Agent
from strands.models import BedrockModel
from strands.multiagent import Swarm
from strands.types.content import ContentBlock

# Agent prompts
TREND_ANALYST_PROMPT = "You are a trend analyst who analyzes viral content. You can view images and videos to understand what visual elements are trending. Identify patterns in successful content and timing windows."

MEME_CREATOR_PROMPT = "You are a visual meme creator who can both analyze existing memes and generate new ones. Create engaging visual content using trending formats. You can generate images and critique visual content."

VIDEO_CREATOR_PROMPT = "You are a video content creator who can create new videos. Generate short-form videos and analyze what makes videos go viral on different platforms."

COPYWRITER_PROMPT = "You are a social media copywriter. You can view images and videos to write appropriate captions and copy that matches the visual content."
COMMUNITY_MANAGER_PROMPT = "You are a community manager who can analyze visual content and engagement patterns. You can view memes, videos, and posts to understand audience reactions."



if __name__ == "__main__":
    """
    Main function to execute the swarm process.
    """

    # Create the agents
    trend_analyst = Agent(
        model=BedrockModel(model_id="amazon.nova-pro-v1:0", temperature=0.7),
        system_prompt=TREND_ANALYST_PROMPT,
    )
    meme_creator = Agent(
        model=BedrockModel(
            model_id="amazon.nova-canvas-v1:0", temperature=0.8
        ),  # Image generation model
        system_prompt=MEME_CREATOR_PROMPT,
    )
    video_creator = Agent(
        model=BedrockModel(
            model_id="amazon.nova-reel-v1:0", temperature=0.8
        ),  # Video generation model
        system_prompt=VIDEO_CREATOR_PROMPT,
    )
    copywriter = Agent(
        model=BedrockModel(model_id="amazon.nova-lite-v1:0", temperature=0.9),
        system_prompt=COPYWRITER_PROMPT,
    )
    community_manager = Agent(
        model=BedrockModel(model_id="amazon.nova-lite-v1:0", temperature=0.6),
        system_prompt=COMMUNITY_MANAGER_PROMPT,
    )

    # Create the swarm
    viral_content_swarm = Swarm(
        [trend_analyst, meme_creator, video_creator, copywriter, community_manager],
        max_handoffs=15,
        max_iterations=10,
    )

    # Creating the prompt using ContentBlocks
    trending_meme_bytes = b"..."  # Placeholder for trending meme image bytes
    content_blocks = [
        ContentBlock(
            text="Create a viral campaign about AI breakthroughs. Something similar to the campaign from last year:"
        ),
        ContentBlock(image={"format": "png", "source": {"bytes": trending_meme_bytes}}),
    ]

    result = viral_content_swarm(content_blocks)

    print(f"Swarm Status: {result.status}")
    print(f"Node history: {[node.node_id for node in result.node_history]}")
```

### [Graph Multi-Agent Pattern](https://github.com/LokaHQ/agentic-patterns-101/blob/main/agentic-patterns/graph-multi-agent-tool.py)

![Graph Multi-Agent Pattern](https://miro.medium.com/v2/resize:fit:640/format:webp/1*ka0lfnyM3NtAi2mn55_L3g.png)

The Graph Multi-Agent Pattern is a structured, DAG-based orchestration approach where each node represents an agent (or custom multi-agent group), and edges define execution dependencies. This pattern lets you build complex workflows with explicit structure, allowing agents to coordinate their actions in a defined sequence. Nodes can house specialized agents with distinct capabilities, while edges control dependencies and data flow between them. The pattern supports conditional branching and multi-modal inputs, so workflows can adapt based on intermediate results or different input types.

**Characteristics of Graph Multi-Agent Pattern**:

- Deterministic execution order creates predictable and reproducible workflows
- Clear dependency management supports nested patterns, hierarchical coordination, and conditional logic
- Nodes can contain custom agent types tailored for specific workflow tasks
- Handles multi-modal inputs and conditional branching for complex scenarios
- Modular architecture allows you to add or remove agents without breaking the overall workflow

**Use Cases & Industries**:

- **Financial Analysis**: Agents fetch market or company data, analyze trends, summarize insights, and generate reports — e.g., in Banking, Investment Research, or Risk Management.
- **Content Creation**: Multi-agent pipelines research topics, draft content, edit, and publish — e.g., in Marketing, Journalism, or Educational Content Development.
- **IoT & Industrial Automation**: Agents monitor devices, diagnose issues, optimize performance, and trigger alerts — e.g., in Smart Manufacturing, Energy Management, or Logistics.
- **Healthcare & Life Sciences**: Data-collection agents gather patient or research data, analysis agents process it, diagnostic agents provide recommendations, and reporting agents summarize findings — e.g., in Clinical Decision Support or Hospital Workflow Management.

**When to Use / When Not to Use**:

**Use when**: Your workflows have complex dependencies, multiple agents bring distinct expertise, task outcomes depend on coordinated execution, or intermediate results need to influence downstream agents.
**Avoid when**: Tasks are straightforward or linear, a single agent can handle the workflow effectively, or when deterministic orchestration adds complexity without meaningful benefits.

**Code Example**:

- This example shows how to use Strands agents in a graph-based multi-agent workflow to create a structured Dinner Plan. Each specialized agent handles a specific task (recipe creation, beverage pairing, music curation), while a central planner agent combines their outputs into a cohesive final result.

```python
"""Example of a graph-based multi-agent workflow for generating a structured Dinner Plan."""

from strands import Agent
from strands.models import BedrockModel
from strands.multiagent import GraphBuilder

ROUTER_INSTRUCTION = (
    "Interpret the user's dinner request and dispatch tasks to appropriate agents."
)
RECIPE_CREATOR_INSTRUCTION = (
    "Propose a refined recipe suitable for the requested dinner style."
)
BEVERAGE_SUGGESTOR_INSTRUCTION = (
    "Suggest a beverage that complements the dinner style and recipe."
)
MUSIC_CURATOR_INSTRUCTION = "Curate a playlist that matches the dinner ambiance and enhances the dining experience."
DINNER_PLANNER_INSTRUCTION = (
    "Combine the outputs from RecipeAgent, BeverageAgent, and MusicAgent into a single structured DinnerPlan object. "
    "The DinnerPlan should have the following fields:"
    "1. recipe: include all recipe details with subfields:"
    "   - name (str)"
    "   - ingredients (list of str)"
    "   - preparation_time (int, in minutes)"
    "   - instructions (str)"
    "2. beverage: include beverage details with subfields:"
    "   - name (str)"
    "   - type (str, e.g., wine, cocktail, non-alcoholic)"
    "   - pairing_notes (str, explaining why it pairs with the recipe)"
    "3. music_playlist: include music details with subfields:"
    "   - title (str)"
    "   - tracks (list of objects, each with:"
    "       - track_name (str)"
    "       - artist (str)"
    "       - genre (str, e.g., Classical, Jazz, Pop)"
    "     )"
    "Return the combined DinnerPlan object in valid JSON format that matches the DinnerPlan Pydantic model exactly."
)

if __name__ == "__main__":
    user_query = "Dinner style: formal Italian evening with elegance."

    # Define the model for the agents
    model = BedrockModel(
        model_id="amazon.nova-lite-v1:0", temperature=0.7, max_tokens=3000
    )

    # Create specialized agents
    input_router = Agent(
        model=model,
        name="InputRouter",
        system_prompt=ROUTER_INSTRUCTION,
    )
    recipe_creator = Agent(
        model=model,
        name="RecipeAgent",
        system_prompt=RECIPE_CREATOR_INSTRUCTION,
    )
    beverage_suggester = Agent(
        model=model,
        name="BeverageAgent",
        system_prompt=BEVERAGE_SUGGESTOR_INSTRUCTION,
    )
    music_curator = Agent(
        model=model,
        name="MusicAgent",
        system_prompt=MUSIC_CURATOR_INSTRUCTION,
    )

    dinner_planner = Agent(
        model=model,
        name="DinnerPlanAgent",
        system_prompt=DINNER_PLANNER_INSTRUCTION,
    )

    # Build the multi-agent graph
    builder = GraphBuilder()

    # Add nodes for each agent
    builder.add_node(input_router)
    builder.add_node(recipe_creator)
    builder.add_node(beverage_suggester)
    builder.add_node(music_curator)
    builder.add_node(dinner_planner)

    # Define dependencies
    builder.add_edge(
        "InputRouter", "RecipeAgent"
    )  # Router → Recipe (user preference input)
    builder.add_edge(
        "InputRouter", "BeverageAgent"
    )  # Router → Beverage (user preference input)
    builder.add_edge(
        "RecipeAgent", "BeverageAgent"
    )  # Recipe → Beverage (recipe-specific input)
    builder.add_edge(
        "InputRouter", "MusicAgent"
    )  # Router → Music (user preference input)
    builder.add_edge(
        "RecipeAgent", "DinnerPlanAgent"
    )  # Recipe → Plan (recipe-specific input)
    builder.add_edge(
        "BeverageAgent", "DinnerPlanAgent"
    )  # Beverage → Plan (beverage-specific input)
    builder.add_edge(
        "MusicAgent", "DinnerPlanAgent"
    )  # Music → Plan (music-specific input)

    # Set the entry point/s for the graph (Optional - If not spesified, it will be auto-detected)
    builder.set_entry_point("InputRouter")
    graph = builder.build()

    # Execute the graph with a prompt
    result = graph(user_query)
```

- The other example shows how a single agent can dynamically build and run a multi-agent workflow using the Strands graph tool, without manually constructing a GraphBuilder.

```python
"""Example of graph multi-agent pattern using the graph tool (strands tool package)."""

from strands import Agent
from strands.models import BedrockModel
from strands_tools import graph

GRAPH_AGENT_INSTRUCTION = "Create a graph of agents to interpret the user's dinner request and dispatch tasks to appropriate agents."

if __name__ == "__main__":
    user_query = "Dinner style: formal Italian evening with elegance."

    # Define the model for the agents
    model = BedrockModel(
        model_id="amazon.nova-lite-v1:0", temperature=0.7, max_tokens=3000
    )

    agent = Agent(tools=[graph], system_prompt=GRAPH_AGENT_INSTRUCTION)
    agent(user_query)
```

## Conclusion

Across these two posts, we’ve seen nine Agentic Pattern — from straightforward workflows to swarm and graph-based systems. The takeaway is simple: choose the pattern that fits the problem. Workflows give you structure and reliability; agents give you flexibility and autonomy; hybrids let you blend both.

Use these patterns as a toolbox to build systems that are robust, scalable, and future-ready and remember: the best agentic pattern design is the one that keeps things simple while delivering real impact.

# References

If you haven’t check Part 1 of the blog post series, you can check it on this link:
- [Part 1: Agentic Patterns 101 with Loka & Strands-Agents](https://medium.com/loka-engineering/part-1-agentic-patterns-101-with-loka-strands-agents-13f45ea62c70)