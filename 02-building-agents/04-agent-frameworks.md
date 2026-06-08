# Agent Frameworks

## Framework vs Raw API

You can build agents with raw API calls (as shown in [01-first-agent](01-first-agent.md)). Frameworks provide abstractions for common patterns: agent loops, tool management, memory, state persistence, and observability. The question is whether the abstraction saves more time than it costs in complexity.

## The Landscape (2026)

### LangChain / LangGraph

LangChain provides building blocks (chains, tools, memory). LangGraph adds graph-based orchestration for multi-step agents.

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return web_search_api(query)

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return safe_eval(expression)

model = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(model, [search, calculate])

result = agent.invoke({"messages": [("user", "What is 15% of the population of Tokyo?")]})
print(result["messages"][-1].content)
```

**Strengths**: Large ecosystem, many integrations, graph visualization, built-in persistence. LangGraph's StateGraph gives fine-grained control over agent flow.

**Weaknesses**: Complexity. LangChain's abstraction layers can be confusing. Debugging through multiple abstraction layers is harder than debugging raw API calls. Version churn is frequent.

**When to use**: Multi-step agents with complex state, agents that need persistence across sessions, teams that want ecosystem integrations.

### CrewAI

Multi-agent orchestration focused on role-based agents.

```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Research Analyst",
    goal="Find comprehensive, factual information on the given topic",
    backstory="You are an experienced researcher who prioritizes accuracy.",
    tools=[search_tool],
)

writer = Agent(
    role="Technical Writer",
    goal="Write clear, well-structured reports",
    backstory="You are a skilled writer who makes complex topics accessible.",
)

research_task = Task(
    description="Research the current state of AI agent frameworks",
    agent=researcher,
    expected_output="A comprehensive research summary with key findings",
)

writing_task = Task(
    description="Write a report based on the research findings",
    agent=writer,
    expected_output="A well-structured markdown report",
)

crew = Crew(agents=[researcher, writer], tasks=[research_task, writing_task])
result = crew.kickoff()
```

**Strengths**: Multi-agent patterns are first-class. Role-based design maps naturally to real workflows. Easy to set up.

**Weaknesses**: Less control over execution flow than LangGraph. Fewer integrations. Overhead for single-agent use cases.

**When to use**: Multi-agent workflows where each agent has a distinct role. Content generation pipelines. Research and analysis tasks.

### OpenAI Agents SDK

Official SDK from OpenAI. Lightweight, tight integration with OpenAI models.

```python
from agents import Agent, Runner, function_tool

@function_tool
def search(query: str) -> str:
    """Search the web for information."""
    return web_search_api(query)

agent = Agent(
    name="Research Assistant",
    instructions="You are a helpful research assistant. Use tools to find information.",
    tools=[search],
)

result = Runner.run_sync(agent, "Research quantum computing trends")
print(result.final_output)
```

**Strengths**: First-party support, simple API, guaranteed compatibility with OpenAI models, built-in tracing.

**Weaknesses**: OpenAI models only. Less flexible than LangGraph for complex flows. Younger ecosystem.

**When to use**: OpenAI-centric stacks. Teams that want simplicity and first-party support.

### Google ADK (Agent Development Kit)

Google's open-source framework for building agents with Gemini models. Multi-language (Python, Java).

```python
from google.adk import Agent, Runner
from google.adk.tools import function_tool

@function_tool
def search(query: str) -> str:
    """Search the web for information."""
    return web_search_api(query)

agent = Agent(
    name="research_agent",
    model="gemini-2.5-flash",
    instruction="You are a research assistant. Use tools to find accurate information.",
    tools=[search],
)

runner = Runner(agent=agent)
result = runner.run("What are the latest trends in AI agents?")
print(result.text)
```

**Strengths**: First-party Google support, Gemini model integration, built-in evaluation tools, supports A2A (Agent-to-Agent) protocol for inter-agent communication.

**Weaknesses**: Tightly coupled to Google ecosystem. Less community traction than LangChain. Documentation still maturing.

**When to use**: Google Cloud shops using Gemini. Teams wanting A2A protocol for multi-agent systems. Java-based teams (ADK has first-class Java support).

### Strands Agents (AWS)

AWS's open-source framework for building agents with any model. Model-agnostic, designed for production on AWS.

```python
from strands import Agent, tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return web_search_api(query)

@tool
def query_database(sql: str) -> str:
    """Query the product database."""
    return execute_sql(sql)

agent = Agent(
    model="us.anthropic.claude-sonnet-4-20250514",
    tools=[search, query_database],
    system_prompt="You are a data analyst assistant.",
)

result = agent("Find the top 10 selling products this month")
print(result.message)
```

**Strengths**: Model-agnostic (works with Claude, GPT, Gemini, Llama, Mistral). Built-in AWS integrations (Bedrock, S3, DynamoDB). Tool ecosystem with MCP (Model Context Protocol) support. Production-focused with built-in observability.

**Weaknesses**: AWS-biased architecture. Smaller community than LangChain. Python only (for now).

**When to use**: AWS deployments using Bedrock. Teams that want model flexibility. Production agents that need AWS-native tooling.

### Semantic Kernel (Microsoft)

Microsoft's SDK for building AI agents in .NET, Python, and Java. Enterprise-focused.

```csharp
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;

var builder = Kernel.CreateBuilder()
    .AddAzureOpenAIChatCompletion("gpt-4o", endpoint, apiKey);

builder.Plugins.AddFromType<SearchPlugin>();
builder.Plugins.AddFromType<CalculatorPlugin>();

var kernel = builder.Build();

var result = await kernel.InvokePromptAsync(
    "What is 15% of the population of Tokyo?",
    new OpenAIPromptExecutionSettings { ToolCallBehavior = ToolCallBehavior.AutoInvokeKernelTools }
);

Console.WriteLine(result);
```

**Strengths**: First-class .NET support (the only major framework with C# as primary). Enterprise integration with Azure. Plugin architecture. Process framework for deterministic workflows.

**Weaknesses**: Less Python-native feel. Heavier abstraction layer. Smaller agent-specific community than LangChain.

**When to use**: .NET/Azure shops. Enterprise environments with compliance requirements. Teams that want to integrate AI into existing .NET applications.

### AWS Bedrock AgentCore

Managed agent infrastructure on AWS. Not a framework — a hosted service that handles agent lifecycle, memory, and orchestration.

```python
import boto3

client = boto3.client("bedrock-agentcore")

response = client.create_agent(
    agentName="product-assistant",
    agentResourceRoleArn="arn:aws:iam::123456789012:role/AgentRole",
    foundationModel="anthropic.claude-sonnet-4-20250514-v1:0",
    instruction="You are a product assistant. Help users find and compare products.",
)

response = client.create_action_group(
    agentId=response["agent"]["agentId"],
    actionGroupName="ProductSearch",
    actionGroupExecutor={
        "lambda": "arn:aws:lambda:us-east-1:123456789012:function:product-search"
    },
    functionSchema={
        "functions": [{
            "name": "search_products",
            "description": "Search products by category and price range",
            "parameters": {
                "category": {"type": "string"},
                "max_price": {"type": "number"}
            }
        }]
    },
)
```

**Strengths**: Fully managed — no infrastructure. Built-in memory, guardrails, and knowledge bases. Automatic scaling. Integrates with AWS IAM for security.

**Weaknesses**: AWS lock-in. Less control over agent execution. Limited customization of agent loop. Cost can grow with usage.

**When to use**: Teams that want agents without managing infrastructure. AWS-native architectures. Rapid prototyping with production-grade guardrails.

## Comparison

| Framework | Language | Model | Multi-Agent | Complexity | Best For |
|-----------|----------|-------|-------------|------------|----------|
| Raw API | Any | Any | Manual | Low | Learning, simple agents |
| LangGraph | Python, JS | Any | Yes | High | Complex stateful agents |
| CrewAI | Python | Any | First-class | Medium | Multi-agent role workflows |
| OpenAI SDK | Python, JS | OpenAI | Basic | Low | Simple OpenAI agents |
| Google ADK | Python, Java | Gemini | Yes (A2A) | Medium | Google Cloud + Gemini stacks |
| Strands | Python | Any | Yes | Medium | AWS Bedrock, model flexibility |
| Semantic Kernel | C#, Python, Java | Any | Yes | Medium | .NET/Azure enterprise |
| Bedrock AgentCore | Any (managed) | AWS Bedrock | Managed | Low | Managed agents on AWS |

## When to Use Raw API vs Framework

Use raw API when:
- You are learning (understand the fundamentals before abstracting them)
- Your agent is simple (1-3 tools, linear flow)
- You need maximum control and minimal overhead
- You want to avoid dependency risk

Use a framework when:
- Your agent has complex state (branching, loops, persistence)
- You need multi-agent coordination
- You want built-in observability and debugging
- Your team is already using one

Use a managed service (Bedrock AgentCore) when:
- You want zero infrastructure management
- Your org is already on AWS and wants integrated security
- You need guardrails and compliance built-in

The best framework choice is the one your team can maintain. A well-understood raw API agent is better than a poorly understood framework agent. Do not adopt a framework to solve a problem you do not have yet.
