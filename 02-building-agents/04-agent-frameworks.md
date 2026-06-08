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
from openai import Agent, Runner

agent = Agent(
    name="Research Assistant",
    instructions="You are a helpful research assistant. Use tools to find information.",
    tools=[search_function, calculate_function],
)

result = Runner.run_sync(agent, messages=[{"role": "user", "content": "Research quantum computing trends"}])
print(result.final_output)
```

**Strengths**: First-party support, simple API, guaranteed compatibility with OpenAI models, built-in tracing.

**Weaknesses**: OpenAI models only. Less flexible than LangGraph for complex flows. Younger ecosystem.

**When to use**: OpenAI-centric stacks. Teams that want simplicity and first-party support.

## Comparison

| Framework | Complexity | Flexibility | Multi-Agent | Ecosystem | Best For |
|-----------|-----------|-------------|-------------|-----------|----------|
| Raw API | Low | Full | Manual | None | Learning, simple agents |
| LangGraph | High | High | Yes | Large | Complex stateful agents |
| CrewAI | Medium | Medium | First-class | Growing | Multi-agent workflows |
| OpenAI SDK | Low | Medium | Basic | OpenAI only | Simple OpenAI agents |

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

The best framework choice is the one your team can maintain. A well-understood raw API agent is better than a poorly understood framework agent. Do not adopt a framework to solve a problem you do not have yet.
