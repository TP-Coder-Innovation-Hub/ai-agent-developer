`[Senior]`

# Multi-Agent Systems

## Why Multiple Agents

A single agent that tries to do everything is a single point of failure. It tries to search the web, write code, review code, and deploy -- and does none of them well. Multi-agent systems specialize: each agent does one thing and does it well.

Reasons to use multiple agents:
- **Specialization**: A researcher agent reads papers. A writer agent drafts reports. Different system prompts, different tools, different evaluation criteria.
- **Parallelism**: Independent subtasks run concurrently. Three research agents search different sources simultaneously.
- **Quality control**: One agent produces output, another reviews it. Reviewer catches errors the producer missed.
- **Separation of concerns**: Different permission levels, different tool access, different safety boundaries.

## Orchestrator Pattern

The orchestrator is the coordinator. It receives the user request, delegates to specialist agents, and synthesizes the results.

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class AgentMessage:
    sender: str
    content: str
    metadata: dict[str, Any] | None = None

class Orchestrator:
    def __init__(self):
        self.agents: dict[str, callable] = {}
        self.history: list[AgentMessage] = []

    def register(self, name: str, agent_fn: callable):
        self.agents[name] = agent_fn

    def run(self, task: str) -> str:
        # Step 1: Decide which agents to use
        plan = self._plan(task)

        # Step 2: Execute agents (sequential or parallel)
        results = {}
        for step in plan:
            agent_name = step["agent"]
            agent_task = step["task"]
            results[agent_name] = self.agents[agent_name](agent_task)

        # Step 3: Synthesize
        return self._synthesize(task, results)

    def _plan(self, task: str) -> list[dict]:
        available = list(self.agents.keys())
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[{
                "role": "system",
                "content": f"""You are an orchestrator. Available agents: {available}
                Return JSON: {{"plan": [{{"agent": "name", "task": "description"}}]}}"""
            }, {
                "role": "user",
                "content": task
            }]
        )
        import json
        return json.loads(response.choices[0].message.content)["plan"]

    def _synthesize(self, original_task: str, results: dict) -> str:
        result_text = "\n".join(f"{k}: {v}" for k, v in results.items())
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Synthesize the following agent outputs into a coherent response."},
                {"role": "user", "content": f"Task: {original_task}\n\nAgent results:\n{result_text}"}
            ]
        )
        return response.choices[0].message.content
```

## Example: Researcher + Writer + Reviewer

Three agents collaborate on a research report.

```python
def researcher(task: str) -> str:
    """Searches and gathers information."""
    return agent_loop(
        system_prompt="You are a research agent. Gather factual information. Use search tools. Be thorough.",
        user_message=task,
        tools=search_tools,
    )

def writer(task: str) -> str:
    """Drafts a report from research findings."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a technical writer. Write clear, structured reports. Use markdown."},
            {"role": "user", "content": f"Write a report based on these findings:\n\n{task}"}
        ]
    )
    return response.choices[0].message.content

def reviewer(draft: str) -> str:
    """Reviews and provides feedback."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """You are an editor. Review for:
1. Factual accuracy (flag unsupported claims)
2. Logical consistency
3. Completeness (missing angles?)
4. Clarity and structure
Return specific, actionable feedback."""},
            {"role": "user", "content": f"Review this report:\n\n{draft}"}
        ]
    )
    return response.choices[0].message.content

# Wire it up
orchestrator = Orchestrator()
orchestrator.register("researcher", researcher)
orchestrator.register("writer", writer)
orchestrator.register("reviewer", reviewer)

result = orchestrator.run("Write a report on the current state of AI agent frameworks")
```

## Communication Between Agents

Agents communicate through structured messages, not free-form text. Define a message schema:

```python
class AgentResult(BaseModel):
    agent: str
    status: Literal["success", "failure", "needs_input"]
    output: str
    confidence: float = 1.0
    sources: list[str] = []
    errors: list[str] = []
```

This ensures the orchestrator can parse results reliably, handle failures, and route follow-up tasks.

## Failure Modes

- **Cascading failures**: One agent fails and downstream agents get garbage input. Mitigate with input validation at each agent boundary.
- **Deadlock**: Agent A waits for Agent B which waits for Agent A. Mitigate with timeouts and dependency ordering.
- **Cost explosion**: More agents = more LLM calls = more tokens. Track cost per agent and set budgets.
- **Context loss**: Information gets lost between agents. Mitigate with structured message passing and explicit context inclusion.

## When Not to Use Multi-Agent

If a single agent with good tools can do the job, use a single agent. Multi-agent systems add complexity, latency, and cost. They are justified when specialization provides clear value (different system prompts, different tools, different evaluation criteria) or when tasks can run in parallel.
