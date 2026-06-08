# What Are AI Agents

## LLM is not an Agent

An LLM takes text in and predicts text out. It responds. An agent acts.

The difference:

```python
# LLM: responds to input
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is the weather in Tokyo?"}]
)
# Output: "I don't have access to real-time data..." (correct but useless)

# Agent: perceives, reasons, acts, learns
def agent_loop(user_query):
    while True:
        # Perceive: understand the current state
        context = gather_context(user_query)

        # Reason: decide what to do
        decision = llm_reason(context)

        # Act: execute the decision (call a tool, respond, etc.)
        if decision.action == "call_tool":
            result = execute_tool(decision.tool, decision.args)
            add_to_context(result)
        elif decision.action == "respond":
            return decision.response

        # Learn: update state based on results (implicit in context updates)
```

The agent loop is `perceive -> reason -> act -> learn`. The LLM alone stops at "reason." The agent wraps the LLM in a loop that can interact with the world.

> 🖼️ **[IMAGE_PLACEHOLDER]** — LLM vs agent perceive reason act loop comparison

## Why Agents Matter

LLMs answer questions. Agents automate workflows.

Consider a customer support scenario:
- **LLM approach**: User asks a question, LLM answers based on training data. If the question requires looking up an order, the LLM cannot help.
- **Agent approach**: User asks a question. The agent decides whether to answer directly, look up the order in a database, issue a refund via an API, or escalate to a human. It acts.

Agents matter because most real-world tasks are not question-answering. They are workflows: gather information, make decisions, execute actions, handle exceptions. The LLM provides the reasoning engine. The agent provides the action capability.

## When to Use an Agent

Not everything needs an agent. Use this decision framework:

| Task Type | Solution | Example |
|-----------|----------|---------|
| Deterministic, single step | Script or function | Parse a date, compute a total |
| Single LLM call, no tools | Direct API call | Summarize text, translate |
| Multi-step, needs tools | Agent | Research a topic, process a refund |
| Complex, needs coordination | Multi-agent system | Full research pipeline with review |

> 🖼️ **[IMAGE_PLACEHOLDER]** — agent vs workflow vs script decision tree task complexity

If a script works, use a script. If a single API call works, use a single API call. Agents add complexity, latency, and cost. Use them when the task genuinely requires autonomous decision-making.

## Agent vs Workflow

This distinction is critical for architecture decisions:

**Workflow** -- predefined control flow. The developer decides the sequence of steps. The LLM executes within that structure. Predictable, testable, debuggable.

**Agent** -- autonomous control flow. The LLM decides which steps to take and in what order. More flexible but less predictable.

Start with workflows. Add agent autonomy only where the task requires it. Most production systems use a hybrid: workflows for the overall structure, agent autonomy within specific steps.
