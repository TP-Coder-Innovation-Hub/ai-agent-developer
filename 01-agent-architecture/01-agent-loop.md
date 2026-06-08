# The Agent Loop

## The Core Loop: Observe, Think, Act

The agent loop is the fundamental execution model. Without it, an LLM is a stateless function: text in, text out. With it, the LLM becomes an agent that can interact with the world.

> 🖼️ **[IMAGE_PLACEHOLDER]** — agent loop observe think act tool call respond cycle diagram

```python
import json
from openai import OpenAI

client = OpenAI()

def agent_loop(system_prompt: str, user_message: str, tools: list, max_iterations: int = 10):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    for i in range(max_iterations):
        # 1. OBSERVE: Send current state to the LLM
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )

        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        # 2. THINK: Did the model want to use a tool or respond?
        if assistant_message.tool_calls:
            # 3. ACT: Execute each tool call
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                result = execute_tool(function_name, function_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })
        else:
            return assistant_message.content

    return "Agent exceeded maximum iterations without completing the task."
```

## How It Works Step by Step

1. **Observe**: The full conversation history (user messages, previous tool results, system prompt) is sent to the LLM. The LLM sees everything.

2. **Think**: The LLM decides what to do next. It can call a tool or respond directly. This decision is the model's reasoning step.

3. **Act**: If the LLM chose a tool, the agent executes it and adds the result to the conversation. If the LLM chose to respond, the loop ends.

4. **Repeat**: The agent sends the updated conversation (now including the tool result) back to the LLM. The LLM sees the result and decides what to do next.

## The ReAct Pattern

ReAct (Reason + Act) is the most common agent pattern. The model explicitly reasons before acting:

```
User: What is the population of the capital of France?

Agent reasoning: The user wants the population of the capital of France.
France's capital is Paris. I need to look up Paris's current population.

Agent action: call search("Paris population 2026")

Tool result: "Paris population: 2,161,000 (city), 12.4 million (metro area)"

Agent reasoning: I have the data. I can answer now.

Agent response: "Paris has a population of approximately 2.16 million within
the city limits, and 12.4 million in the metropolitan area."
```

The reasoning is not always visible in the output (some implementations hide it), but the pattern is the same: reason about what to do, do it, observe the result, reason again.

## Why the Loop Matters

LLMs alone cannot act. They generate text. The agent loop provides:

- **Action capability**: The loop connects the LLM to tools that can affect the real world.
- **State management**: The messages array accumulates context across iterations.
- **Termination**: The loop provides a stopping condition (no tool call = done) and a safety limit (max iterations).
- **Error recovery**: If a tool fails, the result is added to the context and the LLM can decide how to handle it.

## Failure Modes

- **Infinite loops**: The agent calls tools repeatedly without converging. Mitigate with `max_iterations`.
- **Hallucinated tool calls**: The model generates tool names or arguments that do not exist. Mitigate with schema validation.
- **Context overflow**: Long tool results fill the context window. Mitigate by summarizing or truncating tool output.
- **Premature termination**: The agent responds before gathering sufficient information. Mitigate with explicit system instructions about when to respond vs. when to keep searching.
