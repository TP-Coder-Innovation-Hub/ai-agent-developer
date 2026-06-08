# Memory

## Why Agents Need Memory

An agent without memory forgets everything between interactions. It cannot remember what the user said two turns ago, what tools it already called, or what it learned in previous sessions. Memory gives the agent context.

There are three types of memory in agent systems. Each solves a different problem.

> 🖼️ **[IMAGE_PLACEHOLDER]** — agent memory types short-term conversation long-term vector working task state

## Short-Term Memory (Conversation History)

This is the `messages` array. It contains the current conversation: user inputs, assistant responses, and tool results. It lives in memory for the duration of a single interaction.

```python
messages = [
    {"role": "system", "content": "You are a research assistant."},
    {"role": "user", "content": "What is RAG?"},
    {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation..."},
    {"role": "user", "content": "How does it differ from fine-tuning?"},  # model sees full history
]
```

Problem: the context window is finite. A long conversation will eventually exceed it.

Solution: manage the conversation history. Common strategies:

```python
def trim_messages(messages: list, max_tokens: int, model: str = "gpt-4o") -> list:
    """Keep system prompt + last N messages that fit within max_tokens."""
    import tiktoken
    enc = tiktoken.encoding_for_model(model)

    system = messages[0]  # always keep
    conversation = messages[1:]

    trimmed = []
    total = len(enc.encode(system["content"]))

    for msg in reversed(conversation):
        msg_tokens = len(enc.encode(msg.get("content", "")))
        if total + msg_tokens > max_tokens:
            break
        trimmed.insert(0, msg)
        total += msg_tokens

    return [system] + trimmed

def summarize_old_messages(messages: list, client) -> list:
    """Summarize old messages instead of dropping them."""
    if len(messages) <= 6:
        return messages

    system = messages[0]
    old = messages[1:-4]  # everything except last 2 turns
    recent = messages[-4:]

    summary = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize this conversation concisely, preserving key facts and decisions."},
            {"role": "user", "content": str(old)}
        ]
    ).choices[0].message.content

    return [
        system,
        {"role": "system", "content": f"Previous conversation summary: {summary}"},
        *recent,
    ]
```

## Long-Term Memory (Vector Store)

Conversation history disappears when the session ends. Long-term memory persists across sessions. The standard implementation: store information as embeddings in a vector database, retrieve relevant chunks when needed.

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def store_fact(user_id: str, fact: str, vector_store):
    """Store a fact in long-term memory."""
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=fact
    ).data[0].embedding

    vector_store.upsert(
        id=f"{user_id}_{hash(fact)}",
        values=embedding,
        metadata={"user_id": user_id, "text": fact}
    )

def recall_facts(user_id: str, query: str, vector_store, top_k: int = 5) -> list[str]:
    """Retrieve relevant facts from long-term memory."""
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    results = vector_store.query(
        vector=query_embedding,
        filter={"user_id": user_id},
        top_k=top_k,
    )

    return [match["metadata"]["text"] for match in results["matches"]]
```

When to store: user preferences, decisions made, facts mentioned, task outcomes. Not everything. Store what the agent will need later.

When to recall: at the start of each new interaction, query long-term memory with the user's message and inject relevant facts into the system prompt.

## Working Memory

Working memory holds the agent's current task state: what it is trying to do, what it has done so far, what remains. This is often implemented as a structured object, not as part of the conversation.

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class WorkingMemory:
    task: str
    goal: str
    completed_steps: list[str] = field(default_factory=list)
    pending_steps: list[str] = field(default_factory=list)
    intermediate_results: dict[str, Any] = field(default_factory=dict)
    tool_call_count: int = 0
    max_tool_calls: int = 20

    def add_result(self, step: str, result: Any):
        self.completed_steps.append(step)
        self.intermediate_results[step] = result

    def to_context(self) -> str:
        return f"""Current task: {self.task}
Goal: {self.goal}
Completed: {', '.join(self.completed_steps) if self.completed_steps else 'None'}
Pending: {', '.join(self.pending_steps) if self.pending_steps else 'None'}
Tool calls used: {self.tool_call_count}/{self.max_tool_calls}"""
```

## Memory Architecture Decision

| Type | Scope | Storage | When to Use |
|------|-------|---------|-------------|
| Short-term | Single conversation | Messages array | Always (required) |
| Long-term | Across sessions | Vector store | User preferences, learned facts |
| Working | Current task | Structured object | Multi-step tasks with progress tracking |

Most agents need short-term memory. Complex agents need working memory. Agents that learn across sessions need long-term memory.
