`[Entry]`

# LLM Fundamentals

## How LLMs Work

An LLM is a statistical model trained to predict the next token. Given a sequence of tokens, it outputs a probability distribution over possible next tokens. Sampling from that distribution generates text.

```python
# Simplified: what the model actually does
input_tokens = ["The", "weather", "in", "Tokyo", "is"]
next_token_probs = model.predict(input_tokens)
# {"sunny": 0.31, "warm": 0.22, "rainy": 0.18, "cold": 0.12, ...}
next_token = sample(next_token_probs, temperature=0.7)
# "sunny" (probably)
```

There is no understanding. No knowledge. No reasoning. The model has learned statistical patterns from training data and applies those patterns to predict likely continuations. This is surprisingly useful, but it is important to be precise about what is happening.

## Tokens

Text is split into tokens before processing. A token is roughly 4 characters in English, but varies:

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")
tokens = enc.encode("Hello, world!")
print(len(tokens))  # 4 tokens: ["Hello", ",", " world", "!"]
print(enc.decode(tokens))  # "Hello, world!"
```

Token count determines cost (you pay per token) and context window limits (each model has a maximum). Count tokens before sending requests.

## Context Window

The context window is the maximum number of tokens the model can process in a single request (input + output combined).

```python
# Model context windows (approximate, 2026)
CONTEXT_WINDOWS = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "claude-sonnet-4": 200_000,
    "gemini-2.0-flash": 1_000_000,
}
```

Context window is a hard constraint. If your input exceeds it, the request fails. If your input is close to the limit, output gets truncated. For agents, this means: manage your conversation history, summarize old turns, and be strategic about what you include in each request.

## Temperature and Sampling

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a haiku about code"}],
    temperature=0.0,  # deterministic: always pick the most likely token
    # temperature=1.0  # default: some randomness
    # temperature=2.0  # creative: more randomness, less coherence
)
```

- `temperature=0`: Deterministic output. Use for structured tasks, tool calling, code generation.
- `temperature=0.7`: Balanced. Good for general conversation.
- `temperature=1.5+`: High creativity, low coherence. Use sparingly.

`top_p` (nucleus sampling) is an alternative to temperature. It restricts sampling to tokens whose cumulative probability exceeds `top_p`. Use one or the other, not both.

## Completion vs Chat

Two API paradigms:

```python
# Completion (legacy): text in, text out
response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="Translate to French: Hello"
)

# Chat (modern): messages in, messages out
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a translator."},
        {"role": "user", "content": "Translate to French: Hello"},
    ]
)
```

Use the chat API. Completion is deprecated for most models. The chat API provides a structured message format that maps cleanly to agent patterns.

## Message Roles

```python
messages = [
    # System: set behavior, personality, constraints
    {"role": "system", "content": "You are a helpful assistant that answers concisely."},

    # User: the human's input
    {"role": "user", "content": "What is 2+2?"},

    # Assistant: the model's response (stored for multi-turn)
    {"role": "assistant", "content": "4"},

    # User: follow-up (model sees full history)
    {"role": "user", "content": "And 2+3?"},
]
```

In agent systems, the messages array is the primary state. Tools add `tool` role messages. The agent manages the conversation history by appending to this array. Think of it as a log of the interaction.
