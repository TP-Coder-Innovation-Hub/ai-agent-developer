# Cost Optimization

## Token Economics

You pay per token. Every input token and every output token costs money. In an agent system, costs multiply because each agent loop iteration makes an LLM call.

```python
# Approximate pricing (2026, check current rates)
PRICING = {
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "claude-sonnet-4": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
}

def calculate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    rates = PRICING.get(model, PRICING["gpt-4o"])
    return (tokens_in * rates["input"]) + (tokens_out * rates["output"])

# Example: 5-step agent run on gpt-4o
# Each step: ~1000 input tokens (growing context), ~200 output tokens
# Step 1: 1000 in + 200 out = $0.0045
# Step 2: 1200 in + 200 out = $0.0050
# Step 3: 1400 in + 200 out = $0.0055
# Step 4: 1600 in + 200 out = $0.0060
# Step 5: 1800 in + 200 out = $0.0065
# Total: ~$0.028 per query
# At 1000 queries/day: $28/day = $840/month

# Same agent on gpt-4o-mini: ~$0.0025 per query = $75/month (10x cheaper)
```

The numbers look small per query. They compound at scale. A poorly optimized agent can cost 10-100x more than necessary.

## Strategy 1: Route to Cheaper Models

Not every step needs GPT-4o. Use a router to classify query complexity and route accordingly.

```python
def route_query(query: str) -> str:
    """Route to the appropriate model based on query complexity."""
    complexity_check = client.chat.completions.create(
        model="gpt-4o-mini",  # Use the cheap model for routing
        response_format={"type": "json_object"},
        messages=[{
            "role": "system",
            "content": """Classify query complexity.
{"complexity": "simple" | "medium" | "complex"}
simple: factual lookup, single tool call, basic math
medium: multi-step reasoning, 2-3 tool calls
complex: multi-step planning, analysis, creative tasks"""
        }, {
            "role": "user",
            "content": query
        }],
        temperature=0.0,
        max_tokens=20,
    )

    import json
    complexity = json.loads(complexity_check.choices[0].message.content)["complexity"]

    model_map = {
        "simple": "gpt-4o-mini",   # $0.15/$0.60 per M tokens
        "medium": "gpt-4o",        # $2.50/$10.00 per M tokens
        "complex": "gpt-4o",       # $2.50/$10.00 per M tokens
    }
    return model_map[complexity]
```

The routing call costs almost nothing (20 output tokens on the mini model) and can save 10x on the actual agent call.

## Strategy 2: Caching

Cache LLM responses for identical inputs. Also cache tool results that do not change frequently.

```python
import hashlib
import json
from datetime import datetime, timedelta

class ResponseCache:
    def __init__(self, ttl_minutes: int = 60):
        self.cache: dict[str, tuple[str, datetime]] = {}
        self.ttl = timedelta(minutes=ttl_minutes)

    def _key(self, messages: list[dict], model: str) -> str:
        content = json.dumps(messages, sort_keys=True) + model
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, messages: list[dict], model: str) -> str | None:
        key = self._key(messages, model)
        if key in self.cache:
            response, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return response
            del self.cache[key]
        return None

    def set(self, messages: list[dict], model: str, response: str):
        key = self._key(messages, model)
        self.cache[key] = (response, datetime.now())

cache = ResponseCache(ttl_minutes=30)

def cached_llm_call(messages: list[dict], model: str = "gpt-4o", **kwargs) -> str:
    cached = cache.get(messages, model)
    if cached:
        return cached

    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    result = response.choices[0].message.content
    cache.set(messages, model, result)
    return result
```

OpenAI also provides server-side prompt caching. If your system prompt and conversation prefix are the same across requests, you pay less for input tokens. Enable it by using the same prompt structure consistently.

## Strategy 3: Batch Processing

For non-real-time tasks, use batch APIs. OpenAI's Batch API processes requests asynchronously at 50% cost.

```python
import json

def create_batch_request(requests: list[dict]) -> str:
    """Create a batch file for OpenAI's Batch API."""
    lines = []
    for req in requests:
        lines.append(json.dumps({
            "custom_id": req["id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": req["messages"],
                "temperature": 0.0,
            }
        }))
    return "\n".join(lines)

# Upload batch file, submit, get results at 50% cost
# Use for: eval suites, bulk processing, offline analysis
```

## Strategy 4: Reduce Context Size

The biggest cost driver in agent systems is the growing conversation context. Each iteration adds the previous context plus new tool results.

```python
def optimize_context(messages: list[dict], max_tokens: int = 4000) -> list[dict]:
    """Reduce context size by summarizing old turns."""
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o")

    total = sum(len(enc.encode(m.get("content", ""))) for m in messages)
    if total <= max_tokens:
        return messages

    # Keep system prompt + last 2 turns, summarize the rest
    system = [m for m in messages if m["role"] == "system"]
    recent = [m for m in messages if m["role"] != "system"][-4:]
    old = [m for m in messages if m["role"] != "system"][:-4]

    if old:
        summary = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize concisely, preserving key facts."},
                {"role": "user", "content": str(old)}
            ]
        ).choices[0].message.content
        return system + [{"role": "system", "content": f"Summary: {summary}"}] + recent

    return system + recent
```

## Monitoring Costs

Track cost per agent run, per tool, and per user. Set budgets and alert when thresholds are exceeded.

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class CostTracker:
    daily_budget: float = 100.0
    spent: float = 0.0
    runs: list[dict] = field(default_factory=list)

    def record(self, user_id: str, tokens_in: int, tokens_out: int, model: str):
        cost = calculate_cost(model, tokens_in, tokens_out)
        self.spent += cost
        self.runs.append({
            "user_id": user_id, "cost": cost,
            "model": model, "timestamp": datetime.now().isoformat()
        })
        if self.spent > self.daily_budget:
            raise RuntimeError(f"Daily budget exceeded: ${self.spent:.2f} / ${self.daily_budget:.2f}")
```
