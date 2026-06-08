`[Mid]`

# Structured Outputs

## The Problem

LLMs return strings. Downstream code needs structured data. If you ask for JSON, the model will usually produce valid JSON. "Usually" is not good enough for production.

```python
# Fragile: parse the LLM's string output
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract entities as JSON: 'Apple sued Samsung in California'"}]
)
import json
data = json.loads(response.choices[0].message.content)  # May crash. May be malformed.
```

Structured output solves this by guaranteeing the response conforms to a schema.

## JSON Mode

The simplest form: tell the model to output valid JSON.

```python
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[{
        "role": "user",
        "content": """Extract entities from this text as JSON with keys:
{"companies": [...], "locations": [...], "actions": [...]}

Text: 'Apple sued Samsung in California over patent violations.'"""
    }]
)

data = json.loads(response.choices[0].message.content)
# {"companies": ["Apple", "Samsung"], "locations": ["California"], "actions": ["sued"]}
```

JSON mode guarantees valid JSON. It does NOT guarantee the JSON matches your expected schema. The model might add extra keys or miss required ones.

## Structured Outputs with Pydantic (OpenAI)

OpenAI's structured output feature guarantees the response matches a Pydantic schema.

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

class EntityExtraction(BaseModel):
    companies: list[str]
    locations: list[str]
    actions: list[str]

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": "Apple sued Samsung in California over patent violations."
    }],
    response_format=EntityExtraction,
)

result = response.choices[0].message.parsed
print(result.companies)   # ["Apple", "Samsung"]
print(result.locations)   # ["California"]
print(result.actions)     # ["sued"]
```

The response is guaranteed to be valid `EntityExtraction`. If the model cannot produce valid output, the API returns a refusal. No parsing errors.

## Complex Schemas

Structured outputs handle nested objects, enums, and optional fields.

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AffectedComponent(BaseModel):
    name: str = Field(description="Component name, e.g. 'auth-service', 'payment-gateway'")
    severity: Severity
    error_rate: Optional[float] = Field(None, description="Error rate as decimal, e.g. 0.15 for 15%")

class IncidentReport(BaseModel):
    title: str
    summary: str = Field(description="One paragraph summary of the incident")
    affected_components: list[AffectedComponent]
    root_hypothesis: str
    recommended_actions: list[str] = Field(description="Ordered list of actions to take")
    requires_escalation: bool

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": """Analyze this error log and create an incident report:

2026-03-15 08:23:11 ERROR auth-service: Token validation failed. Error rate: 12%
2026-03-15 08:23:15 WARN payment-gateway: Timeout connecting to auth-service
2026-03-15 08:23:20 ERROR payment-gateway: Payment processing failed. Error rate: 8%
2026-03-15 08:23:25 ERROR auth-service: Database connection pool exhausted"""
    }],
    response_format=IncidentReport,
)

report = response.choices[0].message.parsed
print(report.title)
print(report.requires_escalation)
for component in report.affected_components:
    print(f"  {component.name}: {component.severity.value} ({component.error_rate})")
```

## Why This Matters for Agents

Agents call tools. Tool arguments must be valid JSON with the correct schema. Without structured output, the agent loop crashes when the model produces malformed arguments.

```python
# Without structured output: fragile
tool_call_args = json.loads(assistant_message.tool_calls[0].function.arguments)
# Could be: {"city": "Tokyo"} -- good
# Could be: {"city": "Tokyo", "contry": "Japan"} -- extra key, might cause issues
# Could be: "Tokyo" -- not even a dict, crashes json.loads

# With structured output: reliable
# The tool schema enforces the structure at the API level
```

Structured output is the difference between an agent that works 90% of the time and one that works 99.9% of the time. For production agents, 90% is not acceptable.

## When to Use Each Approach

| Approach | Guarantee | Cost | When to Use |
|----------|-----------|------|-------------|
| Freeform text | None | Baseline | User-facing conversational responses |
| JSON mode | Valid JSON | Baseline | Simple extraction, quick prototyping |
| Structured output (Pydantic) | Valid + schema-conformant | Slightly higher | Tool arguments, production pipelines |

Use structured output for anything that downstream code parses. Use freeform text only for direct user consumption.
