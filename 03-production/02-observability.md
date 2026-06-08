# Observability

## Why Tracing Matters

An agent makes multiple LLM calls, each with reasoning and tool execution. When something goes wrong, you cannot debug by looking at the final output. You need to see every step: what the LLM thought, what tool it called, what the tool returned, and what the LLM did with that result.

Observability for agents means tracing the full execution path. This is equivalent to distributed tracing in microservices -- each LLM call and tool execution is a span in a trace.

## What to Log

```python
import time
import uuid
from dataclasses import dataclass, field

@dataclass
class Span:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    name: str = ""
    input: str = ""
    output: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

class AgentTracer:
    def __init__(self):
        self.traces: dict[str, list[Span]] = {}

    def start_trace(self, trace_id: str = None) -> str:
        trace_id = trace_id or str(uuid.uuid4())
        self.traces[trace_id] = []
        return trace_id

    def add_span(self, trace_id: str, span: Span):
        span.trace_id = trace_id
        self.traces[trace_id].append(span)

    def get_trace(self, trace_id: str) -> list[Span]:
        return self.traces.get(trace_id, [])

    def print_trace(self, trace_id: str):
        spans = self.get_trace(trace_id)
        print(f"\n=== Trace: {trace_id} ===")
        for span in spans:
            print(f"  [{span.name}] {span.duration_ms:.0f}ms | tokens: {span.tokens_in}in/{span.tokens_out}out")
            if span.input:
                print(f"    input:  {span.input[:100]}...")
            if span.output:
                print(f"    output: {span.output[:100]}...")
```

## Instrumented Agent Loop

Wrap the agent loop with tracing to capture every step.

```python
tracer = AgentTracer()

def traced_agent_loop(user_message: str, max_iterations: int = 10) -> str:
    trace_id = tracer.start_trace()
    messages = [
        {"role": "system", "content": "You are a helpful assistant with tool access."},
        {"role": "user", "content": user_message},
    ]

    # Log the initial user message
    tracer.add_span(trace_id, Span(
        name="user_input", input=user_message, start_time=time.time(), end_time=time.time()
    ))

    for i in range(max_iterations):
        # Log LLM call
        span = Span(name=f"llm_call_{i}", start_time=time.time())
        span.input = str(messages[-1])[:200]

        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, tools=tools, temperature=0.0
        )

        span.end_time = time.time()
        span.tokens_in = response.usage.prompt_tokens if response.usage else 0
        span.tokens_out = response.usage.completion_tokens if response.usage else 0

        assistant_msg = response.choices[0].message
        span.output = (assistant_msg.content or str(assistant_msg.tool_calls))[:200]
        tracer.add_span(trace_id, span)

        messages.append(assistant_msg)

        if assistant_msg.tool_calls:
            for tool_call in assistant_msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = tool_call.function.arguments

                # Log tool execution
                tool_span = Span(
                    name=f"tool_{fn_name}",
                    input=fn_args[:200],
                    start_time=time.time(),
                )

                result = TOOL_MAP[fn_name](**json.loads(fn_args))

                tool_span.end_time = time.time()
                tool_span.output = str(result)[:200]
                tracer.add_span(trace_id, tool_span)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })
        else:
            tracer.print_trace(trace_id)
            return assistant_msg.content

    return "Max iterations reached."
```

## Production Tools

Build your own tracer for learning. Use production tools for real systems.

### LangSmith

LangChain's observability platform. Works with any framework, not just LangChain.

```python
import os
os.environ["LANGSMITH_API_KEY"] = "your-key"
os.environ["LANGSMITH_TRACING"] = "true"

# Every LLM call and tool execution is automatically traced
# View traces at smith.langchain.com
```

### Langfuse

Open-source alternative. Self-host or cloud.

```python
from langfuse.decorators import observe

@observe()
def my_agent(user_message: str) -> str:
    # All LLM calls within this function are traced automatically
    return agent_loop(user_message)
```

### OpenTelemetry

Standard observability protocol. Send traces to any backend (Jaeger, Datadog, Honeycomb).

```python
from opentelemetry import trace
tracer = trace.get_tracer("ai-agent")

def agent_with_otel(user_message: str) -> str:
    with tracer.start_as_current_span("agent.run") as span:
        span.set_attribute("user.message", user_message)
        # ... agent logic
        span.set_attribute("agent.result", result)
        return result
```

## What to Look For in Traces

- **Unexpected tool calls**: The model called a tool it should not have. Check the tool descriptions.
- **Retry loops**: The model called the same tool multiple times with similar arguments. Add better error messages or a retry limit.
- **High token usage per step**: The context is growing too large. Trim or summarize.
- **Long tool execution times**: A tool is slow. Cache results or optimize the tool.
- **Incorrect termination**: The agent stopped too early or too late. Check the system prompt's termination criteria.
