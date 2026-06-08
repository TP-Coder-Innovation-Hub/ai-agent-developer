`[Senior]`

# Deployment

## Wrapping an Agent in an API

Agents run in a loop. Users expect a response. The standard pattern: wrap the agent in a web API that accepts requests and returns responses.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid

app = FastAPI()

class AgentRequest(BaseModel):
    message: str
    conversation_id: str | None = None

class AgentResponse(BaseModel):
    response: str
    conversation_id: str
    tokens_used: int

conversations: dict[str, list[dict]] = {}

@app.post("/agent", response_model=AgentResponse)
def run_agent_api(request: AgentRequest):
    conv_id = request.conversation_id or str(uuid.uuid4())
    messages = conversations.get(conv_id, [
        {"role": "system", "content": "You are a helpful assistant with tool access."}
    ])
    messages.append({"role": "user", "content": request.message})

    total_tokens = 0
    for _ in range(10):
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, tools=tools, temperature=0.0
        )
        total_tokens += response.usage.total_tokens if response.usage else 0
        assistant_msg = response.choices[0].message
        messages.append(assistant_msg)

        if assistant_msg.tool_calls:
            for tc in assistant_msg.tool_calls:
                result = TOOL_MAP[tc.function.name](**json.loads(tc.function.arguments))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
        else:
            conversations[conv_id] = messages
            return AgentResponse(response=assistant_msg.content, conversation_id=conv_id, tokens_used=total_tokens)

    raise HTTPException(status_code=504, detail="Agent timeout: max iterations reached")
```

## Streaming Responses

Agent loops take time. Users stare at a spinner. Streaming lets the user see the response as it is generated.

```python
from fastapi.responses import StreamingResponse

@app.post("/agent/stream")
def stream_agent(request: AgentRequest):
    def generate():
        messages = [{"role": "user", "content": request.message}]

        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, tools=tools, stream=True, temperature=0.0
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"

            if chunk.choices[0].delta.tool_calls:
                # Handle tool calls in streaming mode
                yield f"data: {json.dumps({'status': 'calling_tool'})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

## Rate Limiting

LLM APIs have rate limits. Your application needs its own rate limiting to prevent abuse and manage costs.

```python
import time
from collections import defaultdict
from fastapi import Request, HTTPException

class RateLimiter:
    def __init__(self, requests_per_minute: int = 20, tokens_per_day: int = 100_000):
        self.rpm = requests_per_minute
        self.tpd = tokens_per_day
        self.requests: dict[str, list[float]] = defaultdict(list)
        self.token_usage: dict[str, int] = defaultdict(int)

    def check(self, user_id: str):
        now = time.time()
        # Clean old entries
        self.requests[user_id] = [t for t in self.requests[user_id] if now - t < 60]

        if len(self.requests[user_id]) >= self.rpm:
            raise HTTPException(status_code=429, detail="Rate limit: too many requests")
        if self.token_usage[user_id] >= self.tpd:
            raise HTTPException(status_code=429, detail="Rate limit: daily token budget exceeded")

        self.requests[user_id].append(now)

    def record_tokens(self, user_id: str, tokens: int):
        self.token_usage[user_id] += tokens

limiter = RateLimiter(requests_per_minute=10)

@app.post("/agent")
def run_agent_limited(request: AgentRequest, user_id: str = "default"):
    limiter.check(user_id)
    result = run_agent_api(request)
    limiter.record_tokens(user_id, result.tokens_used)
    return result
```

## Error Handling

Agents fail in many ways. Handle each explicitly.

```python
from openai import APIError, RateLimitError, APITimeoutError
import logging

logger = logging.getLogger("agent")

def safe_agent_run(request: AgentRequest) -> AgentResponse:
    try:
        return run_agent_api(request)

    except RateLimitError:
        logger.warning("LLM rate limit hit")
        raise HTTPException(status_code=503, detail="Service busy. Try again in a moment.")

    except APITimeoutError:
        logger.error("LLM request timeout")
        raise HTTPException(status_code=504, detail="Request timeout. Try a simpler query.")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse tool arguments: {e}")
        raise HTTPException(status_code=500, detail="Internal error: malformed tool call.")

    except Exception as e:
        logger.exception(f"Unexpected agent error: {e}")
        raise HTTPException(status_code=500, detail="Internal error.")

@app.exception_handler(HTTPException)
async def custom_handler(request: Request, exc: HTTPException):
    return {"error": exc.detail, "status_code": exc.status_code}
```

## Health Checks and Readiness

```python
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    try:
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return {"status": "ready"}
    except Exception:
        raise HTTPException(status_code=503, detail="LLM API unavailable")
```

## Production Checklist

- [ ] Rate limiting per user (prevent abuse and cost overruns)
- [ ] Timeout on agent loop (prevent infinite runs)
- [ ] Error handling for LLM API failures (rate limits, timeouts, malformed responses)
- [ ] Token usage tracking per request and per user
- [ ] Conversation persistence (users can resume)
- [ ] Logging for debugging (do not log user PII)
- [ ] Health check endpoint for load balancers
- [ ] Input validation (reject empty messages, enforce max length)
- [ ] CORS configuration if the API is called from a browser
- [ ] Authentication (API key, JWT, or session-based)
