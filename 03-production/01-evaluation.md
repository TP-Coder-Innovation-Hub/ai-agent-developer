`[Senior]`

# Evaluation

## Why Evaluation is Non-Negotiable

Agents are nondeterministic. The same input can produce different outputs on different runs. You cannot test them with exact-match assertions the way you test regular code. Without evaluation, you have no way to know if your agent works, if a change improved it, or if it is getting worse over time.

Evaluation is to agents what testing is to software. Build the eval suite before (or alongside) the agent.

## Metrics

### Task Completion Rate

The percentage of tasks the agent completes successfully. This is the most important metric.

```python
def evaluate_task_completion(
    test_cases: list[dict],
    agent_fn: callable,
) -> dict:
    """Run test cases through the agent and measure success."""
    results = {"passed": 0, "failed": 0, "errors": [], "total": len(test_cases)}

    for case in test_cases:
        try:
            output = agent_fn(case["input"])

            if case.get("exact_match"):
                success = output.strip() == case["expected"].strip()
            elif case.get("contains"):
                success = all(s.lower() in output.lower() for s in case["contains"])
            elif case.get("eval_fn"):
                success = case["eval_fn"](output, case["expected"])
            else:
                success = output.strip() == case["expected"].strip()

            if success:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["errors"].append({
                    "input": case["input"],
                    "expected": case["expected"],
                    "got": output,
                })
        except Exception as e:
            results["failed"] += 1
            results["errors"].append({"input": case["input"], "error": str(e)})

    results["rate"] = results["passed"] / results["total"]
    return results

# Example test cases
test_cases = [
    {
        "input": "What is the weather in Tokyo?",
        "contains": ["Tokyo", "temperature"],  # response must contain these
    },
    {
        "input": "Calculate 15% tip on $80",
        "contains": ["12"],  # 80 * 0.15 = 12
    },
    {
        "input": "Compare weather in Tokyo and London",
        "eval_fn": lambda output, _: "Tokyo" in output and "London" in output,
    },
]
```

### Cost per Task

How many tokens did the agent consume to complete the task?

```python
def track_cost(agent_fn: callable) -> callable:
    """Decorator that tracks token usage and cost."""
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        total_tokens = 0
        total_cost = 0.0

        # Patch the OpenAI client to track usage
        original_create = client.chat.completions.create
        def tracked_create(**kwargs):
            resp = original_create(**kwargs)
            if resp.usage:
                nonlocal total_tokens, total_cost
                total_tokens += resp.usage.total_tokens
                total_cost += calculate_cost(resp.model, resp.usage)
            return resp
        client.chat.completions.create = tracked_create

        try:
            result = agent_fn(*args, **kwargs)
        finally:
            client.chat.completions.create = original_create

        return result, {"tokens": total_tokens, "cost": total_cost, "time": time.time() - start_time}
    return wrapper
```

### Latency

Time from user input to agent response. Critical for user experience.

```python
import time

def measure_latency(agent_fn: callable, queries: list[str]) -> dict:
    latencies = []
    for query in queries:
        start = time.time()
        agent_fn(query)
        latencies.append(time.time() - start)

    return {
        "mean": sum(latencies) / len(latencies),
        "p50": sorted(latencies)[len(latencies) // 2],
        "p95": sorted(latencies)[int(len(latencies) * 0.95)],
        "max": max(latencies),
    }
```

## LLM-as-Judge

Use an LLM to evaluate the quality of another LLM's output. This is the standard approach for evaluating open-ended responses.

```python
def llm_judge(query: str, response: str, criteria: str) -> dict:
    """Use an LLM to judge the quality of a response."""
    judge_prompt = f"""Evaluate this agent response on a scale of 1-5.

Query: {query}
Response: {response}
Criteria: {criteria}

Rate on:
1. Accuracy: Is the information correct?
2. Completeness: Does it fully address the query?
3. Clarity: Is it well-organized and easy to understand?
4. Actionability: Can the user act on this information?

Return JSON: {{"score": 1-5, "reasoning": "brief explanation", "issues": ["list of issues"]}}"""

    judge_response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
    )

    return json.loads(judge_response.choices[0].message.content)

# Usage
quality = llm_judge(
    query="What is RAG?",
    response="RAG is a technique that combines retrieval with generation.",
    criteria="Technical accuracy for a software engineering audience.",
)
print(quality["score"])  # 4
print(quality["issues"]) # ["Could include more detail about the retrieval step"]
```

## Building an Eval Suite

```python
class EvalSuite:
    def __init__(self, name: str):
        self.name = name
        self.test_cases = []

    def add(self, input: str, expected: str = None, contains: list = None, eval_fn = None):
        self.test_cases.append({
            "input": input, "expected": expected,
            "contains": contains, "eval_fn": eval_fn,
        })

    def run(self, agent_fn: callable) -> dict:
        completion = evaluate_task_completion(self.test_cases, agent_fn)
        latency = measure_latency(agent_fn, [tc["input"] for tc in self.test_cases[:5]])
        return {"name": self.name, **completion, "latency": latency}

# Define your eval suite once, run it on every change
suite = EvalSuite("weather-agent-v1")
suite.add("What's the weather in Tokyo?", contains=["Tokyo"])
suite.add("Calculate 20% of 350", contains=["70"])
suite.add("Should I bring a jacket to London?", contains=["London"])

# Run after every change
results = suite.run(my_agent)
print(f"Completion rate: {results['rate']:.0%}")
```

Evaluation is not a one-time task. Run your eval suite on every change to the agent's prompt, tools, or configuration. Track results over time. If the completion rate drops, you know exactly when and why.
