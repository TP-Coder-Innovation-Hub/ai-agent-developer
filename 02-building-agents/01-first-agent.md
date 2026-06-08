`[Mid]`

# Your First Agent

A complete, runnable agent using the OpenAI API with function calling. Every line explained.

## The Complete Agent

```python
import json
from openai import OpenAI

client = OpenAI()

# 1. DEFINE TOOLS ----------------------------------------------------------
# Each tool is a JSON schema describing a function the LLM can call.

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city. Returns temperature, conditions, and humidity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Tokyo', 'New York'"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression. Use for any calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. '72 * 1.1 + 5'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# 2. IMPLEMENT TOOLS --------------------------------------------------------
# The actual Python functions that execute when the LLM calls a tool.

def get_weather(city: str) -> str:
    """In production, call a real weather API. Here we simulate."""
    weather_db = {
        "tokyo": "18C, partly cloudy, humidity 65%",
        "new york": "22C, sunny, humidity 45%",
        "london": "12C, rainy, humidity 85%",
        "paris": "15C, overcast, humidity 70%",
    }
    result = weather_db.get(city.lower())
    if result:
        return f"Weather in {city}: {result}"
    return f"No weather data available for {city}"

def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    import ast
    import operator
    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg,
    }
    try:
        node = ast.parse(expression, mode='eval')
        def eval_node(n):
            if isinstance(n, ast.Constant): return n.value
            if isinstance(n, ast.BinOp): return ops[type(n.op)](eval_node(n.left), eval_node(n.right))
            if isinstance(n, ast.UnaryOp): return ops[type(n.op)](eval_node(n.operand))
            raise ValueError(f"Unsupported: {type(n)}")
        return str(eval_node(node.body))
    except Exception as e:
        return f"Error: {e}"

# Map tool names to functions
TOOL_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
}

# 3. THE AGENT LOOP --------------------------------------------------------

def run_agent(user_message: str, max_iterations: int = 10) -> str:
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant with access to weather data and a calculator.
Use tools when they are relevant. For math questions, always use the calculator tool.
For weather questions, always use the weather tool. Be concise."""
        },
        {"role": "user", "content": user_message},
    ]

    for _ in range(max_iterations):
        # Send the full conversation to the LLM
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            temperature=0.0,
        )

        choice = response.choices[0]
        assistant_msg = choice.message
        messages.append(assistant_msg)

        # Check if the model wants to call tools
        if assistant_msg.tool_calls:
            for tool_call in assistant_msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                print(f"  [Tool call] {fn_name}({fn_args})")

                # Execute the tool
                result = TOOL_MAP[fn_name](**fn_args)

                # Add the tool result to the conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        else:
            # No tool call: the agent is done
            return assistant_msg.content

    return "Agent reached maximum iterations."

# 4. RUN IT ----------------------------------------------------------------

if __name__ == "__main__":
    queries = [
        "What is the weather in Tokyo?",
        "If it's 18C in Tokyo, what is that in Fahrenheit?",
        "What's the weather difference between Tokyo and London?",
    ]

    for q in queries:
        print(f"\nUser: {q}")
        answer = run_agent(q)
        print(f"Agent: {answer}\n")
```

## Step by Step Explanation

**Step 1 -- Define tools**: The `tools` list tells the LLM what functions exist, what they do, and what parameters they accept. This is the API contract. The LLM reads these schemas to decide which tool to call.

**Step 2 -- Implement tools**: `get_weather` and `calculate` are regular Python functions. They take the arguments defined in the schema and return strings. Strings because tool results get added to the messages array, which is text-based.

**Step 3 -- The agent loop**: The core loop from [01-agent-loop](../01-agent-architecture/01-agent-loop.md). Each iteration: send messages to LLM, check if it called a tool, execute the tool, add the result, repeat. If no tool call, the agent is done.

**Step 4 -- Run**: Pass a user message, get a response. The agent decides whether to use tools based on the query.

## What Happens at Runtime

For the query "What is the weather in Tokyo?":

1. LLM sees the message and tool schemas. It decides to call `get_weather(city="Tokyo")`.
2. Agent executes `get_weather("Tokyo")`, returns "Weather in Tokyo: 18C, partly cloudy, humidity 65%".
3. LLM sees the tool result and generates a natural language response.
4. No more tool calls. Loop ends. Response returned.

For "If it's 18C in Tokyo, what is that in Fahrenheit?":

1. LLM calls `calculate(expression="18 * 9/5 + 32")`.
2. Agent returns "64.4".
3. LLM responds: "18C is 64.4F."

The LLM chose the right tool for each query. It read the tool descriptions, understood the user's intent, and matched them. This is agent reasoning.
