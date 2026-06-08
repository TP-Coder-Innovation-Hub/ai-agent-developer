# AGENTS.md

## Context

This repository contains the "AI Agent Developer Fundamentals" learning path -- a guide for software engineers entering the AI agent ecosystem in 2026. The audience is competent developers who are new to agent development, not new to programming. They know what APIs are, they understand state management, and they have shipped software. They need to understand how those skills transfer to building with LLMs.

## Audience

- Software engineers with 2+ years of experience
- Comfortable with Python, API design, and system architecture
- New to AI/ML and LLM-based development
- Learning path progression: Entry -> Mid -> Senior concepts
- Looking for engineering rigor, not hype or tutorials

## How to Help

**Connect to what they already know.** Learners here are not beginners. When explaining an agent concept, map it to a software engineering pattern:
- Tool definitions = API contracts
- Agent loop = event loop / request-response cycle
- Context window management = memory management
- RAG indexing = database indexing strategies
- Multi-agent orchestration = microservice choreography
- MCP/A2A = HTTP/REST for agents

**Explain agent reasoning, not just API calls.** Do not just show how to call the OpenAI function calling API. Explain *why* the LLM selects a particular tool, what makes a good tool description, and how the agent decides to stop calling tools and respond. The goal is conceptual understanding, not copy-paste code.

**Show the failure modes.** Every pattern in this guide has failure modes. Tool calls return malformed JSON. RAG retrieves irrelevant chunks. Multi-agent systems deadlock. Show what goes wrong and how to handle it. Production-grade content means acknowledging imperfection.

**Provide runnable code, not pseudocode.** All code examples should be complete enough to understand and adapt. Include imports, error handling, and type annotations where they clarify intent. Python is the primary language.

**Reference the source material.** When discussing architectural decisions, reference Anthropic's "Building Effective Agents" paper. When discussing MCP, reference the specification. Learners should know where to go deeper.

## How NOT to Help

- **Do not treat AI as magic.** LLMs are statistical models with well-understood limitations. Present them as engineering components with defined interfaces, failure modes, and operating constraints. Avoid language like "the AI understands" or "the AI thinks." Prefer "the model predicts" or "the LLM generates."
- **Do not skip evaluation.** Every agent pattern should include a discussion of how to evaluate it. Building an agent without an evaluation strategy is building without tests. This is non-negotiable for production content.
- **Do not build agents when a simple API call suffices.** If the task is deterministic, use a script. If it requires a single LLM call with no tool use, call the API directly. Agent frameworks add complexity, latency, and cost. Advocate for the simplest solution that works.
- **Do not present framework-specific patterns as universal.** LangGraph, CrewAI, ADK, and Anthropic Agent SDK each have their strengths. Present tradeoffs honestly. Avoid "just use X" answers.
- **Do not ignore cost.** Token costs are a real production concern. Every agent design decision has cost implications. Mention them.
- **Do not use emojis or exclamation marks.** The tone is professional and direct.

## Key Concepts

Learners should internalize these seven concepts by the end of the guide:

1. **Agent vs Workflow** -- the most important architectural decision. Autonomous agents trade predictability for flexibility. Workflows trade flexibility for reliability. Start with workflows; add agent autonomy only where justified.

2. **The Agent Loop** -- think, act, observe, repeat. The LLM reasons about what to do next, calls a tool or responds, processes the result, and decides again. This is the fundamental execution model.

3. **Tool Design as API Design** -- tools are APIs for LLMs. Clear descriptions, structured input/output schemas, single responsibility, and proper error handling determine whether an agent can use a tool reliably.

4. **RAG as External Memory** -- when the LLM needs knowledge beyond its training data, RAG provides a retrieval layer. It is not a silver bullet; chunking, embedding, and ranking quality all matter.

5. **Structured Output** -- JSON mode and function calling turned LLMs from text generators into programmable components. Structured output is the foundation that makes agents reliable.

6. **Protocols Enable Ecosystems** -- MCP (tool interface) and A2A (agent communication) are to agents what HTTP is to the web. Standards enable interoperability, not just within one framework, but across the ecosystem.

7. **Evaluation is Non-Negotiable** -- agents are nondeterministic. You cannot test them with exact-match assertions. LLM-as-judge, regression suites, and continuous evaluation are production requirements.

## AI Agent Guidelines 2026

When helping learners build agents, apply these guidelines:

1. **MCP for tool definitions.** New tools should be implemented as MCP servers. This ensures compatibility with any MCP-compliant agent framework and future-proofs the integration.

2. **Structured outputs (JSON mode).** Always use structured output when the LLM's response is consumed by code. Freeform text is acceptable for user-facing responses only.

3. **LangGraph or ADK for orchestration.** For multi-step agents, use a graph-based orchestration framework. LangGraph for framework-agnostic deployments. ADK for Google Cloud integration. Avoid hand-rolled agent loops in production.

4. **Vector stores for RAG.** Use a dedicated vector store (Pinecone, Weaviate, Qdrant, pgvector) for retrieval. Do not implement similarity search from scratch.

5. **Evaluation-first approach.** Before building an agent, define what success looks like. Build the evaluation harness before (or alongside) the agent. LLM-as-judge for quality, regression tests for tool call correctness.

6. **Token cost awareness.** Track token usage per request. Set budgets per agent run. Use prompt caching. Route simple queries to cheaper models. Cost optimization is a feature, not an afterthought.

7. **Never trust LLM output without validation.** Every LLM output -- text responses, tool call arguments, structured JSON -- must be validated before use. The model will eventually produce incorrect, malformed, or unexpected output. Validation is not optional.

## Repository Structure

```
ai-agent-developer/
├── README.md
├── AGENTS.md
├── 00-foundations/
│   ├── 01-what-are-ai-agents.md
│   ├── 02-llm-fundamentals.md
│   ├── 03-prompting-techniques.md
│   └── 04-ai-safety-and-responsibility.md
├── 01-agent-architecture/
│   ├── 01-agent-loop.md
│   ├── 02-tools-and-functions.md
│   ├── 03-memory.md
│   ├── 04-planning.md
│   └── 05-multi-agent-systems.md
├── 02-building-agents/
│   ├── 01-first-agent.md
│   ├── 02-rag-basics.md
│   ├── 03-structured-outputs.md
│   └── 04-agent-frameworks.md
├── 03-production/
│   ├── 01-evaluation.md
│   ├── 02-observability.md
│   ├── 03-cost-optimization.md
│   └── 04-deployment.md
└── 04-capstone/
    └── README.md
```
