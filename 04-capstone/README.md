# Capstone Project

## Objective

Build an AI agent that can search the web, read documents, and answer questions with citations.

## Requirements

Your agent must:

1. **Search the web** -- Given a query, search for relevant information using a web search API.
2. **Read documents** -- Accept a URL or uploaded document, extract text, and use it as context.
3. **Answer with citations** -- Every factual claim must reference a specific source (URL or document section).
4. **Handle multi-step queries** -- "Compare X and Y" requires searching for X, searching for Y, then synthesizing.

## Architecture

```
User Query
    |
    v
[Agent] -- decides what to do
    |
    +--> [Search Tool] -- searches the web
    +--> [Document Tool] -- reads/extracts from documents
    +--> [Synthesize] -- combines results into cited response
    |
    v
Response with citations
```

## Tools to Implement

### Web Search Tool
```python
@tool
def search_web(query: str, num_results: int = 5) -> str:
    """Search the web. Returns results with titles, URLs, and snippets."""
    # Use any search API: Tavily, Serper, Brave Search, etc.
    pass
```

### Document Reader Tool
```python
@tool
def read_document(url: str) -> str:
    """Fetch and extract text content from a URL."""
    # Fetch the page, extract text (use BeautifulSoup, readability, etc.)
    pass
```

### Citation Formatter
```python
@tool
def format_citation(source_url: str, title: str, relevant_text: str) -> str:
    """Format a citation for a specific claim."""
    return f"[{title}]({source_url}): \"{relevant_text[:200]}...\""
```

## System Prompt

```python
SYSTEM_PROMPT = """You are a research agent. Your job is to find accurate information and present it with citations.

Rules:
1. Before answering any factual question, search for current information.
2. Every factual claim must include a citation with the source URL.
3. If sources conflict, present both perspectives with citations.
4. If you cannot find reliable information, say so. Do not fabricate sources.
5. When comparing topics, search for each independently before synthesizing.

Format citations as: [Source Title](URL)
"""
```

## Evaluation Criteria

| Criterion | Measure | Target |
|-----------|---------|--------|
| Citation accuracy | Do cited sources actually support the claim? | > 90% |
| Completeness | Does the answer address the full question? | > 85% |
| Hallucination rate | Claims without valid sources | < 5% |
| Cost per query | Average tokens per research query | < 5000 |
| Latency | Time to first response | < 30 seconds |

## Test Queries

Run these through your agent to verify it works:

1. "What are the latest developments in AI agent frameworks in 2026?" -- Tests web search and recency.
2. "Compare LangGraph and CrewAI" -- Tests multi-step search and synthesis.
3. "Summarize this article: [URL]" -- Tests document reading.
4. "What is the population of Tokyo? Cite your source." -- Tests citation requirement.
5. "What is the meaning of life?" -- Tests graceful handling of unanswerable questions (should not fabricate sources).

## Stretch Goals

- **RAG integration**: Index documents into a vector store and retrieve relevant chunks.
- **Multi-agent**: Separate researcher and writer agents with an orchestrator.
- **Streaming**: Stream the agent's response as it works.
- **Evaluation harness**: Automated eval suite that runs test queries and scores results.
- **Deployment**: Deploy as an API with rate limiting and cost tracking.

## What You Should Have Learned

By completing this project, you have demonstrated:
- Building an agent with the observe-think-act loop
- Designing and implementing tools with clear schemas
- Integrating RAG for document-based retrieval
- Enforcing structured output with citations
- Evaluating agent performance with measurable criteria
