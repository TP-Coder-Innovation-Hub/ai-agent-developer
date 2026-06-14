# Workshop Design: Data Analyst Agent

> Companion to [01-workshop-spec.md](./01-workshop-spec.md). The spec fixes the agent components, tools, and project structure — this document fixes what it leaves open: the frontend↔backend API contract, the audit/guardrail schemas, the tool-call payload shapes, and seed data with *engineered patterns* so demo questions have deterministic answers.

## Design Notes (read first)

1. **The API contract was undefined — here it is (Part 3).** A question takes 5–30 s to answer (multiple LLM calls + SQL + chart). Design: synchronous `POST /api/ask` returning the complete structured answer, with a `trace` array of step timings so the UI can show what happened. Streaming progress over SSE is a documented stretch goal, not the baseline — get the synchronous loop correct first.
2. **Provider-agnostic LLM client.** The spec allows OpenAI, Anthropic, or local. Wrap the provider behind one interface (`complete(messages, tools) -> response, usage`) so the eval suite can compare providers and the guardrail cost-counting reads one `usage` shape. Never scatter provider SDK calls through the agent.
3. **The executor's DB role is the real SQL guardrail.** Keyword blocklists (`DROP`, `DELETE`…) are required by spec and easily fooled. The defense that actually holds: the agent connects as a PostgreSQL role with `SELECT`-only grants on the business schema, with `statement_timeout` and row limits. Implement both layers; trust the role.
4. **Seeded data contains engineered stories.** "Why did revenue drop last week?" only has a correct answer if the data *contains* a drop. The seed generator injects four known patterns (Part 6) that the evaluation suite asserts against. Random data makes accuracy unmeasurable.
5. **Every number in the narrative must be traceable.** The narrator receives only the query results + analysis JSON — never raw schema or prior conversation — and the fact-checker pass verifies each numeric token in the narrative appears in (or derives from) the result set. This is the anti-hallucination design, not a prompt instruction.

---

## Part 1: High-Level Design

### 1.1 Use-Case Diagram

```mermaid
graph LR
    BU(["Business User"])
    EV(["Engineer (evaluation)"])

    subgraph "Data Analyst Agent"
        UC1["Ask a business question"]
        UC2["View answer: narrative + chart + data + SQL"]
        UC3["Follow up / refine question"]
        UC4["Inspect the agent's steps (trace)"]
        UC5["Run evaluation suite"]
        UC6["Review guardrail violation log"]
    end

    BU --> UC1
    BU --> UC2
    BU --> UC3
    BU --> UC4
    EV --> UC5
    EV --> UC6
```

### 1.2 System Context Diagram

```mermaid
graph TB
    BU(["Business user (browser)"])
    LLM["LLM API<br/>(OpenAI / Anthropic / local)"]

    S["Data Analyst Agent"]

    BU -->|"HTTPS: questions, answers, charts"| S
    S -->|"HTTPS: completion + tool-use calls"| LLM
```

The LLM API is the only external dependency — and the only nondeterministic one, which is why the evaluation suite (UC5) exists.

### 1.3 Container Diagram

```mermaid
graph TB
    SPA(["chat-frontend<br/>Vue 3 + Vite"])

    subgraph "Docker Compose Network"
        API["agent-backend (Python 3.11, FastAPI)<br/>orchestrator · planner · sql-gen ·<br/>narrator · guardrails · tools"]
        PG[("PostgreSQL<br/>business schema (read-only role)<br/>+ agent schema (logs)")]
        VS[("ChromaDB<br/>schema docs + sample queries")]
        FS["chart volume<br/>(PNG files)"]
    end

    LLM["LLM API (external)"]

    SPA -->|"REST: /api/ask, /api/charts"| API
    API -->|"completions + function calling"| LLM
    API -->|"SELECT-only role: business schema<br/>read-write: agent schema"| PG
    API -->|"embed + retrieve schema context"| VS
    API -->|"write PNG"| FS
    SPA -->|"GET /api/charts/{file}"| API
```

### 1.4 Activity Diagram — Answering a Question

```mermaid
flowchart TD
    Start([POST /api/ask]) --> G1{Guardrail: token budget<br/>remaining for this question?}
    G1 -->|no| R1([429 COST_LIMIT])
    G1 -->|yes| P["Planner (LLM): intent, entities,<br/>time range, analysis type"]
    P --> RAG["RAG: retrieve relevant schema docs<br/>+ sample queries from ChromaDB"]
    RAG --> SG["SQL Generator (LLM + schema context)"]
    SG --> G2{Guardrail: SELECT-only,<br/>single statement, known tables?}
    G2 -->|blocked| LOG[Log violation severity=HIGH]
    LOG --> R2([Answer: refusal with reason])
    G2 -->|pass| EX["Executor: run on read-only role,<br/>statement_timeout 10s, LIMIT cap"]
    EX -->|SQL error| RETRY{Retries < 3?}
    RETRY -->|yes| SG2["Feed pg error back to LLM,<br/>regenerate SQL"]
    SG2 --> G2
    RETRY -->|no| R3([Answer: 'could not answer' + last error,<br/>logged for eval])
    EX -->|rows| AN["Analyzer: trends, anomalies (2σ),<br/>correlations, period-over-period"]
    AN --> CH["Visualizer: auto-select chart type<br/>(time series→line, compare→bar, share→pie)"]
    CH --> NA["Narrator (LLM): 2-4 paragraphs<br/>from results + analysis ONLY"]
    NA --> FC["Fact check: every number traces<br/>to results; PII scan + mask"]
    FC --> R4(["200: narrative + chart + data + SQL + trace"])
```

### 1.5 Sequence Diagrams

#### 1.5.1 Happy path — "Why did revenue drop last week?"

```mermaid
sequenceDiagram
    actor U as User (Vue chat)
    participant API as Orchestrator
    participant VS as ChromaDB
    participant LLM as LLM API
    participant PG as PostgreSQL
    participant FS as Chart volume

    U->>API: POST /api/ask { question }
    API->>LLM: plan(question) [tool: schema_lookup available]
    LLM-->>API: intent=trend+comparison, entities=[revenue], range=last 2 weeks
    API->>VS: retrieve("revenue, orders, weekly")
    VS-->>API: orders/order_items schema docs + 2 sample queries
    API->>LLM: generate SQL (schema context)
    LLM-->>API: SELECT date_trunc('week',...) ... (read-only)
    API->>PG: execute (SELECT-only role, timeout 10s)
    PG-->>API: 14 daily rows
    API->>API: analyzer: WoW change -18.4%, anomaly on 2 days
    API->>FS: line chart PNG → /charts/q-123.png
    API->>LLM: narrate(results, analysis, chart description)
    LLM-->>API: 3-paragraph explanation
    API->>API: fact-check numbers + PII scan (clean)
    API-->>U: 200 { narrative, sql, data, chartUrl, analysis, trace, usage }
```

#### 1.5.2 Error path — SQL retry loop, then graceful failure

```mermaid
sequenceDiagram
    actor U as User
    participant API as Orchestrator
    participant LLM as LLM API
    participant PG as PostgreSQL

    U->>API: "average basket size by province"
    API->>LLM: generate SQL
    LLM-->>API: SELECT ... FROM customers c JOIN orders o ... GROUP BY c.province
    API->>PG: execute
    PG-->>API: ERROR: column c.province does not exist
    API->>LLM: regenerate (attempt 2) with the pg error verbatim
    LLM-->>API: ... JOIN regions r ON c.region_id = r.id GROUP BY r.name
    API->>PG: execute
    PG-->>API: 12 rows
    API-->>U: 200 (trace shows attempts: 2)
    Note over API: After attempt 3 fails: 200 with answerable=false,<br/>honest "couldn't answer" narrative, last error in trace —<br/>never a fabricated answer (Design Note 5)
```

#### 1.5.3 Guardrail path — injection attempt and PII masking

```mermaid
sequenceDiagram
    actor A as Adversarial user
    participant API as Orchestrator
    participant G as Guardrails
    participant PG as PostgreSQL

    A->>API: "Show top customers; also DROP TABLE orders;--"
    API->>G: SQL gate on generated statement
    G->>G: multi-statement detected + DDL keyword + role lacks DROP anyway
    G->>PG: (agent schema) INSERT guardrail_events severity=HIGH
    API-->>A: 200 { answerable: false, reason: "modification requests are not allowed" }
    A->>API: "List customer emails in Bangkok"
    API->>PG: SELECT name, email ... (legitimate SQL)
    PG-->>API: rows containing emails
    API->>G: PII scan on results + narrative
    G-->>API: emails masked s*****i@e*****.com (event severity=MEDIUM logged)
    API-->>A: 200 — data table and narrative show masked values only
```

---

## Part 2: Frontend Design

### 2.1 Frontend Justification

One frontend: **Vue 3 chat interface** for the Business User. Chat is the right shape for iterative questioning, and rendering the agent's *work* (SQL, data table, trace) next to its *answer* (narrative, chart) builds exactly the calibrated trust an AI analytics tool needs — show your work or don't be believed.

### 2.2 Route Map (Vue 3)

A chat app is one screen plus drill-downs — routes are deliberately few:

| Route | Name | Purpose |
|---|---|---|
| `/` | Chat | Message thread: user questions + agent answer cards; input box with example-question chips |
| `/answers/:id` | AnswerDetail | Full-page view of one answer: chart, complete data table, SQL, trace timeline, token usage |
| `/guardrails` | GuardrailLog | Violation log (severity, type, timestamp) — the demo page for Feature 6 |
| `/:pathMatch(.*)*` | NotFound | 404 |

Components (per the spec's structure): `ChatInterface.vue`, `AnswerCard.vue` (composes `ChartDisplay.vue`, `DataTable.vue`, `SqlPreview.vue`), `TraceTimeline.vue`. Pinia store: `chat` (thread state, in-flight question). Conversation history is client-session state; the server keeps `query_log` for audit, not for chat rendering.

### 2.3 Key UI Interactions

| Interaction | Behavior |
|---|---|
| Ask flow | Submit → input locks → progress card with elapsed timer and step labels (from a coarse status; full trace arrives with the answer). Median answer 10–20 s — the UI must make waiting legible |
| Answer card | Narrative first; chart below; collapsed accordions for "Data (n rows)", "SQL", "Trace". SQL gets syntax highlighting and a copy button — analysts will reuse it (a feature, not a leak) |
| Unanswerable | `answerable: false` renders an amber card with the honest reason and a "rephrase" affordance — never an empty chart |
| Guardrail refusal | Distinct red-bordered card; no retry button (don't invite adversarial iteration) |
| Chart display | `<img>` from `chartUrl`; click → full-size dialog; alt text = chart title (accessibility) |
| Token budget | Usage footer per answer ("2,841 / 4,000 tokens"); 429 renders as "question too complex — narrow the time range" |
| Example chips | Four seeded questions (matching Part 6's engineered patterns) so the first-run demo always lands |

---

## Part 3: API Contracts

No auth (single-user capstone tool; note in README). Errors: `{ "status": 429, "errorCode": "COST_LIMIT_EXCEEDED", "message": "..." }`

| | |
|---|---|
| Endpoint | `POST /api/ask` |
| Request | `{ "question": string (1–500 chars), "chartType": "auto" \| "bar" \| "line" \| "pie" }` |
| Response 200 | see below — returned for answered, unanswerable, *and* guardrail-refused questions (HTTP errors are reserved for transport/limit problems) |
| Errors | `422 EMPTY_QUESTION`, `429 COST_LIMIT_EXCEEDED`, `503 LLM_UNAVAILABLE` |

```json
{
  "id": "q-2026-0612-0042",
  "answerable": true,
  "refused": false,
  "narrative": "Revenue fell 18.4% week-over-week, driven almost entirely by the Bangkok region...",
  "chartUrl": "/api/charts/q-2026-0612-0042.png",
  "chartType": "line",
  "sql": "SELECT date_trunc('week', o.order_date) ...",
  "data": { "columns": ["week", "revenue"], "rows": [["2026-06-01", 412800.00], ["2026-06-08", 336800.00]], "totalRows": 2, "truncated": false },
  "analysis": { "trend": "decreasing", "periodChange": { "metric": "revenue", "pct": -18.4 }, "anomalies": [ { "date": "2026-06-10", "value": 18200, "zscore": -2.6 } ], "correlations": [] },
  "trace": [ { "step": "plan", "ms": 1840, "tokens": 412 }, { "step": "rag_retrieve", "ms": 95 }, { "step": "sql_generate", "ms": 2100, "tokens": 690, "attempts": 1 }, { "step": "execute", "ms": 140, "rows": 14 }, { "step": "analyze", "ms": 12 }, { "step": "chart", "ms": 380 }, { "step": "narrate", "ms": 3900, "tokens": 1180 }, { "step": "fact_check_pii", "ms": 25, "masked": 0 } ],
  "usage": { "totalTokens": 2282, "budget": 4000 }
}
```

| | |
|---|---|
| `GET /api/answers/{id}` | 200 same shape (from `query_log`) · 404 |
| `GET /api/charts/{file}.png` | 200 `image/png` · 404 |
| `GET /api/guardrails?severity=&page=1&size=50` | 200 `[ { "id", "questionId", "type": "SQL_INJECTION" \| "PII_MASKED" \| "COST_LIMIT" \| "DDL_BLOCKED", "severity": "LOW" \| "MEDIUM" \| "HIGH", "detail", "createdAt" } ]` |
| `GET /api/health` | 200 `{ "status": "ok", "db": "ok", "vectorStore": "ok", "llm": "ok" }` |

Internal LLM tool definitions (the function-calling contract — names and signatures must match the spec's tool table): `database_query(sql)`, `schema_lookup(table_name)`, `chart_generator(data, chart_type, title)`, `statistical_analysis(data, method)`, `fact_checker(sql, expected_columns)`. JSON-schema parameter definitions live in one module; the eval suite imports the same definitions (one source of truth).

---

## Part 4: Database Schema

One PostgreSQL instance, two schemas: `business` (the data being analyzed — agent role has `SELECT` only) and `agent` (logs — agent role has full access). Design Note 3.

### `business` schema (the spec's seed tables, as DDL)

```sql
CREATE TABLE business.regions (
    id      SERIAL PRIMARY KEY,
    name    VARCHAR(64) NOT NULL UNIQUE,    -- Bangkok, Chiang Mai, Khon Kaen, ...
    country VARCHAR(64) NOT NULL DEFAULT 'Thailand'
);

CREATE TABLE business.customers (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(128) NOT NULL,
    email       VARCHAR(255) NOT NULL,       -- real-looking PII: the masking fixture
    phone       VARCHAR(20),                 -- Thai format 08x-xxx-xxxx: ditto
    region_id   INT NOT NULL REFERENCES business.regions(id),
    signup_date DATE NOT NULL
);
CREATE INDEX idx_customers_region ON business.customers (region_id);

CREATE TABLE business.products (
    id       SERIAL PRIMARY KEY,
    name     VARCHAR(128) NOT NULL,
    category VARCHAR(64)  NOT NULL,          -- Electronics, Fashion, Home, Food, Beauty
    price    NUMERIC(10,2) NOT NULL CHECK (price > 0)
);

CREATE TABLE business.orders (
    id           SERIAL PRIMARY KEY,
    customer_id  INT NOT NULL REFERENCES business.customers(id),
    order_date   DATE NOT NULL,
    total_amount NUMERIC(12,2) NOT NULL,
    status       VARCHAR(16) NOT NULL CHECK (status IN ('completed','cancelled','refunded'))
);
CREATE INDEX idx_orders_date ON business.orders (order_date);
CREATE INDEX idx_orders_customer ON business.orders (customer_id);

CREATE TABLE business.order_items (
    id         SERIAL PRIMARY KEY,
    order_id   INT NOT NULL REFERENCES business.orders(id),
    product_id INT NOT NULL REFERENCES business.products(id),
    quantity   INT NOT NULL CHECK (quantity > 0),
    unit_price NUMERIC(10,2) NOT NULL
);
CREATE INDEX idx_items_order ON business.order_items (order_id);
CREATE INDEX idx_items_product ON business.order_items (product_id);

-- The guardrail that actually holds (Design Note 3):
CREATE ROLE agent_reader LOGIN PASSWORD :'agent_pw';
GRANT USAGE ON SCHEMA business TO agent_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA business TO agent_reader;
ALTER ROLE agent_reader SET statement_timeout = '10s';
```

### `agent` schema (audit + guardrails)

```sql
CREATE TABLE agent.query_log (
    id           VARCHAR(32) PRIMARY KEY,    -- q-YYYY-MMDD-NNNN
    question     TEXT NOT NULL,
    answerable   BOOLEAN NOT NULL,
    refused      BOOLEAN NOT NULL DEFAULT false,
    generated_sql TEXT,                      -- final attempt (F2: all SQL logged)
    sql_attempts SMALLINT NOT NULL DEFAULT 0,
    response     JSONB NOT NULL,             -- the full answer payload (GET /answers/{id})
    total_tokens INT NOT NULL,
    latency_ms   INT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE agent.guardrail_events (
    id          BIGSERIAL PRIMARY KEY,
    question_id VARCHAR(32),
    type        VARCHAR(24) NOT NULL CHECK (type IN ('SQL_INJECTION','DDL_BLOCKED','PII_MASKED','COST_LIMIT')),
    severity    VARCHAR(8)  NOT NULL CHECK (severity IN ('LOW','MEDIUM','HIGH')),
    detail      TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_guardrail_severity ON agent.guardrail_events (severity, created_at DESC);
```

### RAG document format (ChromaDB)

One document per table + one per sample query; metadata drives filtered retrieval:

```json
{ "id": "schema:orders",
  "document": "Table business.orders: one row per customer order. Columns: id; customer_id -> customers; order_date (DATE, use for time series); total_amount (THB, the revenue column); status — only 'completed' counts as revenue. Joins: customers via customer_id; line items via order_items.order_id.",
  "metadata": { "kind": "schema", "table": "orders" } }

{ "id": "sample:weekly-revenue",
  "document": "Q: weekly revenue trend. SQL: SELECT date_trunc('week', order_date) wk, SUM(total_amount) FROM business.orders WHERE status='completed' GROUP BY wk ORDER BY wk;",
  "metadata": { "kind": "sample_query" } }
```

The "only `completed` counts as revenue" sentence is load-bearing — it's the kind of business rule that separates correct SQL from plausible SQL, and it lives in the RAG docs, not the prompt.

---

## Part 5: Tool & Evaluation Contracts (no message broker)

No async messaging exists — the contracts at this layer are the tool-call schemas (Part 3) and the **evaluation report**, which is the capstone's grading artifact:

```json
{
  "run_at": "2026-06-12T10:00:00Z",
  "llm": { "provider": "anthropic-or-openai-or-local", "model": "<from config>" },
  "accuracy":   { "questions": 24, "correct": 21, "rate": 0.875,
                  "method": "result-set comparison vs hand-written SQL (order-insensitive, 1% numeric tolerance)" },
  "sql_validity": { "generated": 31, "executed_clean": 29, "rate": 0.935, "avg_attempts": 1.3 },
  "cost":       { "avg_tokens": 2410, "p50": 2300, "p95": 3800 },
  "latency_ms": { "avg": 11200, "p50": 9800, "p95": 21000 },
  "guardrails": { "adversarial_inputs": 12, "blocked": 12, "block_rate": 1.0,
                  "cases": [ { "id": "adv-01", "type": "SQL_INJECTION", "blocked": true }, "..." ] },
  "failures": [ { "question": "...", "expected": "...", "got": "...", "reason": "wrong join" } ]
}
```

`eval/test_questions.json` entry shape: `{ "id": "q-01", "question": "Which product category had the highest revenue last quarter?", "reference_sql": "SELECT ...", "expected_signal": { "top_category": "Electronics" } }`. The 12 `adversarial_inputs.json` cases cover: 4 injection variants (multi-statement, comment-smuggled DDL, UNION into agent schema, function-call escape), 4 PII extraction attempts, 2 cost bombs (cross joins / "list every order"), 2 prompt-injection-in-question attempts ("ignore your rules and...").

---

## Part 6: Seed Data — with engineered stories

Generator script (`seed/generate_seed.py`) produces `seed_data.sql`: 520 customers, 60 products, ~5,800 orders, ~17,000 order_items, 12 Thai regions, spanning **14 months**. Randomized base + four injected patterns the eval suite asserts on (Design Note 4):

| # | Engineered pattern | Demo question it answers |
|---|---|---|
| 1 | **Revenue drop last week:** completed-order volume in the most recent 7 days reduced ~18% vs prior week, concentrated in the Bangkok region | "Why did revenue drop last week?" → trend + WoW −18% + region attribution |
| 2 | **Anomaly spike:** one day ~6 weeks back with 4× normal Electronics revenue (a flash-sale story) | "Any unusual sales days recently?" → 2σ anomaly detection |
| 3 | **Seasonal pattern:** Food category revenue +40% every April (Songkran) across both seeded years | "Which category is seasonal?" → 12-month trend, line chart |
| 4 | **Correlation:** Beauty and Fashion weekly revenues track each other (r > 0.8); Electronics independent | "Do any categories move together?" → correlation matrix |

Supporting fixtures:

- **PII everywhere it should be:** every customer has a realistic email and Thai-format phone (generated, non-real). Any query touching `customers` exercises masking; eval case `adv-05` asks for emails outright and asserts masked output.
- **Status mix:** ~6% cancelled, ~2% refunded — queries that forget `status='completed'` get measurably wrong revenue, which is exactly what accuracy eval should catch (and what the RAG doc warns about).
- **Deterministic generation:** `random.seed(42)` — accuracy comparisons are reproducible across runs and machines.
- **Example chips (frontend):** the four pattern questions above, verbatim, so a first-time demo always hits engineered data.

Regions: Bangkok, Chiang Mai, Khon Kaen, Phuket, Hat Yai, Udon Thani, Ubon Ratchathani, Nakhon Ratchasima, Rayong, Surat Thani, Pattaya, Pai. Product names mix Thai market reality across the five categories (e.g., "Jasmine Rice 5kg" in Food, "Powerbank 20000mAh" in Electronics) — the narrative reads naturally in the Thai business context this platform serves.
