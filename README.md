# AutoStream Social-to-Lead Agent

> **Assignment:** ServiceHive × Inflx — Social-to-Lead Agentic Workflow  
> **Company:** AutoStream — AI-powered video editing SaaS for content creators  
> **Stack:** Python 3.9+ · LangChain · LangGraph · FAISS · GPT-4o-mini

---

## What This Is

A **stateful, multi-turn conversational AI agent** that:

1. Detects user **intent** (greeting / product query / high intent) using an LLM classifier
2. Answers product questions using a **FAISS-backed RAG pipeline** grounded in a local knowledge base
3. Identifies **high-intent leads** and qualifies them over multiple turns (name → email → platform)
4. Executes a **gated mock CRM tool** only when all required fields are present
5. Maintains **full conversation state** across the entire session using LangGraph

---

## Project Structure

```
autostream-agent/
│
├── agent/
│   ├── __init__.py       # Package marker
│   ├── state.py          # AgentState TypedDict (LangGraph schema)
│   ├── intent.py         # LLM-based intent classifier
│   ├── rag.py            # FAISS RAG pipeline
│   ├── lead.py           # Lead qualification helpers + mock_lead_capture tool
│   └── graph.py          # LangGraph StateGraph (full workflow wiring)
│
├── data/
│   └── knowledge_base.md # AutoStream product docs (RAG source)
│
├── main.py               # CLI chat interface (entry point)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 1. How to Run Locally

### Prerequisites

- Python 3.9 or higher
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Step-by-Step Setup

```bash
# 1. Clone / enter the project directory
cd autostream-agent

# 2. Create and activate a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
#    Copy the template and fill in your real key:
copy .env.example .env        # Windows
# cp .env.example .env         # macOS/Linux
#
#    Then open .env and replace the placeholder:
#    OPENAI_API_KEY=sk-your-actual-key-here

# 5. Run the agent
python main.py
```

### Demo Script (Follow this for evaluation)

```
You: Hi
Agent: [Warm greeting, no RAG, no lead ask]

You: What are your pricing plans?
Agent: [RAG answer: Basic $29/mo, Pro $79/mo with details]

You: That sounds great, I want to try the Pro plan for my YouTube channel
Agent: [Detects high_intent, transitions to lead qualification]
       "Could you please share your name?"

You: My name is Alex Johnson
Agent: "Thanks! What's the best email address to reach you at?"

You: alex@gmail.com
Agent: "Great! Which platform(s) do you primarily create content on?"

You: YouTube
Agent: [TOOL fires] "🎉 You're all set, Alex Johnson! ..."

[State] lead_captured=True
Lead successfully captured.
```

---

## 2. Architecture Explanation

### Why LangGraph?

LangGraph was chosen over AutoGen because this workflow is **deterministic and sequential** — it follows a predictable state machine rather than a free-form multi-agent conversation. LangGraph's `StateGraph` provides first-class typed state (`AgentState`) that flows through every node, making state transitions explicit, auditable, and safe. Its conditional edge API lets us implement precise routing (e.g., "only call the CRM tool when all fields are populated") without hiding control flow inside the LLM itself.

### State Management

`AgentState` is a `TypedDict` with six fields: `messages`, `intent`, `name`, `email`, `platform`, and `lead_captured`. The `messages` field uses LangGraph's `add_messages` reducer, which **appends** new messages instead of overwriting — preserving the full conversation history across all turns. Every node receives the full state and returns only the fields it modifies; LangGraph merges these partial updates automatically.

### RAG vs. Intent Detection — Strict Separation

Intent classification (`agent/intent.py`) and RAG retrieval (`agent/rag.py`) are **completely separate modules** that never call each other. The graph classifier runs first on every turn and stores the intent label in state. The RAG node is only wired to handle the `product_query` route — it is structurally impossible for RAG to trigger on a greeting or high-intent message.

### Gated Tool Execution

`mock_lead_capture()` is **never called by the LLM directly**. Instead, it is called by the `high_intent_node` in `graph.py` only after an explicit Python-level check: `get_next_missing_field(merged_state) is None and not lead_captured`. This guard runs every turn and prevents premature execution regardless of what the LLM says.

---

## 3. WhatsApp Integration (Conceptual)

### Overview

This agent can be deployed to WhatsApp Business by adding a thin webhook layer that translates WhatsApp events into agent calls.

### Components

```
[WhatsApp User]
      │
      │  (sends message)
      ▼
[WhatsApp Business API]
      │
      │  POST /webhook (JSON payload)
      ▼
[FastAPI Webhook Server]
      │
      │  1. Validate signature (HMAC-SHA256)
      │  2. Extract user_id + message text
      │  3. Load AgentState from Redis using user_id as key
      │  4. Append HumanMessage to state["messages"]
      │  5. Call graph.invoke(state)
      │  6. Save updated state back to Redis (TTL: 24h)
      │  7. Send response via WhatsApp API
      ▼
[WhatsApp User receives response]
```

### State Persistence

In production, `AgentState` is serialized as JSON and stored in **Redis** with the WhatsApp `user_id` (or thread ID) as the key:

```
redis.set(f"agent:state:{user_id}", json.dumps(state), ex=86400)
```

This enables stateful multi-turn conversations that survive across server restarts, multiple webhook calls, and user inactivity gaps.

### Message Flow

1. WhatsApp sends a `POST` to your webhook endpoint with a message event.
2. The webhook server extracts the sender's ID and message text.
3. It loads the existing `AgentState` from Redis (or initialises a fresh one for new users).
4. The message is appended and `graph.invoke()` is called — identical to the CLI flow.
5. The AI response is extracted from state and sent back via the WhatsApp Send Message API.
6. The updated state is persisted back to Redis.

### Scaling Considerations

- Use **Celery + Redis** for async webhook handling (prevents timeout if LLM is slow)
- Add **message deduplication** to handle WhatsApp's at-least-once delivery guarantee
- Implement **session expiry** to reset state after 24h of inactivity
- Use **structured logging** (JSON) for observability of per-user agent state

---

## Technical Notes

| Component | Detail |
|---|---|
| LLM | `gpt-4o-mini` — cost-efficient, strong instruction following |
| Embeddings | `text-embedding-3-small` — fast, accurate, cheap |
| Vector Store | FAISS (local, in-memory) — zero infra overhead for demo |
| Chunking | `RecursiveCharacterTextSplitter` — 400 chars / 60 overlap |
| State Persistence | In-memory dict (CLI demo) / Redis (production) |
| Tool Execution | Python gate in `graph.py` — never LLM-triggered blindly |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | Your OpenAI API key |

---

*Built by ServiceHive × Inflx — AutoStream Social-to-Lead Agentic Workflow*
