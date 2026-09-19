# RAG Documentation Assistant

A production-quality Retrieval-Augmented Generation pipeline built with Python, LangGraph, and ChromaDB. Every concept is modularised into its own file so you can read and understand each piece in isolation.

---

## What This Project Teaches

1. **Character vs Semantic Chunking** — Why splitting at fixed byte counts destroys code blocks and mixes topics, and how splitting at Markdown `##` headers preserves meaning boundaries.
2. **Hybrid Chunking** — How a word-count gate triggers `###` sub-splitting for oversized sections, combining the best of structural and size-aware chunking.
3. **Embedding & Vector Storage** — How `sentence-transformers` converts text into fixed-size vectors, and how ChromaDB's HNSW index stores and retrieves them from disk.
4. **Hybrid Search (BM25 + Vector + RRF)** — Why keyword search (BM25) finds exact function names that semantic search misses, and how Reciprocal Rank Fusion merges both ranked lists.
5. **LangGraph RAG Pipeline** — How a `StateGraph` wires three pure-function nodes (retrieve → build_prompt → generate) into a traceable, error-handled pipeline.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy the example env file and add your key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

---

## Add Your Documents

Place `.md` files in `docs/`. Two documents are already included:

| File | Content |
|------|---------|
| `docs/MCP_Documentation.md` | Healthcare Patient Records MCP Server guide |
| `docs/DOCUMENTATION.md` | AI Agents — Traditional Function Tools vs MCP |

---

## Demo — Character vs Semantic Chunking

This is the core demo. Run the steps below in order to see exactly how chunking strategy affects RAG answer quality.

---

### Step 1 — Ingest with Character Splitting

Character splitting cuts at fixed 500-character intervals, ignoring section boundaries, code blocks, and tables.

```bash
python demo/02_ingest_char.py
```

Expected output: ~60–70 chunks created from the two documents.

---

### Step 2 — Query with Character-Split Chunks

```bash
python demo/03_query.py "Does this project use a remote Google Maps MCP server?"
```

**What to observe:**
- The BM25 and vector results printed before the answer will show fragmented chunks — content mid-sentence, broken code blocks, sections mixed together.
- The answer may be vague, miss key details, or reference the wrong context because the retrieved chunks don't align with meaningful document sections.

---

### Step 3 — Delete the Vector Store

```bash
rm -rf chroma_data/*
```

This clears all stored chunks and embeddings. The folder itself is preserved (it's in `.gitignore`).

---

### Step 4 — Ingest with Semantic (Structure-Based) Chunking

Semantic splitting cuts at `##` Markdown headers. Sections over 400 words are further split at `###` headers. Every chunk is a complete, self-contained section.

```bash
python demo/02_ingest.py
```

Expected output: ~18–22 chunks — far fewer but each chunk is a whole section.

---

### Step 5 — Query with Semantic Chunks

```bash
python demo/03_query.py "What Python code do I need to connect an agent to the Maps MCP server?"
```

**What to observe:**
- The retrieved chunks will be complete sections — the MCPToolset setup code, the StreamableHTTPConnectionParams block — intact and in context.
- The answer will include the actual Python code snippet and a precise citation like `[Source: DOCUMENTATION.md, Section: Modern Approach: MCP_Maps_Agent]`.

---

### Optional — Chunking Stats Comparison

To see a side-by-side statistical comparison of both strategies (broken code blocks, broken tables, average words per chunk) without touching the database:

```bash
python demo/01_chunking_comparison.py
```

---

### Non-Hallucination Test

Ask a question that is not in any document. The system should refuse rather than invent an answer:

```bash
python demo/03_query.py "What is the rate limit on the Stripe API?"
```

Expected: `I could not find an answer to this question in the provided documentation.`

---

## Project Structure

```
RAG_Production/
├── docs/
│   ├── MCP_Documentation.md     # Source document 1
│   └── DOCUMENTATION.md         # Source document 2
├── config/
│   └── settings.py              # All constants — chunk size, model name, paths
├── chunking/
│   ├── character_splitter.py    # Fixed-interval splitting (the "bad" baseline)
│   ├── semantic_splitter.py     # Header-boundary splitting with hybrid H3 gate
│   └── comparison.py            # Runs both strategies and computes stats
├── vectorstore/
│   ├── embedder.py              # Loads SentenceTransformer, encodes chunks
│   └── store.py                 # ChromaDB PersistentClient — upsert and load
├── retrieval/
│   ├── semantic_search.py       # Pure vector search via ChromaDB
│   └── hybrid_search.py         # BM25 + vector + Reciprocal Rank Fusion
├── generation/
│   ├── prompt_builder.py        # Assembles system prompt + grounded user message
│   └── answer_generator.py      # Calls OpenAI API, extracts citations
├── pipeline/
│   ├── state.py                 # RAGState TypedDict shared across all nodes
│   ├── nodes.py                 # Three pure-function LangGraph nodes
│   └── graph.py                 # StateGraph wiring + run_query() entry point
├── demo/
│   ├── 01_chunking_comparison.py  # stats comparison, no DB required
│   ├── 02_ingest_char.py          # ingest with character splitting (step 1 of demo)
│   ├── 02_ingest.py               # ingest with semantic splitting  (step 4 of demo)
│   └── 03_query.py                # query the pipeline with any question
├── chroma_data/                 # Auto-created by ChromaDB — in .gitignore
├── main.py                      # Runs all three demos in sequence
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Upgrading to Production

ChromaDB's HNSW index parameters (`m`, `ef_construct`) are chosen automatically here — good for getting started, but not tunable. To configure them explicitly, swap `vectorstore/store.py` for a Qdrant client:

```python
# vectorstore/store.py (Qdrant version)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
)
```

No other file needs to change — `nodes.py`, `graph.py`, and the demo scripts are all unaffected.
