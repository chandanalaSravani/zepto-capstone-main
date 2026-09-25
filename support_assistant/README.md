# Module 3 - Zepto Support Assistant (`/support_assistant`)

A small RAG service that answers questions about Zepto's policies, grounded in the 8 policy
documents in [`docs/`](docs/). It uses local embeddings and ChromaDB, a LangGraph intent
router, a Pydantic-validated JSON response, and a FastAPI endpoint packaged in Docker.

**By default the service makes no LLM call at all.** When `MOCK_LLM` is unset or set to `1`,
every generation step runs deterministic, rule-based mock logic. There's no API key, no
signup, and no network call to any LLM provider. That default is what's documented and
recorded below.

## Files

| File | Role |
|---|---|
| `docs/doc_01.txt` ... `doc_08.txt` | the corpus, copied verbatim from the brief |
| `ingest.py` | ingestion, chunking, embedding (`all-MiniLM-L6-v2`), ChromaDB storage, and the `retrieve()` query function |
| `prompts.py` | structured prompt templates (role / context / task / format / length) for the optional real-LLM path |
| `graph.py` | Pydantic schemas, the LangGraph `StateGraph` (3 nodes and a conditional edge), `MOCK_LLM` branching, retry-on-invalid-JSON logic |
| `main.py` | FastAPI app: `POST /ask`, `GET /health` |
| `Dockerfile`, `requirements-docker.txt` | container image serving `/ask` on port 7860 |

## Run

```bash
cd support_assistant
python ingest.py                 # builds ./chroma_db and prints sample retrievals
python graph.py                  # runs 4 example queries through the graph (no server)
uvicorn main:app --port 7860     # API at http://localhost:7860  (interactive docs at /docs)
```

### Docker (locally buildable and runnable)

```bash
cd support_assistant
docker build -t zepto-support .
docker run --rm -p 7860:7860 zepto-support
# then, from another terminal:
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "What is the delivery fee on a small order?"}'
```

The image is `python:3.11-slim` with CPU-only PyTorch. At **build** time, `RUN python ingest.py`
downloads the embedding model and builds the ChromaDB index, so the running container needs
no network access in mock mode. `CMD` runs
`uvicorn main:app --host 0.0.0.0 --port 7860`. `MOCK_LLM=1` is baked in as the default. To try
the optional real LLM, use `docker run -e MOCK_LLM=0 -e GROQ_API_KEY=... -p 7860:7860 zepto-support`.
Never commit the key to the repo.

## Example calls (recorded with `MOCK_LLM` at its default)

The server was started with `uvicorn main:app --port 7860` and no `MOCK_LLM` set
(`GET /health` returned `{"status":"ok","mock_llm":true}`). These are the raw responses.

**1. A policy question, which triggers retrieval** (keyword `delivery` -> `policy_question` -> `retrieve_and_answer`)

```
$ curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"query": "What is the delivery fee on a small order?"}'
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard...","sources":["doc_01_chunk_0","doc_05_chunk_0","doc_02_chunk_0"],"confidence":1.0}
```

The top chunk is `doc_01` (Delivery Policy), which contains the INR 25 fee for orders under INR
149, with cosine similarity 0.556.

**2. Another retrieval example** (keyword `refund`)

```
$ curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"query": "How long do refunds take?"}'
{"answer":"Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in...","sources":["doc_02_chunk_0","doc_06_chunk_0","doc_05_chunk_0"],"confidence":1.0}
```

The top chunk is `doc_02` (Returns & Refunds), which holds the "3-5 business days" refund
rule.

**3. A general question, with no retrieval** (no keyword -> `general_question` -> `direct_answer`)

```
$ curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"query": "Can you recommend a good movie?"}'
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

The request model rejects an empty query with a 422:
`{"detail":[{"type":"string_too_short","loc":["body","query"],"msg":"String should have at least 1 character",...}]}`

`python graph.py` shows the routing and the top-3 retrieval scores directly. Each policy query's
top hit is the matching document:

| Query | Intent | Top 3 chunks (cosine similarity) |
|---|---|---|
| What is the delivery fee on a small order? | policy_question | doc_01 (0.556), doc_05 (0.368), doc_02 (0.359) |
| How do I cancel my order? | policy_question | doc_05 (0.538), doc_06 (0.327), doc_02 (0.287) |
| Can I use two gift cards together? | policy_question | doc_07 (0.524), doc_01 (0.105), doc_02 (0.102) |
| What's the weather in Mumbai today? | general_question | none (no retrieval) |

**Offline check of the retry logic.** `call_llm` was replaced with a fake that returns plain
text, then JSON with `confidence: 7`, then valid JSON. `generate_structured` accepted the third
attempt, after exactly 3 calls. A fake that always returns invalid output produced
`{"answer": "[ERROR] Model output failed schema validation after 3 attempts: ...", "sources": [], "confidence": 0.0}`.

## Architecture: ingestion -> embedding -> retrieval -> generation

```
 docs/doc_01..08.txt
        |  ingest.py: load_chunks()   (1 chunk per document, id "doc_NN_chunk_0")
        v
 all-MiniLM-L6-v2 (sentence-transformers, local)   ingest.py: embed()  -> 384-d normalized vectors
        v
 ChromaDB PersistentClient ./chroma_db, collection "zepto_policies" (hnsw:space = cosine)
        ^
        |  ingest.py: retrieve(query, k=3)
        |
 POST /ask {"query"} --> main.py --> graph.ask() --> LangGraph StateGraph (graph.py)
                                                   |
                                          [classify_intent]
                                  policy_question /       \ general_question
                             [retrieve_and_answer]       [direct_answer]
                              top-3 chunks + answer       fixed / LLM reply
                                                   \       /
                                        AskResponse {answer, sources, confidence}
```

1. **Ingestion** (`ingest.py`: `load_chunks`). Reads the 8 `.txt` files. Each document becomes
   **one chunk**. The documents are only 55-90 words, well under MiniLM's 256-token input
   limit, and each covers exactly one policy topic. Splitting further would separate facts
   that belong together (for example, a time limit from the item type it applies to). Each
   chunk has the id `doc_NN_chunk_0`, plus metadata (`doc_id`, `title`, `source` filename).
2. **Embedding** (`ingest.py`: `embed`, `build_index`). `sentence-transformers` runs
   `all-MiniLM-L6-v2` locally and produces L2-normalized 384-dimension vectors. They are
   upserted with the text and metadata into the ChromaDB **`zepto_policies`** collection,
   persisted in `./chroma_db`, using cosine distance. `main.py` calls `build_index()` on
   startup, which only does work if the collection is empty. The Docker build runs it ahead
   of time.
3. **Retrieval** (LangGraph node **`retrieve_and_answer`** -> `ingest.retrieve`). The query is
   embedded with the same model, and ChromaDB returns the **top 3** chunks by cosine
   similarity (reported as `1 - distance`). This step always runs for real, in both modes.
   Before retrieval, **`classify_intent`** decides whether retrieval is needed at all. A
   conditional edge (`route()`) sends `policy_question` to `retrieve_and_answer` and
   `general_question` to **`direct_answer`**, which does no retrieval. The routing logic
   doesn't depend on `MOCK_LLM`.
4. **Generation** (inside `retrieve_and_answer` / `direct_answer`). The node builds an
   `AskResponse` Pydantic model with `answer: str`, `sources: list[str]` and
   `confidence: float` constrained to 0-1. FastAPI validates and serializes it as the
   endpoint's `response_model`.

### What `MOCK_LLM` changes

Only the **generation steps inside the three nodes** branch on `MOCK_LLM`. Ingestion,
embedding, retrieval and routing are identical in both modes.

| Node | Default: `MOCK_LLM` unset or `1` (no LLM call) | Optional: `MOCK_LLM=0` (Groq free tier) |
|---|---|---|
| `classify_intent` | keyword heuristic: `policy_question` if the lowercased query contains any of `delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`; otherwise `general_question` | `CLASSIFY_PROMPT` sent to the LLM; if the reply isn't a valid label, it falls back to the heuristic |
| `retrieve_and_answer` | `"Based on the retrieved context: " + <first ~200 chars of the top chunk, cut at a word boundary>`, `sources` = the 3 retrieved chunk ids, `confidence` = 1.0 | `RAG_PROMPT` filled with the 3 chunks and the question; the model must answer only from that context |
| `direct_answer` | fixed string `"I can only answer questions about Zepto policies right now."`, `sources` = `[]`, `confidence` = 1.0 | `GENERAL_PROMPT`, no retrieval |

**Structured-output guarantee.** In mock mode the response is built directly from code, so it
always validates. On the real-LLM path, `generate_structured()` parses the model's text with
`AskResponse.model_validate_json` and also checks that every cited source is one of the
retrieved ids. If validation fails, it re-prompts with `CORRECTIVE_INSTRUCTION`, which
includes the validation error, **up to 2 more times**. After that it returns a clearly marked
`"[ERROR] Model output failed schema validation ..."` response with `confidence` 0.0.

### Prompt template (from `prompts.py`, used by the optional real-LLM path)

`RAG_PROMPT` has the five required sections: `### ROLE` (Zepto support assistant),
`### CONTEXT` (the retrieved chunks, each labelled with its id), `### TASK` (answer only from
context), `### FORMAT` (a single JSON object `{answer, sources, confidence}`) and `### LENGTH`
(at most 3 sentences, under 70 words). It also includes:

- **Negative constraints:** "Do NOT answer using information that is not present in the
  provided context, and do NOT guess, invent numbers, or rely on general knowledge about
  other companies."
- **A few-shot example:** a gift-card context excerpt, the question "How long is a Zepto
  gift card valid?", and the expected JSON answer citing `doc_07_chunk_0`.

See [`prompts.py`](prompts.py) for the full text of all three templates.

### Optional extensions

Neither optional extension (a real LLM via `MOCK_LLM=0`, or deployment to Hugging Face
Spaces) was used to produce any of the results above. The code for the real-LLM path is
present, but it was not required and it isn't what's graded.
