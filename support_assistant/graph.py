"""LangGraph flow: classify_intent -> (retrieve_and_answer | direct_answer).

MOCK_LLM unset or "1" (default, graded): no LLM call anywhere, deterministic output.
MOCK_LLM="0" (optional): the generation step in each node calls Groq's free-tier API.
Routing and retrieval behave identically in both modes.
"""

import os
from typing import List, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from ingest import retrieve
from prompts import CLASSIFY_PROMPT, CORRECTIVE_INSTRUCTION, GENERAL_PROMPT, RAG_PROMPT, format_context

POLICY_KEYWORDS = ["delivery", "return", "refund", "membership", "tracking",
                   "cancel", "gift card", "support hours"]
GENERAL_REPLY = "I can only answer questions about Zepto policies right now."
SNIPPET_CHARS = 200
MAX_RETRIES = 2  # extra attempts after the first, real-LLM path only


def mock_mode():
    return os.getenv("MOCK_LLM", "1") != "0"


# ------------------------------------------------------------------ schemas
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["What is the delivery fee for small orders?"])


class AskResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)


class GraphState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved: List[dict]
    response: dict


# --------------------------------------------------------------- real LLM
def call_llm(prompt, history=None):  # pragma: no cover - optional, needs GROQ_API_KEY
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    messages = (history or []) + [{"role": "user", "content": prompt}]
    out = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), messages=messages, temperature=0)
    return out.choices[0].message.content


def generate_structured(prompt, allowed_sources: Optional[List[str]] = None):
    """Real-LLM path: validate the reply against AskResponse; on failure, retry up to
    MAX_RETRIES more times with a corrective instruction, then return a marked error."""
    history, error = [], None
    for attempt in range(1 + MAX_RETRIES):
        message = prompt if attempt == 0 else CORRECTIVE_INSTRUCTION.format(error=error)
        try:
            raw = call_llm(message, history)
        except Exception as exc:  # network / auth / quota problems are not retryable here
            return AskResponse(answer=f"[ERROR] LLM call failed: {type(exc).__name__}",
                               sources=[], confidence=0.0).model_dump()
        history += [{"role": "user", "content": message}, {"role": "assistant", "content": raw}]
        try:
            text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = AskResponse.model_validate_json(text)
            if allowed_sources is not None and not set(parsed.sources) <= set(allowed_sources):
                raise ValueError(f"sources must be a subset of {allowed_sources}")
            return parsed.model_dump()
        except (ValidationError, ValueError) as exc:
            error = " ".join(str(exc).split())[:300]   # full message, flattened, for the corrective prompt
    return AskResponse(answer=f"[ERROR] Model output failed schema validation after "
                              f"{1 + MAX_RETRIES} attempts: {error}",
                       sources=[], confidence=0.0).model_dump()


# ------------------------------------------------------------------- nodes
def keyword_intent(query):
    q = query.lower()
    return "policy_question" if any(k in q for k in POLICY_KEYWORDS) else "general_question"


def classify_intent(state: GraphState) -> GraphState:
    if mock_mode():
        return {"intent": keyword_intent(state["query"])}
    label = call_llm(CLASSIFY_PROMPT.format(question=state["query"])).strip().lower()
    if label not in ("policy_question", "general_question"):
        label = keyword_intent(state["query"])      # unparseable label -> heuristic fallback
    return {"intent": label}


def snippet(text, limit=SNIPPET_CHARS):
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def retrieve_and_answer(state: GraphState) -> GraphState:
    chunks = retrieve(state["query"], k=3)          # real embedding + ChromaDB search, both modes
    ids = [c["id"] for c in chunks]
    if mock_mode():
        response = AskResponse(answer=f"Based on the retrieved context: {snippet(chunks[0]['text'])}",
                               sources=ids, confidence=1.0).model_dump()
    else:
        prompt = RAG_PROMPT.format(context=format_context(chunks), question=state["query"])
        response = generate_structured(prompt, allowed_sources=ids)
    return {"retrieved": chunks, "response": response}


def direct_answer(state: GraphState) -> GraphState:
    if mock_mode():
        response = AskResponse(answer=GENERAL_REPLY, sources=[], confidence=1.0).model_dump()
    else:
        response = generate_structured(GENERAL_PROMPT.format(question=state["query"]), allowed_sources=[])
    return {"retrieved": [], "response": response}


def route(state: GraphState) -> str:
    return state["intent"]


# ------------------------------------------------------------------- graph
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("classify_intent", classify_intent)
    g.add_node("retrieve_and_answer", retrieve_and_answer)
    g.add_node("direct_answer", direct_answer)
    g.add_edge(START, "classify_intent")
    g.add_conditional_edges("classify_intent", route, {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer",
    })
    g.add_edge("retrieve_and_answer", END)
    g.add_edge("direct_answer", END)
    return g.compile()


graph = build_graph()


def ask(query: str) -> AskResponse:
    final = graph.invoke({"query": query})
    return AskResponse.model_validate(final["response"])


if __name__ == "__main__":
    from ingest import build_index
    build_index()
    print(f"MOCK_LLM mode: {mock_mode()}")
    for q in ["What is the delivery fee on a small order?",
              "How do I cancel my order?",
              "Can I use two gift cards together?",
              "What's the weather in Mumbai today?"]:
        state = graph.invoke({"query": q})
        print(f"\nQ: {q}\n  intent={state['intent']}  "
              f"retrieved={[(c['id'], c['similarity']) for c in state['retrieved']]}\n"
              f"  {state['response']}")
