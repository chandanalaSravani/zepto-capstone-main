"""FastAPI wrapper around the LangGraph support assistant.

Local:   cd support_assistant && uvicorn main:app --port 7860
Docker:  docker build -t zepto-support support_assistant && docker run -p 7860:7860 zepto-support
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from graph import AskRequest, AskResponse, ask, mock_mode
from ingest import build_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    build_index()          # no-op if the ChromaDB collection is already populated
    yield


app = FastAPI(title="Zepto Support Assistant", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "mock_llm": mock_mode()}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    return ask(req.query)
