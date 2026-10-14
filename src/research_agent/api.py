
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import traceback
from src.research_agent.agent import build_agent


app = FastAPI(
    title="OpenLLM Research Agent",
    description=(
        "Ask questions about open-source LLMs "
        "using retrieval-augmented generation."
    ),
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="Question about open-source LLMs",
    )


class Source(BaseModel):
    source: str
    chunk_id: str
    similarity: float


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


@lru_cache(maxsize=1)
def get_agent():
    return build_agent()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        result = get_agent().invoke({
            "question": question,
            "context": "",
            "sources": [],
            "answer": "",
        })

        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
        }

    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Failed to process the question.",
        ) from error
