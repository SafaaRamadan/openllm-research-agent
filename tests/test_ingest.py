
from src.research_agent.ingest import (
    load_documents,
    chunk_text,
    prepare_chunks,
)


def test_load_documents():
    docs = load_documents("data/raw")

    assert len(docs) >= 3
    assert all(doc["text"] for doc in docs)


def test_chunk_overlap():
    text = "A" * 600
    chunks = chunk_text(text, chunk_size=500, overlap=100)

    assert len(chunks) == 2
    assert len(chunks[0]) == 500
    assert len(chunks[1]) == 200
    assert chunks[0][-100:] == chunks[1][:100]


def test_prepare_chunks():
    chunks = prepare_chunks("data/raw")

    assert len(chunks) > 0
    assert all("source" in chunk for chunk in chunks)
    assert all("text" in chunk for chunk in chunks)