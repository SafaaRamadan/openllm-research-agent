#Data ingestion pipeline for research agent
#chunking of documents 

from pathlib import Path


def load_documents(data_dir: str = "data/raw") -> list[dict]:
    """Load Markdown and text files from a directory."""
    folder = Path(data_dir)
    documents = []

    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue

        text = path.read_text(encoding="utf-8").strip()

        if text:
            documents.append({
                "source": path.name,
                "text": text,
            })

    return documents


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    """Split text into overlapping character-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = text.strip()
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def prepare_chunks(
    data_dir: str = "data/raw",
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict]:
    """Load documents and split them into chunks."""
    documents = load_documents(data_dir)
    all_chunks = []

    for document in documents:
        chunks = chunk_text(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{document['source']}:{index}",
                "source": document["source"],
                "text": chunk,
            })

    return all_chunks