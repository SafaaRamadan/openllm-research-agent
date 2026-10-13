
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from src.research_agent.ingest import prepare_chunks


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DB_PATH = "chroma_db"
COLLECTION_NAME = "research_documents"


def build_index(data_dir: str = "data/raw") -> int:
    chunks = prepare_chunks(data_dir)

    if not chunks:
        raise ValueError("No document chunks found.")

    # Persistent database saved locally.
    client = chromadb.PersistentClient(path=DB_PATH)

    # Rebuild the collection to avoid keeping deleted or stale chunks.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception as error:
        # Ignore only the case where the collection does not exist.
        if "does not exist" not in str(error).lower():
            raise

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=8,
        show_progress_bar=True,
    )

    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
            }
            for chunk in chunks
        ],
    )

    print(f"Indexed {collection.count()} chunks.")
    print(f"Database location: {Path(DB_PATH).resolve()}")

    return collection.count()


if __name__ == "__main__":
    build_index()
