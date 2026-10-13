
import numpy as np
from sentence_transformers import SentenceTransformer

from src.research_agent.ingest import prepare_chunks


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class ResearchRetriever:
    def __init__(self, data_dir: str = "data/raw"):
        self.chunks = prepare_chunks(data_dir)

        if not self.chunks:
            raise ValueError("No document chunks found.")

        self.model = SentenceTransformer(MODEL_NAME)

        texts = [chunk["text"] for chunk in self.chunks]

        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    def search(self, question: str, top_k: int = 3) -> list[dict]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding = self.model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        # Normalized vectors: dot product gives cosine similarity.
        scores = self.embeddings @ query_embedding

        # Return the highest-scoring chunks first.
        indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in indices:
            chunk = self.chunks[int(index)]

            results.append({
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(scores[index]),
            })

        return results


if __name__ == "__main__":
    retriever = ResearchRetriever()

    while True:
        question = input("\nAsk a question (or type 'exit'): ").strip()

        if question.lower() == "exit":
            break

        try:
            results = retriever.search(question)

            for result in results:
                print(f"\nSource: {result['source']}")
                print(f"Similarity: {result['score']:.3f}")
                print(result["text"])

        except ValueError as error:
            print(f"Error: {error}")