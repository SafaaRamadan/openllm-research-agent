
import chromadb
from sentence_transformers import SentenceTransformer

from src.research_agent.index_store import (
    DB_PATH,
    COLLECTION_NAME,
    MODEL_NAME,
)


class ResearchRetriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=DB_PATH)

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        if self.collection.count() == 0:
            raise ValueError("The vector collection is empty.")

        self.model = SentenceTransformer(MODEL_NAME)

    def search(
        self,
        question: str,
        top_k: int = 3,
    ) -> list[dict]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding = self.model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        matches = []

        for i, document in enumerate(result["documents"][0]):
            metadata = result["metadatas"][0][i]
            distance = result["distances"][0][i]

            matches.append({
                "source": metadata["source"],
                "chunk_id": metadata["chunk_id"],
                "text": document,
                "distance": float(distance),
                "similarity": 1.0 - float(distance),
            })

        return matches


if __name__ == "__main__":
    retriever = ResearchRetriever()

    while True:
        question = input("\nAsk a question (or type 'exit'): ").strip()

        if question.lower() == "exit":
            break

        try:
            for result in retriever.search(question):
                print(f"\nSource: {result['source']}")
                print(f"Similarity: {result['similarity']:.3f}")
                print(result["text"])

        except ValueError as error:
            print(f"Error: {error}")
