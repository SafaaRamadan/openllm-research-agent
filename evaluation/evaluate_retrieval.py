
import json
from pathlib import Path

from src.research_agent.retriever import ResearchRetriever


def main():
    questions_path = Path("evaluation/questions.json")

    with questions_path.open(encoding="utf-8") as file:
        questions = json.load(file)

    if not questions:
        raise ValueError("No evaluation questions found.")

    retriever = ResearchRetriever()
    hits_at_1 = 0
    hits_at_3 = 0


    for item in questions:
        results = retriever.search(
            item["question"],
            top_k=3,
        )

        expected_source = item["expected_source"]
        retrieved_sources = [
            result["source"] for result in results
        ]

        hit_at_1 = (
            bool(retrieved_sources)
            and retrieved_sources[0] == expected_source
        )
        hit_at_3 = expected_source in retrieved_sources

        hits_at_1 += int(hit_at_1)
        hits_at_3 += int(hit_at_3)

        print(f"\nQuestion: {item['question']}")
        print(f"Expected source: {expected_source}")

        for rank, result in enumerate(results, start=1):
            print(
                f"Rank {rank}: {result['source']} | "
                f"similarity={result['similarity']:.3f}"
            )

        print(f"Top 3 sources: {retrieved_sources}")
        print(f"Hit@1: {hit_at_1}")
        print(f"Hit@3: {hit_at_3}")


   


    total = len(questions)

    print("\n--- Retrieval Evaluation ---")
    print(f"Questions: {total}")
    print(f"Hit@1: {hits_at_1 / total:.2%}")
    print(f"Recall@3: {hits_at_3 / total:.2%}")


if __name__ == "__main__":
    main()
