
import json
from pathlib import Path

from src.research_agent.retriever import ResearchRetriever


def main():
    questions_path = Path("evaluation/questions.json")

    with questions_path.open(encoding="utf-8") as file:
        questions = json.load(file)

    retriever = ResearchRetriever()
    hits = 0

    for item in questions:
        results = retriever.search(
            item["question"],
            top_k=1,
        )

        predicted_source = (
            results[0]["source"] if results else None
        )

        correct = predicted_source == item["expected_source"]
        hits += int(correct)

        print(f"\nQuestion: {item['question']}")
        print(f"Expected: {item['expected_source']}")
        print(f"Retrieved: {predicted_source}")
        print(f"Correct: {correct}")

    hit_at_1 = hits / len(questions) if questions else 0.0

    print("\n--- Evaluation Summary ---")
    print(f"Total questions: {len(questions)}")
    print(f"Correct top-1 results: {hits}")
    print(f"Hit@1: {hit_at_1:.2%}")


if __name__ == "__main__":
    main()
