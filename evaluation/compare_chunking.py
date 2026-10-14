
import json
from pathlib import Path

from src.research_agent.index_store import build_index
from src.research_agent.retriever import ResearchRetriever


CONFIGS = [
    {"chunk_size": 250, "overlap": 50},
    {"chunk_size": 500, "overlap": 100},
    {"chunk_size": 800, "overlap": 100},
]


def load_questions():
    path = Path("evaluation/questions.json")

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def evaluate_current_index(questions):
    retriever = ResearchRetriever()
    hits_at_1 = 0
    hits_at_3 = 0

    for item in questions:
        results = retriever.search(
            item["question"],
            top_k=3,
        )

        sources = [result["source"] for result in results]
        expected = item["expected_source"]

        if sources and sources[0] == expected:
            hits_at_1 += 1

        if expected in sources:
            hits_at_3 += 1

    total = len(questions)

    return {
        "hit_at_1": hits_at_1 / total if total else 0.0,
        "recall_at_3": hits_at_3 / total if total else 0.0,
    }


def main():
    questions = load_questions()

    if not questions:
        raise ValueError("No evaluation questions found.")

    for config in CONFIGS:
        print("\n" + "=" * 45)
        print(
            f"Chunk size={config['chunk_size']}, "
            f"overlap={config['overlap']}"
        )

        chunk_count = build_index(
            data_dir="data/raw",
            chunk_size=config["chunk_size"],
            overlap=config["overlap"],
        )

        scores = evaluate_current_index(questions)

        print(f"Indexed chunks: {chunk_count}")
        print(f"Hit@1: {scores['hit_at_1']:.2%}")
        print(f"Recall@3: {scores['recall_at_3']:.2%}")


if __name__ == "__main__":
    main()
