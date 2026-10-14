
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from src.research_agent.retriever import ResearchRetriever
from src.research_agent.generator import generate_answer


class AgentState(TypedDict):
    question: str
    context: str
    sources: list[dict]
    answer: str


def retrieve_node(state: AgentState) -> dict:
    """Retrieve relevant document chunks."""
    retriever = ResearchRetriever()

    results = retriever.search(
        state["question"],
        top_k=3,
    )

    context_parts = []
    sources = []

    for result in results:
        context_parts.append(
            f"Source: {result['source']}\n"
            f"Content: {result['text']}"
        )

        sources.append({
            "source": result["source"],
            "chunk_id": result["chunk_id"],
            "similarity": result["similarity"],
        })

    context = "\n\n".join(context_parts)

    return {
        "context": context,
        "sources": sources,
    }


def generate_node(state: AgentState) -> dict:
    """Generate an answer grounded in retrieved context."""
    answer = generate_answer(
        question=state["question"],
        context=state["context"],
    )

    return {"answer": answer}


def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


if __name__ == "__main__":
    agent = build_agent()

    print("OpenLLM Research Agent is ready!")

    while True:
        question = input("\nAsk a question (or type 'exit'): ").strip()

        if question.lower() == "exit":
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            result = agent.invoke({
                "question": question,
                "context": "",
                "sources": [],
                "answer": "",
            })

            print("\nAnswer:")
            print(result["answer"])

            print("\nSources:")
            for source in result["sources"]:
                print(
                    f"- {source['source']} "
                    f"(similarity: {source['similarity']:.3f})"
                )

        except Exception as error:
            print(f"Error: {error}")
