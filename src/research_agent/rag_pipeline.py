
from src.research_agent.retriever import ResearchRetriever
from src.research_agent.generator import generate_answer


class ResearchRAG:
    def __init__(self):
        print("Initializing retriever...")
        self.retriever = ResearchRetriever()
        print("RAG pipeline initialized!")

    def ask(self, question: str) -> dict:
        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        # Step 1: Retrieve relevant chunks
        results = self.retriever.search(question, top_k=3)

        # Step 2: Prepare context with source names
        context_parts = []

        for result in results:
            context_parts.append(
                f"Source: {result['source']}\n"
                f"Content: {result['text']}"
            )

        context = "\n\n".join(context_parts)

        # Step 3: Generate an answer using retrieved context
        answer = generate_answer(
            question=question,
            context=context,
        )

        # Step 4: Return answer and source information
        sources = []

        for result in results:
            source = {
                "source": result["source"],
                "similarity": result["similarity"],
            }

            if source not in sources:
                sources.append(source)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }


if __name__ == "__main__":
    rag = ResearchRAG()

    while True:
        question = input("\nAsk a research question (or type 'exit'): ")

        if question.strip().lower() == "exit":
            break

        try:
            result = rag.ask(question)

            print("\nAnswer:")
            print(result["answer"])

            print("\nRetrieved sources:")
            for source in result["sources"]:
                print(
                    f"- {source['source']} "
                    f"(similarity: {source['similarity']:.3f})"
                )

        except Exception as error:
            print(f"Error: {error}")
