
from llama_cpp import Llama

MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

_model = None


def load_model():
    """Load the model once and reuse it."""
    global _model

    if _model is None:
        print("Loading Qwen model...")

        _model = Llama.from_pretrained(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )

        print("Model loaded successfully!")

    return _model


def generate_answer(question: str, context: str) -> str:
    model = load_model()

    response = model.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an open-source LLM research assistant. "
                    "Answer the user's question using the supplied context "
                    "as your primary and only source of factual information. "
                    "Identify the information relevant to the question before "
                    "forming your answer. "
                    "For comparisons, analyze the relevant values and criteria "
                    "for each item, then state the conclusion clearly and "
                    "consistently with the evidence. "
                    "Perform necessary calculations carefully. "
                    "Distinguish between facts explicitly stated in the context "
                    "and conclusions logically derived from those facts. "
                    "Do not make unsupported assumptions or invent facts. "
                    "If the context lacks sufficient information, clearly state "
                    "that the available sources are insufficient. "
                    "Cite the provided source filenames when possible. "
                    "Answer directly and concisely, addressing every part "
                    "of the user's question."
                ),


            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{question}"
                ),
            },
        ],
        max_tokens=180,
        temperature=0.0,
        repeat_penalty=1.15,
    )

    return response["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    answer = generate_answer(
        question="How many parameters does Qwen2.5-0.5B have?",
        context=(
            "Source: qwen25_05b.md\n"
            "The Qwen2.5-0.5B-Instruct model has approximately "
            "0.49 billion total parameters."
        ),
    )

    print("\nAnswer:")
    print(answer)
