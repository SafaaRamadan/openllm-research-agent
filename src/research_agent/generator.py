
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
                    "Answer using only the supplied context. "
                    "Cite the provided source names when possible. "
                    "If the context does not contain the answer, "
                    "clearly say that the available sources are insufficient. "
                    "Never invent facts."
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
        max_tokens=250,
        temperature=0.1,
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
