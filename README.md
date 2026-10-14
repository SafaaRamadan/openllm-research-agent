# OpenLLM Research Agent

A Retrieval-Augmented Generation (RAG) application that answers questions about open-source language models and RAG research using retrieved document evidence.

The project combines semantic search, a local open-source language model, and a LangGraph workflow, with a FastAPI interface for submitting questions and retrieving source references.

## Key Features

- **Semantic Retrieval:** Uses Sentence Transformers to generate multilingual text embeddings.
- **Vector Search:** Stores and searches document embeddings using ChromaDB.
- **Local LLM Inference:** Uses Qwen2.5-0.5B-Instruct in GGUF format through `llama-cpp-python`.
- **RAG Pipeline:** Supplies retrieved passages to the language model to ground its answers in available documents.
- **Agent Workflow:** Uses LangGraph to organize retrieval and answer generation into separate nodes.
- **REST API:** Exposes the application through FastAPI.
- **Retrieval Evaluation:** Includes scripts to evaluate source retrieval and compare chunking configurations.
- **Automated Tests:** Includes pytest tests for document ingestion.

## Architecture

```text
Research Documents
       |
       v
Document Loading & Chunking
       |
       v
Sentence Transformer Embeddings
       |
       v
ChromaDB Vector Store
       |
       v
User Question
       |
       v
Semantic Retrieval
       |
       v
Retrieved Context + Question
       |
       v
Qwen2.5-0.5B-Instruct
       |
       v
Answer + Source References
       |
       v
FastAPI Response
```

The LangGraph workflow currently consists of two nodes:

1. **Retrieve:** Finds relevant document chunks and prepares the context.
2. **Generate:** Uses the retrieved context to generate an answer.

## Tech Stack

- **Language:** Python
- **LLM:** Qwen2.5-0.5B-Instruct (GGUF, Q4_K_M)
- **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Vector Database:** ChromaDB
- **Orchestration:** LangGraph
- **API:** FastAPI
- **Inference:** `llama-cpp-python`
- **Testing:** pytest

## Project Structure

```text
openllm-research-agent/
├── data/
│   └── raw/
│       ├── qwen25_05b.md
│       ├── smollm2_360m.md
│       └── rag_research.md
├── evaluation/
│   ├── questions.json
│   ├── evaluate_retrieval.py
│   └── compare_chunking.py
├── src/
│   └── research_agent/
│       ├── agent.py
│       ├── api.py
│       ├── generator.py
│       ├── index_store.py
│       ├── ingest.py
│       ├── rag_pipeline.py
│       └── retriever.py
├── tests/
│   └── test_ingest.py
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd openllm-research-agent
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Build the vector index

The default configuration uses a chunk size of 500 characters and an overlap of 100 characters.

```bash
python -m src.research_agent.index_store
```

This creates the persistent ChromaDB index using the documents in `data/raw/`. The embedding model and local language model may require downloads on their first use.

### 5. Start the API

```bash
uvicorn src.research_agent.api:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

## API Usage

### Health Check

```http
GET /health
```

Checks whether the API is running.

### Ask a Question

```http
POST /ask
Content-Type: application/json
```

Example request:

```json
{
  "question": "What is the context length of Qwen2.5-0.5B-Instruct?"
}
```

The response includes the submitted question, the generated answer, and source references returned by the retrieval pipeline. See `/docs` for the exact response schema.

## Retrieval Evaluation

Run the retrieval evaluation:

```bash
python -m evaluation.evaluate_retrieval
```

The evaluation compares retrieved source filenames against the expected source for each question.

The current evaluation set contains 10 questions covering Qwen2.5, SmolLM2, and RAG concepts.

### Chunking Experiment

Run the chunking comparison:

```bash
python -m evaluation.compare_chunking
```

The experiment compares these configurations:

| Chunk Size | Overlap |
|---:|---:|
| 250 characters | 50 characters |
| 500 characters | 100 characters |
| 800 characters | 100 characters |

In the current 10-question evaluation, all three configurations achieved:

- **Hit@1:** 100%
- **Source Hit@3:** 100%

These results indicate successful expected-source retrieval on this evaluation set. They do not establish answer correctness or guarantee performance on unseen questions.

Note: The experiment rebuilds the persistent index for each configuration. Rebuild the index with your chosen settings before running the application afterward.

## Run Tests

```bash
python -m pytest -q
```

The current test suite contains three passing tests for the implemented document-ingestion functionality.

## Limitations

- Retrieval and generation quality depend on the supplied documents and their relevance to the question.
- A small evaluation set cannot establish general retrieval performance.
- The language model can produce incorrect answers, including arithmetic errors, even when relevant evidence is retrieved.
- Source retrieval metrics do not measure factual correctness of generated answers.
- The application is a research prototype and should be evaluated further before production use.

## Future Improvements

- Expand the evaluation dataset with more questions and difficult retrieval cases.
- Evaluate answer faithfulness and factual correctness separately from retrieval quality.
- Improve handling of numerical comparisons and calculations.
- Add structured logging and more comprehensive API and pipeline tests.
- Benchmark inference latency and resource usage.

