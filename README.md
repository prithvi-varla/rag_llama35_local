# Local RAG App

This project is a local Python RAG application with three main parts: retrieval, generation, and observability.

It loads source documents from `data/corpus.jsonl`, splits them into chunks with LangChain, indexes them into an in-process Chroma vector store, combines dense retrieval with BM25 in hybrid mode, reranks with a cross-encoder, and then sends the retrieved context to an LLM backend.

Supported generation backends are `mock`, `mlx`, and `vllm`. Langfuse is integrated as optional observability and should never break runtime if it fails.

## Current Architecture

- Source data is stored in `data/corpus.jsonl`.
- Evaluation questions are stored in `data/eval_queries.jsonl`.
- Runtime retrieval pipeline is implemented in [pipeline.py](/Users/prithvirajvarla/Documents/Playground/rag_llama35_local/src/rag_app/core/pipeline.py).
- Observability wrapper is implemented in [observability.py](/Users/prithvirajvarla/Documents/Playground/rag_llama35_local/src/rag_app/core/observability.py).
- API entrypoint is [app.py](/Users/prithvirajvarla/Documents/Playground/rag_llama35_local/src/rag_app/api/app.py).
- CLI query runner is [run_query.py](/Users/prithvirajvarla/Documents/Playground/rag_llama35_local/scripts/run_query.py).
- Benchmark runner is [benchmark.py](/Users/prithvirajvarla/Documents/Playground/rag_llama35_local/scripts/benchmark.py).

## How Data Is Stored

Persistent source data lives in `data/corpus.jsonl`.

At runtime, the app:
- reads the JSONL file,
- converts rows into LangChain documents,
- splits them into chunks,
- adds those chunks into a Chroma store created in memory.

The current Chroma store is not persisted to disk because no `persist_directory` is configured. That means the index is rebuilt on each app start.

## Retrieval Flow

In `semantic` mode, the app:
- chunks documents with `RecursiveCharacterTextSplitter`,
- embeds chunks with `HuggingFaceEmbeddings`,
- stores vectors in Chroma,
- retrieves with both BM25 and vector similarity,
- combines them with `EnsembleRetriever`,
- reranks with `CrossEncoderReranker`,
- filters weak matches with `EmbeddingsFilter`.

In `fixed` mode, the app uses only dense retrieval over the same chunked/indexed data path.

## Generation Flow

After retrieval, the selected backend generates the answer:
- `mock` returns a deterministic mock response,
- `mlx` loads a local MLX model and generates on Apple Silicon,
- `vllm` calls an OpenAI-compatible vLLM server.

The MLX model path is configured in `config/settings.yaml` as `mlx-community/Llama-3.2-3B-Instruct-4bit`.

## Observability

Langfuse is optional.

If enabled and configured, the app records:
- query start,
- retrieval event,
- validation event,
- completion or error event.

If Langfuse is unavailable, misconfigured, or throws SDK errors, the app continues running without failing the request.

Configuration lives in `config/settings.yaml` under `observability`.

## Setup

```bash
cd /Users/prithvirajvarla/Documents/Playground/rag_llama35_local
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Run A Query

```bash
PYTHONPATH=src python3 scripts/run_query.py \
  --settings config/settings.yaml \
  --mode hybrid \
  --query "Why combine BM25 with dense retrieval?"
```

## Start The API

```bash
PYTHONPATH=src uvicorn rag_app.api:app --host 0.0.0.0 --port 8000
```

## Benchmark

```bash
PYTHONPATH=src python3 scripts/benchmark.py --settings config/settings.yaml
```

Outputs are written to:
- `outputs/benchmark_details.csv`
- `outputs/benchmark_summary.csv`
- `outputs/figures/quality_comparison.svg`
- `outputs/figures/latency_comparison.svg`

## Configuration

Main config file: [settings.yaml](/Users/prithvirajvarla/Documents/Playground/rag_llama35_local/config/settings.yaml)

Important sections:
- `inference`: backend selection and backend-specific settings
- `retrieval`: chunking, embedding, reranker, and top-k settings
- `validation`: minimum support thresholds
- `observability`: Langfuse settings
- `paths`: corpus, eval, and output paths

## Dependencies

Current runtime depends on:
- LangChain
- Chroma
- sentence-transformers
- rank-bm25
- FastAPI
- Langfuse
- MLX or vLLM depending on backend choice

Dependency list is in [requirements.txt](/Users/prithvirajvarla/Documents/Playground/rag_llama35_local/requirements.txt).
