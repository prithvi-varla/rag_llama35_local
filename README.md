# AI Portfolio 2026: Production RAG Project (Built from Transcript Requirements)

This repository implements **Project 1** from the transcript *"5 AI Portfolio Projects That Will Actually Get You Hired in 2026"* and uses the architecture from *"How to Build a Scalable RAG System for AI Apps (Full Architecture)"*.

The goal is not a demo chatbot. The goal is a **production-style RAG system** with:
- hybrid retrieval
- reranking
- citation enforcement
- offline evaluation
- API serving
- measurable approach comparison

## The 5 Portfolio Projects (Transcript-Aligned)

1. **Production-Grade RAG System**
- This repo is the implementation.
- Focus: retrieval quality, grounded answers, evaluation discipline.

2. **Local Offline Assistant (3B-7B model)**
- Build local CLI/API assistant with schema-constrained outputs.
- Benchmark tokens/sec, latency, and quality across models.

3. **RAG Monitoring + Observability**
- Add tracing, citation coverage, p50/p95 latency, failure-rate dashboards.
- Add regression gates in CI.

4. **Task-Specific Fine-Tuning (LoRA)**
- Improve a narrow task (e.g., extraction/tool-call accuracy).
- Compare base vs tuned with quantitative metrics.

5. **Agentic Workflow System**
- Multi-step planner/executor pipeline with tool use, retries, and safety guards.
- Evaluate reliability and failure modes under adversarial prompts.

---

## Project 1 Design (Phased, as described in transcript)

## Phase 1: Core RAG Fundamentals
- Document ingestion from JSONL corpus.
- Chunking with overlap.
- Dense retrieval baseline.
- Context-grounded answer generation with cited chunks.

Code:
- `src/rag_app/core/io.py`
- `src/rag_app/core/chunking.py`
- `src/rag_app/core/retrieval.py`
- `scripts/run_query.py`

## Phase 2: Production Retrieval Quality
- **Hybrid retrieval**: keyword + semantic scoring.
- **Reranking**: rescoring candidates for precision.
- **Citation enforcement**: decline when support is weak.

Code:
- `src/rag_app/core/retrieval.py`
- `src/rag_app/core/rerank.py`
- `src/rag_app/core/validation.py`
- `src/rag_app/core/pipeline.py`

## Phase 3: Shippable Evaluation Discipline
- Golden eval query set (`data/eval_queries.jsonl`).
- Offline benchmark script.
- Output artifacts for quality + latency comparison.

Code:
- `src/rag_app/core/eval.py`
- `scripts/benchmark.py`
- `outputs/benchmark_summary.csv`
- `outputs/figures/*.svg`

---

## Scalable RAG Architecture Mapping (Second Transcript)

Implemented directly:
- retrieval -> augmentation -> generation loop
- hybrid retrieval (semantic + keyword)
- reranking layer
- validation nodes (gatekeeper/auditor/strategist heuristics)
- quantitative evaluation + latency tracking

Represented in repo design (simplified for local run):
- structure-aware chunking mode (`semantic`)
- configurable validation thresholds in `config/settings.yaml`
- API serving surface for production integration

Not fully implemented yet (noted for next iteration):
- real table-aware parser for PDFs/HTML
- full planner + multi-agent orchestration
- prompt injection red-team harness
- online tracing stack (Langfuse/LangSmith)

---

## Inference Backends

- `mock` for offline development/evaluation.
- `mlx` for Apple Silicon local inference.
- `vllm` for high-throughput serving.

Configured in `config/settings.yaml`.

Model note:
- Requested: "llama3.5b class"
- Practical default used: Llama 3.x 3B-class checkpoint (`meta-llama/Llama-3.2-3B-Instruct`) because 3.5B is not a standard public release tier.

---

## Quick Start

```bash
cd /Users/prithvirajvarla/Documents/Playground/rag_llama35_local
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Optional real engines:

```bash
# Apple Silicon local inference
pip install mlx mlx-lm

# vLLM serving stack
pip install vllm
```

Run one query:

```bash
PYTHONPATH=src python3 scripts/run_query.py \
  --settings config/settings.yaml \
  --mode hybrid \
  --query "Why combine BM25 with dense retrieval?"
```

Run benchmark + generate comparison graphs:

```bash
PYTHONPATH=src python3 scripts/benchmark.py --settings config/settings.yaml
```

Start API:

```bash
PYTHONPATH=src uvicorn rag_app.api:app --host 0.0.0.0 --port 8000
```

Call API:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "content-type: application/json" \
  -d '{"query":"What metrics should be monitored in production RAG?"}'
```

---

## Approach Comparison Outputs

Generated files:
- `outputs/benchmark_details.csv`
- `outputs/benchmark_summary.csv`
- `outputs/figures/quality_comparison.svg`
- `outputs/figures/latency_comparison.svg`

Interpretation:
- Use quality metrics (`Hit@K`, `MRR`) to justify retrieval design.
- Use latency metrics to validate SLA impact.
- Keep hybrid+rereank when quality gain justifies latency overhead.

---

## Production Notes

Current production-style controls:
- config-driven architecture
- runtime backend swap (`mock` / `mlx` / `vllm`)
- citation enforcement with safe-decline behavior
- API health endpoint and typed API responses
- offline benchmark artifacts for regression tracking

Recommended next additions:
1. CI regression gate using benchmark thresholds.
2. Full tracing + token cost telemetry.
3. Prompt version files and A/B prompt testing.
4. Red-team prompt injection and data leakage tests.
