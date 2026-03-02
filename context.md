# Context Summary

## Request Completed
You asked to rewrite the project using:
- transcript 1: *5 AI Portfolio Projects That Will Actually Get You Hired in 2026*
- transcript 2: *How to Build a Scalable RAG System for AI Apps (Full Architecture)*

And then:
- update `README.md`
- summarize everything in `context.md`

This file is that summary.

## What Was Rewritten

## 1) README Rewrite (Transcript-Aligned)
`README.md` is now organized around:
- the 5 portfolio projects from transcript 1
- deep implementation details for Project 1 (production RAG)
- architecture mapping from transcript 2
- clear "implemented now" vs "next iteration" boundaries
- local run + benchmark + API usage instructions

## 2) Production-Style Validation Added
New file:
- `src/rag_app/core/validation.py`

Validation nodes added (lightweight heuristics matching transcript intent):
- `gatekeeper`: confirms retrieved chunks support query terms
- `auditor`: checks overlap strength between query and retrieved context
- `strategist`: checks retrieval confidence floor

If validation fails and citation enforcement is on, pipeline declines safely instead of hallucinating.

## 3) Citation Enforcement Wired End-to-End
Updated:
- `src/rag_app/core/pipeline.py`

Behavior:
- runs validation before generation
- returns safe decline response if evidence is weak

## 4) Configurable Validation Thresholds
Updated:
- `config/settings.yaml`
- `src/rag_app/core/config.py`
- `scripts/run_query.py`
- `scripts/benchmark.py`
- `src/rag_app/api/app.py`

New config section:
- `validation.enforce_citations`
- `validation.min_supported_chunks`
- `validation.min_overlap_terms`
- `validation.min_retrieval_score`

## Transcript-to-Implementation Mapping

### From "5 AI Portfolio Projects..."
Implemented here:
- Project 1: production-grade RAG
- Core pieces of Project 3 discipline: measurable evaluation, quality/latency artifacts

Documented as roadmap in README:
- Projects 2, 4, 5 with concrete build direction

### From "Scalable RAG Architecture"
Implemented in simplified local form:
- hybrid retrieval + reranking
- retrieval/augmentation/generation pipeline
- validation nodes before final answer
- quantitative retrieval + latency evaluation

Not fully implemented yet:
- parser for complex tables/layout (true structure parser)
- planner + multi-agent orchestration
- stress-testing/red-team harness
- observability stack with tracing dashboards

## Current Artifacts

Generated benchmark outputs:
- `outputs/benchmark_details.csv`
- `outputs/benchmark_summary.csv`
- `outputs/figures/quality_comparison.svg`
- `outputs/figures/latency_comparison.svg`

## Notes
- The code remains runnable in local/offline-friendly mode (`mock`) and is backend-pluggable for `mlx` and `vllm`.
- The transcript architecture is represented faithfully at the system-design level, with practical simplifications for a local project.
