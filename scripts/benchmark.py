from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from rag_app.core.config import load_settings
from rag_app.core.eval import hit_at_k, load_eval_queries, mrr, summarize
from rag_app.core.pipeline import RAGPipeline
from rag_app.engines.mock_engine import MockEngine


def evaluate_pipeline(name: str, pipeline: RAGPipeline, eval_rows) -> List[Dict[str, float]]:
    """Runs one retrieval approach across eval queries and records metrics."""
    records: List[Dict[str, float]] = []

    for row in eval_rows:
        retrieval = pipeline.retrieve(row.question)
        generated = pipeline.answer(row.question)

        records.append(
            {
                "approach": name,
                "query_id": row.query_id,
                "hit_at_k": hit_at_k(retrieval, row.gold_doc_ids),
                "mrr": mrr(retrieval, row.gold_doc_ids),
                "latency_ms": generated.latency_ms,
            }
        )

    return records


def plot_metrics(summary_rows: List[Dict[str, float]], output_dir: Path) -> None:
    """Writes comparison charts for quality and latency."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_quality_svg(summary_rows, output_dir / "quality_comparison.svg")
    _write_latency_svg(summary_rows, output_dir / "latency_comparison.svg")


def main() -> None:
    """Executes baseline vs hybrid benchmark and saves CSV/SVG artifacts."""
    parser = argparse.ArgumentParser(description="Benchmark baseline vs hybrid RAG")
    parser.add_argument("--settings", default="config/settings.yaml")
    args = parser.parse_args()

    settings = load_settings(args.settings)
    root = Path(args.settings).resolve().parent.parent
    validation = settings.validation

    eval_rows = load_eval_queries((root / settings.paths["eval_path"]).as_posix())

    common = dict(
        corpus_path=(root / settings.paths["corpus_path"]).as_posix(),
        engine=MockEngine(),
        chunk_size_chars=settings.retrieval["chunk_size_chars"],
        chunk_overlap_chars=settings.retrieval["chunk_overlap_chars"],
        top_k=settings.retrieval["top_k"],
        bm25_top_k=settings.retrieval["bm25_top_k"],
        dense_top_k=settings.retrieval["dense_top_k"],
        rerank_top_k=settings.retrieval["rerank_top_k"],
        enforce_citations=validation.get("enforce_citations", True),
        validation_min_supported_chunks=validation.get("min_supported_chunks", 1),
        validation_min_relevance_score=validation.get("min_relevance_score", 0.25),
        embedding_model=settings.retrieval.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        reranker_model=settings.retrieval.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    )

    baseline = RAGPipeline(chunk_mode="fixed", **common)
    hybrid = RAGPipeline(chunk_mode="semantic", **common)

    records = []
    records.extend(evaluate_pipeline("A: fixed + dense", baseline, eval_rows))
    records.extend(evaluate_pipeline("B: semantic + hybrid + rerank", hybrid, eval_rows))

    output_dir = root / settings.paths["output_dir"]
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    detail_csv = output_dir / "benchmark_details.csv"
    with detail_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["approach", "query_id", "hit_at_k", "mrr", "latency_ms"],
        )
        writer.writeheader()
        writer.writerows(records)

    grouped = {}
    for approach in sorted({r["approach"] for r in records}):
        grouped[approach] = summarize([r for r in records if r["approach"] == approach])

    summary_rows = [
        {
            "approach": name,
            "hit_at_k": metrics["hit_at_k"],
            "mrr": metrics["mrr"],
            "latency_ms": metrics["latency_ms"],
        }
        for name, metrics in grouped.items()
    ]

    summary_csv = output_dir / "benchmark_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["approach", "hit_at_k", "mrr", "latency_ms"])
        writer.writeheader()
        writer.writerows(summary_rows)

    plot_metrics(summary_rows, figures_dir)

    print("Benchmark complete.")
    print(f"- Details: {detail_csv}")
    print(f"- Summary: {summary_csv}")
    print(f"- Figures: {figures_dir / 'quality_comparison.svg'}, {figures_dir / 'latency_comparison.svg'}")


def _write_quality_svg(summary_rows: List[Dict[str, float]], path: Path) -> None:
    """Renders a simple SVG bar chart for Hit@K and MRR."""
    width = 900
    height = 460
    margin = 70
    chart_w = width - (2 * margin)
    chart_h = height - (2 * margin)
    group_w = chart_w / max(1, len(summary_rows))
    bar_w = max(20.0, min(80.0, group_w * 0.28))
    colors = {"hit_at_k": "#1b9e77", "mrr": "#377eb8"}

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="36" text-anchor="middle" font-size="20" font-family="Arial">Retrieval Quality Comparison</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#444"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#444"/>',
    ]

    for i in range(6):
        y_val = i / 5
        y = (height - margin) - (chart_h * y_val)
        lines.append(f'<line x1="{margin}" y1="{y:.2f}" x2="{width-margin}" y2="{y:.2f}" stroke="#eee"/>')
        lines.append(f'<text x="{margin-12}" y="{y+4:.2f}" text-anchor="end" font-size="11" font-family="Arial">{y_val:.1f}</text>')

    for i, row in enumerate(summary_rows):
        x_center = margin + (i + 0.5) * group_w
        hit_h = chart_h * row["hit_at_k"]
        mrr_h = chart_h * row["mrr"]
        x1 = x_center - bar_w - 6
        x2 = x_center + 6
        y1 = (height - margin) - hit_h
        y2 = (height - margin) - mrr_h

        lines.append(f'<rect x="{x1:.2f}" y="{y1:.2f}" width="{bar_w:.2f}" height="{hit_h:.2f}" fill="{colors["hit_at_k"]}"/>')
        lines.append(f'<rect x="{x2:.2f}" y="{y2:.2f}" width="{bar_w:.2f}" height="{mrr_h:.2f}" fill="{colors["mrr"]}"/>')
        lines.append(f'<text x="{x_center:.2f}" y="{height-margin+18}" text-anchor="middle" font-size="11" font-family="Arial">{row["approach"]}</text>')

    legend_x = width - margin - 180
    legend_y = margin + 10
    lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="12" height="12" fill="{colors["hit_at_k"]}"/>')
    lines.append(f'<text x="{legend_x+18}" y="{legend_y+11}" font-size="12" font-family="Arial">Hit@K</text>')
    lines.append(f'<rect x="{legend_x+80}" y="{legend_y}" width="12" height="12" fill="{colors["mrr"]}"/>')
    lines.append(f'<text x="{legend_x+98}" y="{legend_y+11}" font-size="12" font-family="Arial">MRR</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_latency_svg(summary_rows: List[Dict[str, float]], path: Path) -> None:
    """Renders a simple SVG bar chart for average latency."""
    width = 900
    height = 460
    margin = 70
    chart_w = width - (2 * margin)
    chart_h = height - (2 * margin)
    max_latency = max([r["latency_ms"] for r in summary_rows] + [1.0])
    bar_w = chart_w / max(1, len(summary_rows)) * 0.55
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="36" text-anchor="middle" font-size="20" font-family="Arial">Average End-to-End Latency</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#444"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#444"/>',
    ]

    for i in range(6):
        val = (max_latency * i) / 5
        y = (height - margin) - (chart_h * (val / max_latency))
        lines.append(f'<line x1="{margin}" y1="{y:.2f}" x2="{width-margin}" y2="{y:.2f}" stroke="#eee"/>')
        lines.append(f'<text x="{margin-12}" y="{y+4:.2f}" text-anchor="end" font-size="11" font-family="Arial">{val:.1f}</text>')

    for i, row in enumerate(summary_rows):
        h = chart_h * (row["latency_ms"] / max_latency)
        x = margin + (i + 0.5) * (chart_w / len(summary_rows)) - (bar_w / 2)
        y = (height - margin) - h
        fill = "#d95f02" if "fixed" in row["approach"] else "#1b9e77"
        lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{fill}"/>')
        lines.append(f'<text x="{x + bar_w / 2:.2f}" y="{height-margin+18}" text-anchor="middle" font-size="11" font-family="Arial">{row["approach"]}</text>')
        lines.append(f'<text x="{x + bar_w / 2:.2f}" y="{y-6:.2f}" text-anchor="middle" font-size="11" font-family="Arial">{row["latency_ms"]:.2f} ms</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
