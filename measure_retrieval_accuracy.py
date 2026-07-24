"""
measure_retrieval_accuracy.py

Run retrieval verification over the benchmark suite.

Responsibilities
----------------
1. Load benchmark queries.
2. Execute retrieval verification.
3. Print aggregate retrieval metrics.
4. Save deterministic JSON results.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import os 
from dotenv import load_dotenv 

from benchmark.benchmark_loader import BenchmarkLoader

from module_loader import load 

retrieval = load(
    "retrieval_runtime",
    "retrieval.py"
)


VectorRetriever = retrieval.VectorRetriever
RetrievalResult = retrieval.RetrievalResult 

from retrieval_verification import RetrievalVerificationRunner

from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS 


load_dotenv()

root_path = os.getenv("DATA_PATH")

def main() -> None:

    benchmark_loader = BenchmarkLoader(
        benchmark_path=os.path.join(root_path, "benchmark", "benchmark_queries.json"),
    )

    benchmarks = benchmark_loader.load()

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    vector_store = FAISS.load_local(
        folder_path=os.path.join(
            root_path,
            "rag",
            "vector_stores",
            "authoritative_faiss",
        ),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = VectorRetriever(
        vector_store=vector_store,
    )

    runner = RetrievalVerificationRunner(
        retriever=retriever,
        top_k=5,
    )

    results = runner.run(benchmarks)

    print("\nRetrieval Verification Results")
    print("-" * 40)

    print(
        f"Benchmarks               : "
        f"{results.benchmark_count}"
    )

    print(
        f"Top-1 Source Accuracy    : "
        f"{results.top1_accuracy:.3f}"
    )

    print(
        f"Hit@K                    : "
        f"{results.hit_at_k:.3f}"
    )

    mean_rank_display = (
        "N/A"
        if results.mean_first_correct_rank is None
        else f"{results.mean_first_correct_rank:.3f}"
    )

    print(
        f"Mean First Correct Rank  : "
        f"{mean_rank_display}"
    )

    output = {
        "summary": {
            "benchmark_count":
                results.benchmark_count,

            "top1_accuracy":
                results.top1_accuracy,

            "hit_at_k":
                results.hit_at_k,

            "mean_first_correct_rank":
                results.mean_first_correct_rank,
        },

        "verifications": [
            asdict(verification)
            for verification in results.verifications
        ],
    }

    output_path = Path(
        "retrieval_verification_results.json"
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nResults written to {output_path}"
    )


if __name__ == "__main__":
    main()