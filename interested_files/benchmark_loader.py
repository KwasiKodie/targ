"""
benchmark_loader.py

Load benchmark queries from JSON. 

Responsibilities
----------------
- Read benchmark JSON.
- Validate JSON structure.
- Construct immutable Benchmark models.

The BenchmarkLoader does NOT:
- perform retrieval;
- compute metrics;
- modify benchmark data. 

Author: 
Training-Free Adaptive Retrieval Gating (TARG) Reproduction
"""

from __future__ import annotations

import json 
from pathlib import Path 
from typing import Any 

from benchmark.benchmark_models import (
    Benchmark, 
    BenchmarkQuery,
)


class BenchmarkLoader:
    """
    Load benchmark queries from disk.
    """

    def __init__(
        self,
        *,
        benchmark_path: str | Path,
    ) -> None: 
        
        self.benchmark_path = Path(benchmark_path)

        self._validate_constructor()

    # -------------------------------------------------------
    # Public API
    # -------------------------------------------------------

    def load(
        self,
    ) -> Benchmark: 
        """
        Load the benchmark from disk.
        """

        payload = self._read_json()

        queries = tuple(
            self._build_query(item)
            for item in payload 
        )

        return Benchmark(
            queries=queries,
        )
    
    # -----------------------------------------------------
    # JSON
    # -----------------------------------------------------

    def _read_json(
        self,
    ) -> list[dict[str, Any]]:
        """
        Read and validate the benchmark JSON.
        """

        with self.benchmark_path.open(
            "r",
            encoding="utf-8",
        ) as fp:
            
            payload = json.load(fp)

        if not isinstance(payload, list):
            raise TypeError(
                "Benchmark JSON must contain a list."
            )
        
        if len(payload) == 0:
            raise ValueError(
                "Benchmark JSON is empty."
            )
        
        return payload 
    
    # ------------------------------------------------------
    # Model construction
    # ------------------------------------------------------

    def _build_query(
        self,
        item: dict[str, Any],
    ) -> BenchmarkQuery:
        """
        Convert one JSON object into a BenchmarkQuery.
        """

        if not isinstance(item, dict):
            raise TypeError(
                "Each benchmark entry must be an object."
            )
        
        self._validate_fields(item)

        return BenchmarkQuery(
            benchmark_id=item["benchmark_id"],
            question=item["question"],
            expected_source=tuple(item["expected_sources"]),
            supporting_pages=tuple(item["supporting_pages"]),
            expected_answer_span=item["expected_answer_span"],
            difficulty=item["difficulty"],
            topic=item["topic"],
        )
    
    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def _validate_constructor(
        self,
    ) -> None: 
        
        if not self.benchmark_path.exists():
            raise FileNotFoundError(
                f"Benchmark file not found: "
                f"{self.benchmark_path}"
            )
        
        if not self.benchmark_path.is_file():
            raise ValueError(
                "benchmark_path must refer to a file."
            )
        
        if self.benchmark_path.suffix.lower() != ".json":
            raise ValueError(
                "Benchmark file must be JSON."
            )
        
    def _validate_fields(
        self,
        item: dict[str, Any],
    ) -> None: 
        """
        Validate required fields.
        """

        required = (
            "benchmark_id",
            "question",
            "expected_sources",
            "supporting_pages",
            "expected_answer_span",
            "difficulty",
            "topic",
        )

        missing = [
            field 
            for field in required 
            if field not in item 
        ]

        if missing:
            raise KeyError(
                "Missing benchmark fields: "
                + ", ".join(missing)
            )