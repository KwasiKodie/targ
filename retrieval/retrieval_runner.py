"""
retrieval_runner.py

Run retrieval inspection for all benchmark queries.

Responsibilities
----------------
- Iterate over benchmark queries.
- Invoke RetrievalInspector.
- Collect RetrievalInspection objects.
- Persist retrieval results.

The RetrievalRunner does NOT:
- perform retrieval;
- compute retrieval metrics;
- assess relevance. 

Author:
Training-Free Adaptive Retrieval Gating (TARG) Reproduction
"""

from __future__ import annotations 

import json
from pathlib import Path 

from benchmark.benchmark_loader import BenchmarkLoader 
from retrieval.retrieval_inspector import RetrievalInspector
from retrieval.retrieval_models import RetrievalInspection 

class RetrievalRunner:
    """
    Orchestrates retrieval inspection over the benchmark.
    """

    def __init__(
        self,
        *,
        benchmark_loader: BenchmarkLoader,
        retrieval_inspector: RetrievalInspector,
    ) -> None: 
        
        self.benchmark_loader = benchmark_loader 
        self.retrieval_inspector = retrieval_inspector

        self._validate_constructor()

    # ----------------------------------------------------
    # Public API
    # ----------------------------------------------------

    def run(
        self,
    ) -> tuple[RetrievalInspection, ...]:
        """
        Run retrieval inspection over the benchmark.
        """

        benchmark = self.benchmark_loader.load()

        inspections = tuple(
            self.retrieval_inspector.inspect(query)
            for query in benchmark 
        )

        return inspections 
    
    def run_and_save(
        self,
        output_path: str | Path,
    ) -> tuple[RetrievalInspection, ...]:
        """
        Run retrieval inspection and save results.
        """

        inspections = self.run()

        self.save(
            inspections=inspections,
            output_path=output_path,
        )

        return inspections 
    
    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------

    def save(
        self,
        *,
        inspections: tuple[RetrievalInspection, ...],
        output_path: str | Path,
    ) -> None: 
        
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True, 
        )

        payload = [
            inspection.to_dict()
            for inspection in inspections
        ]

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as fp:
            
            json.dump(
                payload, 
                fp,
                indent=2,
                ensure_ascii=False,
            )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def _validate_constructor(
        self,
    ) -> None: 
        
        if self.benchmark_loader is None:
            raise TypeError(
                "benchmark_loader cannot be None."
            )
        
        if self.retrieval_inspector is None:
            raise TypeError(
                "retrieval_inspector cannot be None."
            )