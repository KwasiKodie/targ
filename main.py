"""
main.py

CLI entry point for the complete TARG workflow.

Commands:
    prepare    Build and persist the authoritative FAISS vector store.
    calibrate  Run Stage 2.5 and save a calibrated threshold.
    ask        Load persisted artifacts and answer questions.
    run        Execute preparation, calibration, and inference end to end.
    benchmark  Collect raw benchmark inference data for notebook analysis.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer

from answer_evaluator import AnswerEvaluator
from answer_generator import AnswerGenerator
from benchmark.benchmark_loader import BenchmarkLoader
from corpus.chunker import Chunker
from corpus.development_record import DevelopmentRecord
from corpus.representative_corpus import RepresentativeCorpusBuilder
from corpus.stage2_5_experiment import Stage2_5ExperimentRunner

from parallel_experiment_executor import (
    ParallelExperimentExecutor,
    ParallelExperimentConfig,
)
from parallel_benchmark_executor import (
    BenchmarkDatasetWriter,
    BenchmarkRecord,
    ParallelBenchmarkConfig,
    ParallelBenchmarkExecutor,
)
from corpus.threshold_calibrator import ThresholdCalibrator
from corpus.vector_store_builder import VectorStoreBuilder
from draft_generator import DraftGenerator
from margin_uncertainty_scorer import MarginUncertaintyScorer
from module_loader import load
from retrieval_gate import RetrievalGate
from targ_application import CalibrationStrategy, TARGApplication
from targ_pipeline import TARGPipeline

retrieval_module = load("retrieval_runtime", "retrieval.py")
VectorRetriever = retrieval_module.VectorRetriever

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
DEFAULT_EMBEDDING_MODEL = "intfloat/e5-base-v2"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_MINIMUM_CHUNK_SIZE = 200
DEFAULT_PREFIX_LENGTH = 20
DEFAULT_BETA = 3.0
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_RETRIEVAL_TOP_K = 5


def resolve_data_root(explicit_path: str | None) -> Path:
    raw = explicit_path or os.getenv("DATA_PATH")
    return Path(raw).expanduser().resolve() if raw else PROJECT_ROOT


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = resolve_data_root(args.data_path)
    return {
        "data_root": root,
        "pdf_directory": Path(args.pdf_directory).expanduser().resolve()
        if args.pdf_directory else root / "rag" / "knowledge",
        "vector_store_directory": Path(args.vector_store_directory).expanduser().resolve()
        if args.vector_store_directory else root / "rag" / "vector_stores" / "authoritative_faiss",
        "benchmark_path": Path(args.benchmark_path).expanduser().resolve()
        if args.benchmark_path else root / "benchmark" / "benchmark_queries.json",
        "calibration_path": Path(args.calibration_path).expanduser().resolve()
        if args.calibration_path else root / "threshold.json",
    }


def build_embeddings(model_name: str, *, device: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_generation_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    return model, tokenizer


def build_generation_components(args: argparse.Namespace):
    model, tokenizer = load_generation_model(args.model_name)
    return (
        DraftGenerator(model=model, tokenizer=tokenizer, prefix_length=args.prefix_length),
        MarginUncertaintyScorer(beta=args.beta),
        AnswerGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            add_special_tokens=True,
        ),
        AnswerEvaluator(),
    )


def build_chunker(args: argparse.Namespace) -> Chunker:
    return Chunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        minimum_chunk_size=args.minimum_chunk_size,
    )


def build_vector_store_builder(embeddings: HuggingFaceEmbeddings) -> VectorStoreBuilder:
    return VectorStoreBuilder(
        embedding_model=embeddings,
        vector_store_class=FAISS,
    )


def load_vector_store(directory: Path, embeddings: HuggingFaceEmbeddings) -> FAISS:
    if not directory.exists():
        raise FileNotFoundError(
            f"Vector store does not exist: {directory}. Run `python main.py prepare` first."
        )
    return FAISS.load_local(
        folder_path=str(directory),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def persist_vector_store(directory: Path):
    def save(vector_store: FAISS) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(directory))
        return directory
    return save


def command_prepare(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    embeddings = build_embeddings(args.embedding_model, device=args.embedding_device)
    documents = RepresentativeCorpusBuilder(
        pdf_directory=paths["pdf_directory"]
    ).build()
    if not documents:
        raise RuntimeError(f"No corpus documents were produced from {paths['pdf_directory']}.")

    chunks = build_chunker(args).chunk_documents(documents)
    if not chunks:
        raise RuntimeError("No chunks were produced from the authoritative corpus.")

    vector_store = build_vector_store_builder(embeddings).build(chunks)
    output = paths["vector_store_directory"]
    output.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(output))

    print("\nCorpus preparation completed.")
    print(f"Corpus documents : {len(documents)}")
    print(f"Chunks indexed   : {len(chunks)}")
    print(f"Vector store     : {output}")
    return 0


def command_calibrate(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    embeddings = build_embeddings(args.embedding_model, device=args.embedding_device)
    vector_store = load_vector_store(paths["vector_store_directory"], embeddings)
    retriever = VectorRetriever(vector_store=vector_store)

    draft_generator, uncertainty_scorer, answer_generator, answer_evaluator = (
        build_generation_components(args)
    )

    benchmark = BenchmarkLoader(benchmark_path=paths["benchmark_path"]).load()
    development_dataset = tuple(
        DevelopmentRecord(
            example_id=query.benchmark_id,
            question=query.question,
            reference_answer=query.expected_answer_span,
        )
        for query in benchmark
    )
    if not development_dataset:
        raise RuntimeError(f"No benchmark examples were loaded from {paths['benchmark_path']}.")

    if args.parallel:

        executor = ParallelExperimentExecutor(
            config=ParallelExperimentConfig(
                model_name=args.model_name,
                embedding_model_name=args.embedding_model,
                vector_store_directory=paths["vector_store_directory"],
                device=args.device,
                worker_count=args.workers,
                gpu_ids=(
                    tuple(args.gpu_ids)
                    if args.gpu_ids is not None
                    else None
                ),
                prefix_length=args.prefix_length,
                beta=args.beta,
                max_new_tokens=args.max_new_tokens,
                retrieval_top_k=args.retrieval_top_k,
                embedding_device=args.embedding_device,
            )
        )

        experiment = executor.run(
            development_dataset=development_dataset,
        )

    else:

        runner = Stage2_5ExperimentRunner(
            draft_generator=draft_generator,
            uncertainty_scorer=uncertainty_scorer,
            answer_generator=answer_generator,
            retriever=retriever,
            answer_evaluator=answer_evaluator,
        )

        experiment = runner.run(
            development_dataset=development_dataset,
        )

    calibrator = ThresholdCalibrator()
    if args.strategy == CalibrationStrategy.ACCURACY.value:
        calibration = calibrator.calibrate_for_accuracy(
            experiment.development_examples
        )
    else:
        calibration = calibrator.calibrate_for_budget(
            experiment.development_examples,
            target_retrieval_rate=args.target_retrieval_rate,
        )

    saved_path = calibrator.save(calibration, paths["calibration_path"])
    print("\nThreshold calibration completed.")
    print(f"Strategy             : {args.strategy}")
    print(f"Threshold            : {calibration.threshold}")
    print(f"Calibration artifact : {saved_path}")
    return 0


def build_runtime_pipeline(args: argparse.Namespace, paths: dict[str, Path]):
    embeddings = build_embeddings(args.embedding_model, device=args.embedding_device)
    vector_store = load_vector_store(paths["vector_store_directory"], embeddings)
    retriever = VectorRetriever(vector_store=vector_store)
    calibration = ThresholdCalibrator.load(paths["calibration_path"])
    draft_generator, uncertainty_scorer, answer_generator, _ = build_generation_components(args)

    pipeline = TARGPipeline(
        draft_generator=draft_generator,
        scorer=uncertainty_scorer,
        gate=RetrievalGate(calibration=calibration),
        retriever=retriever,
        answer_generator=answer_generator,
    )
    return pipeline, calibration


def print_pipeline_result(result, calibration) -> None:
    generated_text = getattr(result.answer, "generated_text", result.answer)
    print("\nAnswer\n------")
    print(generated_text)
    print("\nDecision\n--------")
    print(f"Threshold (τ)       : {calibration.threshold}")
    print(f"Uncertainty score   : {result.margin.score}")
    print(f"Retrieval triggered : {result.gate.retrieve}")
    print(f"Retrieved documents : {len(result.retrieval.documents)}")
    print("\nTiming\n------")
    for stage, seconds in result.timing.items():
        print(f"{stage:20}: {seconds:.3f} s")


def command_ask(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    if not paths["calibration_path"].exists():
        raise FileNotFoundError(
            f"Calibration artifact does not exist: {paths['calibration_path']}. "
            "Run `python main.py calibrate` first."
        )

    pipeline, calibration = build_runtime_pipeline(args, paths)
    if args.question:
        print_pipeline_result(pipeline.run(args.question.strip()), calibration)
        return 0

    print("\nTARG interactive mode. Type `exit` or `quit` to stop.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            print_pipeline_result(pipeline.run(question), calibration)
    return 0


def _benchmark_record(query) -> BenchmarkRecord:
    """Convert a loaded benchmark query into a picklable runtime record."""
    expected_sources = getattr(
        query,
        "expected_sources",
        getattr(query, "expected_source", ()),
    )
    return BenchmarkRecord(
        benchmark_id=str(query.benchmark_id),
        question=str(query.question),
        expected_answer=str(query.expected_answer_span),
        expected_sources=tuple(str(item) for item in expected_sources),
        section_title=(
            str(query.section_title)
            if getattr(query, "section_title", None) is not None
            else None
        ),
        supporting_pages=tuple(
            int(item) for item in getattr(query, "supporting_pages", ())
        ),
        difficulty=(
            str(query.difficulty)
            if getattr(query, "difficulty", None) is not None
            else None
        ),
        topic=(
            str(query.topic)
            if getattr(query, "topic", None) is not None
            else None
        ),
    )


def command_benchmark(args: argparse.Namespace) -> int:
    """Run every benchmark query and persist raw observations only."""
    paths = resolve_paths(args)
    if not paths["calibration_path"].exists():
        raise FileNotFoundError(
            f"Calibration artifact does not exist: {paths['calibration_path']}. "
            "Run `python main.py calibrate` for this model first."
        )

    benchmark = BenchmarkLoader(benchmark_path=paths["benchmark_path"]).load()
    records = tuple(_benchmark_record(query) for query in benchmark)
    if not records:
        raise RuntimeError(
            f"No benchmark examples were loaded from {paths['benchmark_path']}."
        )

    executor = ParallelBenchmarkExecutor(
        config=ParallelBenchmarkConfig(
            model_name=args.model_name,
            embedding_model_name=args.embedding_model,
            vector_store_directory=paths["vector_store_directory"],
            calibration_path=paths["calibration_path"],
            device=args.device,
            worker_count=args.workers,
            gpu_ids=tuple(args.gpu_ids) if args.gpu_ids is not None else None,
            prefix_length=args.prefix_length,
            beta=args.beta,
            max_new_tokens=args.max_new_tokens,
            retrieval_top_k=args.retrieval_top_k,
            embedding_device=args.embedding_device,
        )
    )
    results = executor.run(records=records)

    jsonl_path = BenchmarkDatasetWriter.write_jsonl(
        results,
        args.output_jsonl,
        append=args.append,
    )
    csv_path = BenchmarkDatasetWriter.write_csv(
        results,
        args.output_csv,
        append=args.append,
    )

    print("\nBenchmark data collection completed.")
    print(f"Model          : {args.model_name}")
    print(f"Queries        : {len(results)}")
    print(f"JSONL dataset  : {jsonl_path}")
    print(f"CSV dataset    : {csv_path}")
    print("Analysis       : deferred to Jupyter Notebook")
    return 0


def command_run(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    embeddings = build_embeddings(args.embedding_model, device=args.embedding_device)
    draft_generator, uncertainty_scorer, answer_generator, answer_evaluator = (
        build_generation_components(args)
    )

    application = TARGApplication(
        chunker=build_chunker(args),
        vector_store_builder=build_vector_store_builder(embeddings),
        draft_generator=draft_generator,
        uncertainty_scorer=uncertainty_scorer,
        answer_generator=answer_generator,
        answer_evaluator=answer_evaluator,
        threshold_calibrator=ThresholdCalibrator(),
        calibration_path=paths["calibration_path"],
        retrieval_top_k=args.retrieval_top_k,
        vector_store_persist=persist_vector_store(paths["vector_store_directory"]),
    )

    corpus_result, calibration_result, inference_results = application.run(
        pdf_directory=paths["pdf_directory"],
        benchmark_path=paths["benchmark_path"],
        questions=tuple(args.question),
        calibration_strategy=args.strategy,
        target_retrieval_rate=(
            args.target_retrieval_rate
            if args.strategy == CalibrationStrategy.BUDGET.value
            else None
        ),
    )

    print("\nComplete TARG workflow finished.")
    print(f"Corpus documents     : {corpus_result.document_count}")
    print(f"Chunks indexed       : {corpus_result.chunk_count}")
    print(f"Vector store         : {paths['vector_store_directory']}")
    print(f"Calibration strategy : {calibration_result.strategy.value}")
    print(f"Threshold            : {calibration_result.calibration.threshold}")
    print(f"Calibration artifact : {calibration_result.calibration_path}")

    for index, result in enumerate(inference_results, start=1):
        print(f"\nQuestion {index}\n==========")
        print_pipeline_result(result.pipeline, calibration_result.calibration)
    return 0


def add_common_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-path")
    parser.add_argument("--pdf-directory")
    parser.add_argument("--vector-store-directory")
    parser.add_argument("--benchmark-path")
    parser.add_argument("--calibration-path")


def add_embedding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-device", default="cpu")


def add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prefix-length", type=int, default=DEFAULT_PREFIX_LENGTH)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)


def add_chunking_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--minimum-chunk-size", type=int, default=DEFAULT_MINIMUM_CHUNK_SIZE)


def add_calibration_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--strategy",
        choices=[CalibrationStrategy.ACCURACY.value, CalibrationStrategy.BUDGET.value],
        default=CalibrationStrategy.ACCURACY.value,
    )
    parser.add_argument("--target-retrieval-rate", type=float, default=0.30)


def add_parallel_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run Stage 2.5 in parallel.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes.",
    )

    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=None,
        metavar="GPU_ID",
        help="CUDA device IDs to use.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training-Free Adaptive Retrieval Gating application.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    add_common_path_arguments(prepare_parser)
    add_embedding_arguments(prepare_parser)
    add_chunking_arguments(prepare_parser)
    prepare_parser.set_defaults(handler=command_prepare)

    calibrate_parser = subparsers.add_parser("calibrate")
    add_common_path_arguments(calibrate_parser)
    add_embedding_arguments(calibrate_parser)
    add_generation_arguments(calibrate_parser)
    add_calibration_arguments(calibrate_parser)
    add_parallel_arguments(calibrate_parser)
    calibrate_parser.set_defaults(handler=command_calibrate)
    calibrate_parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=DEFAULT_RETRIEVAL_TOP_K,
    )

    ask_parser = subparsers.add_parser("ask")
    add_common_path_arguments(ask_parser)
    add_embedding_arguments(ask_parser)
    add_generation_arguments(ask_parser)
    ask_parser.add_argument("--question", "-q")
    ask_parser.set_defaults(handler=command_ask)

    benchmark_parser = subparsers.add_parser("benchmark")
    add_common_path_arguments(benchmark_parser)
    add_embedding_arguments(benchmark_parser)
    add_generation_arguments(benchmark_parser)
    add_parallel_arguments(benchmark_parser)
    benchmark_parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=DEFAULT_RETRIEVAL_TOP_K,
    )
    benchmark_parser.add_argument(
        "--output-jsonl",
        default="benchmark_results.jsonl",
        help="Raw JSON Lines output. Use --append across model runs.",
    )
    benchmark_parser.add_argument(
        "--output-csv",
        default="benchmark_results.csv",
        help="Flat CSV output for Jupyter/pandas.",
    )
    benchmark_parser.add_argument(
        "--append",
        action="store_true",
        help="Append this model run to existing datasets.",
    )
    benchmark_parser.set_defaults(handler=command_benchmark)

    run_parser = subparsers.add_parser("run")
    add_common_path_arguments(run_parser)
    add_embedding_arguments(run_parser)
    add_generation_arguments(run_parser)
    add_chunking_arguments(run_parser)
    add_calibration_arguments(run_parser)
    add_parallel_arguments(run_parser)
    run_parser.add_argument("--retrieval-top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K)
    run_parser.add_argument("--question", "-q", action="append", required=True)
    run_parser.set_defaults(handler=command_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()

    if argv is None:
        argv = sys.argv[1:]
        
        if argv and argv[0].endswith(".json"):
            argv = []

    parser = build_parser()
    args = parser.parse_args(argv)
    
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("\nOperation cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nError: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
