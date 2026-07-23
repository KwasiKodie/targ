import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from benchmark.benchmark_loader import BenchmarkLoader
from retrieval.retrieval_inspector import RetrievalInspector
from retrieval.retrieval_runner import RetrievalRunner


def main() -> None:
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not defined in the environment or .env file."
        )

    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

    if langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
        os.environ["LANGCHAIN_ENDPOINT"] = (
            "https://api.smith.langchain.com"
        )

    # -------------------------------------------------
    # Benchmark loader
    # -------------------------------------------------

    benchmark_loader = BenchmarkLoader(
        benchmark_path="benchmark/benchmark_queries.json",
    )

    # -------------------------------------------------
    # Embedding model
    # -------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={"device":"cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


    # -------------------------------------------------
    # Load persisted FAISS vector store
    #
    # This directory must contain:
    #   index.faiss
    #   index.pkl
    # -------------------------------------------------

    vector_store = FAISS.load_local(
        folder_path="rag/vector_stores/authoritative_faiss",
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    # -------------------------------------------------
    # Retrieval inspector
    # -------------------------------------------------

    retrieval_inspector = RetrievalInspector(
        vector_store=vector_store,
        top_k=5,
    )

    # -------------------------------------------------
    # Retrieval runner
    # -------------------------------------------------

    runner = RetrievalRunner(
        benchmark_loader=benchmark_loader,
        retrieval_inspector=retrieval_inspector,
    )

    # -------------------------------------------------
    # Run and save
    # -------------------------------------------------

    inspections = runner.run_and_save(
        output_path="retrieval/retrieval_results.json",
    )

    print(
        f"Completed {len(inspections)} benchmark queries."
    )


if __name__ == "__main__":
    main()