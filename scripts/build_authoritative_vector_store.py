"""
build_athoritative_vector_store.py

Build the persistent FAISS index for the authoritative TARG corpus.
"""

from __future__ import annotations

from pathlib import Path 

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 

from corpus.chunker import Chunker 
from corpus.representative_corpus import RepresentativeCorpusBuilder
from corpus.vector_store_builder import VectorStoreBuilder

# -------------------------------------------------------
# Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_DIRECTORY = PROJECT_ROOT / "rag" / "knowledge"

VECTOR_STORE_DIRECTORY = (
    PROJECT_ROOT
    / "rag"
    / "vector_stores"
    / "authoritative_faiss"
)

# --------------------------------------------------------
# Configuration
# --------------------------------------------------------

EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 100

MINIMUM_CHUNK_SIZE = 200

# --------------------------------------------------------
# Build pipeline
# --------------------------------------------------------

def build_authoritative_vector_store() -> None: 
    """
    Load, chunk, embed, and persist the authoritative PDF corpus.
    """

    corpus_builder = RepresentativeCorpusBuilder(
        pdf_directory=PDF_DIRECTORY,
    )

    corpus_documents = corpus_builder.build()

    if not corpus_documents:
        raise RuntimeError(
            "No corpus documents were produced from the PDF directory."
        )
    
    chunker = Chunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        minimum_chunk_size=MINIMUM_CHUNK_SIZE,
    )

    chunks = chunker.chunk_documents(
        corpus_documents,
    )

    if not chunks:
        raise RuntimeError(
            "No chunks were produced from the authoritative corpus."
        )
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        # encode_kwargs={
        #     "normalized_embeddings": True,
        # },
    )

    vector_store_builder = VectorStoreBuilder(
        embedding_model=embedding_model,
        vector_store_class=FAISS,
    )

    vector_store = vector_store_builder.build(
        chunks,
    )

    VECTOR_STORE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save_local(
        str(VECTOR_STORE_DIRECTORY),
    )

    print(
        f"Corpus documents: {len(corpus_documents)}"
    )

    print(
        f"Chunks indexed: {len(chunks)}"
    )

    print(
        f"Vector store saved to: {VECTOR_STORE_DIRECTORY}"
    )

if __name__ == "__main__":
    build_authoritative_vector_store()