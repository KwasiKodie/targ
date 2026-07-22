from langchain_core.embeddings import Embeddings 
from langchain_core.vectorstores import VectorStore 
from tests.test_representative_corpus import create_pdf 
from corpus.representative_corpus import RepresentativeCorpusBuilder
from corpus.chunker import Chunker 
from corpus.vector_store_builder import VectorStoreBuilder 


class FakeEmbeddings(Embeddings):

    def embed_documents(self, texts):

        return [
            [float(i)] * 8
            for i, _ in enumerate(texts)
        ]
    
    def embed_query(self, text):

        return [0.0] * 8
    
class FakeVectorStore(VectorStore):

    def __init__(self, documents):
        self.documents = list(documents)

    @classmethod
    def from_documents(
        cls,
        documents,
        embedding,
        **kwargs,
    ):
        return cls(documents)

    @classmethod
    def from_texts(
        cls,
        texts,
        embedding,
        metadatas=None,
        **kwargs,
    ):
        return cls([])

    def add_texts(
        self,
        texts,
        metadatas=None,
        **kwargs,
    ):
        return []

    def similarity_search(
        self,
        query,
        k=4,
        **kwargs,
    ):
        return []
    
def test_complete_pipeline(
    tmp_path,
):

    create_pdf(
        tmp_path / "paper.pdf",
        [
            make_page_text("First page"),
            make_page_text("Second page"),
        ],
    )

    corpus = RepresentativeCorpusBuilder(
        pdf_directory=tmp_path,
    ).build()

    chunker = Chunker(
        chunk_size=1000,
        chunk_overlap=100,
        minimum_chunk_size=200,
    )

    chunks = chunker.chunk_documents(
        corpus,
    )

    assert len(corpus) == 2
    assert len(chunks) == 2

    builder = VectorStoreBuilder(
        embedding_model=FakeEmbeddings(),
        vector_store_class=FakeVectorStore,
    )

    store = builder.build(
        chunks,
    )

    assert isinstance(
        store,
        FakeVectorStore,
    )

    assert len(store.documents) == 2


def test_pipeline_preserves_metadata(
    tmp_path,
):

    create_pdf(
        tmp_path / "paper.pdf",
        [
            make_page_text("Metadata page"),
        ],
    )

    corpus = RepresentativeCorpusBuilder(
        pdf_directory=tmp_path,
    ).build()

    chunks = Chunker(
        chunk_size=1000,
        chunk_overlap=100,
        minimum_chunk_size=200,
    ).chunk_documents(
        corpus,
    )

    assert len(chunks) == 1

    store = VectorStoreBuilder(
        embedding_model=FakeEmbeddings(),
        vector_store_class=FakeVectorStore,
    ).build(
        chunks,
    )

    document = store.documents[0]

    assert document.metadata["source"] == "paper"
    assert document.metadata["page"] == 1
    assert document.metadata["chunk"] == 0
    


def test_pipeline_ordering(
    tmp_path,
):

    create_pdf(
        tmp_path / "paper.pdf",
        [
            make_page_text("Page one"),
            make_page_text("Page two"),
            make_page_text("Page three"),
        ],
    )

    corpus = RepresentativeCorpusBuilder(
        pdf_directory=tmp_path,
    ).build()

    chunks = Chunker(
        chunk_size=1000,
        chunk_overlap=100,
        minimum_chunk_size=200,
    ).chunk_documents(
        corpus,
    )

    assert len(chunks) == 3

    store = VectorStoreBuilder(
        embedding_model=FakeEmbeddings(),
        vector_store_class=FakeVectorStore,
    ).build(
        chunks,
    )

    pages = tuple(
        document.metadata["page"]
        for document in store.documents
    )

    assert pages == (
        1,
        2,
        3,
    )

def test_pipeline_empty_directory(
    tmp_path,
):
    
    corpus = RepresentativeCorpusBuilder(
        pdf_directory=tmp_path,
    ).build()

    assert corpus == ()

def test_pipeline_multiple_pdfs(
    tmp_path,
):

    create_pdf(
        tmp_path / "a.pdf",
        [
            make_page_text("Document A"),
        ],
    )

    create_pdf(
        tmp_path / "b.pdf",
        [
            make_page_text("Document B"),
        ],
    )

    corpus = RepresentativeCorpusBuilder(
        pdf_directory=tmp_path,
    ).build()

    assert len(corpus) == 2

    chunks = Chunker(
        chunk_size=1000,
        chunk_overlap=100,
        minimum_chunk_size=200,
    ).chunk_documents(
        corpus,
    )

    assert len(chunks) == 2

    store = VectorStoreBuilder(
        embedding_model=FakeEmbeddings(),
        vector_store_class=FakeVectorStore,
    ).build(
        chunks,
    )

    assert len(store.documents) == len(chunks)

    sources = tuple(
        document.metadata["source"]
        for document in store.documents
    )

    assert sources == (
        "a",
        "b",
    )

def make_page_text(
    label: str,
    *,
    minimum_length: int = 300,
) -> str:
    """
    Produce deterministic page content long enough to survive
    the configured minimum chunk-size threshold.
    """

    sentence = (
        f"{label} contains authoritative evidence for testing the "
        "retrieval-augmented generation corpus ingestion pipeline. "
    )

    repetitions = (
        minimum_length // len(sentence)
    ) + 1

    return (sentence * repetitions)[:minimum_length]