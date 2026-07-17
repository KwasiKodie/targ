from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedDocument:

    id: str

    text: str

    score: float

    metadata: dict


@dataclass(frozen=True)
class RetrievalResult:

    query: str

    documents: list[RetrievedDocument]

    retrieved: bool

    retrieval_backend: str


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        pass

class VectorRetriever(BaseRetriever):

    def __init__(

        self,

        vector_store,

    ):

        self.vector_store = vector_store

    def retrieve(

        self,

        query: str,

        top_k: int = 5,

    ) -> RetrievalResult:

        docs = self.vector_store.similarity_search(
            query,
            k=top_k,
        )

        return RetrievalResult(

            query=query,

            retrieved=True,

            retrieval_backend="VectorStore",

            documents=[
                RetrievedDocument(
                    id=d.metadata["id"],
                    text=d.page_content,
                    score=d.metadata.get("score", 0.0),
                    metadata=d.metadata,
                )
                for d in docs
            ],
        )
