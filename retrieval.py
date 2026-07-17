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

    @staticmethod
    def _validate_request(
        query: str,
        top_k: int,
    ) -> tuple[str, int]:

        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("query must not be empty.")

        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        return query, top_k

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        ...

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

        query, top_k = self._validate_request(query, top_k)

        docs = self.vector_store.similarity_search(
            query,
            k=top_k,
        )

        for d in docs:
            print(d.metadata)

        return RetrievalResult(

            query=query,

            retrieved=True,

            retrieval_backend="VectorStore",

            documents=[
                RetrievedDocument(
                    id=d.metadata.get(
                        "id",
                        f"doc_{i}",
                    ),
                    text=d.page_content,
                    score=d.metadata.get("score", 0.0),
                    metadata=d.metadata,
                )
                for i, d in enumerate(docs)
            ],
        )
