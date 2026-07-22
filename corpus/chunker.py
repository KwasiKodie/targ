"""
chunker.py

Deterministic document chunking for Retrieval Verification.

This module is responsible for converting extracted corpus documents
into immutable chunks suitable for indexing by the vector store. 

Responsibilities
----------------
* Validate chunking configuration.
* Split text deterministically.
* Generate immutable Chunk objects.
* Preserve source metadata.

The Chunker NEVER:

* Creates embeddings.
* Builds a vector database.
* Performs retrieval.
* Reranks documents.

Author:
Retrieval as a Decision (TARG) Reproduction
"""

from __future__ import annotations 

from collections.abc import Iterable 
from typing import Any

from corpus.corpus_models import Chunk, CorpusDocument 
from corpus.chunk_metadata import ChunkMetadata

class Chunker:
    """
    Deterministic character-based document chunker.

    Default configuration reproduces the preprocessing described in
    the TARG paper. 

    Parameters
    ----------
    chunk_size:
        Maximum characters per chunk.

    chunk_overlap:
        Character overlap between adjacent chunks.

    minimum_chunk_size: 
        Minimum characters required for the final chunk.
    """

    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 100
    DEFAULT_MINIMUM_CHUNK_SIZE = 200

    def __init__(
        self,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        minimum_chunk_size: int = DEFAULT_MINIMUM_CHUNK_SIZE,
    ) -> None: 
        
        self.chunk_size = chunk_size 
        self.chunk_overlap = chunk_overlap 
        self.minimum_chunk_size = minimum_chunk_size 

        self._validate_constructor()

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def chunk_document(
        self,
        document: CorpusDocument,
    ) -> tuple[Chunk, ...]:
        """
        Chunk a single document. 

        Parameters
        ----------
        document:
            Corpus document.

        Returns
        -------
        tuple[Chunk, ...]
        """

        self._validate_document(document)

        pieces = self._split_text(document.text)

        chunks: list[Chunk] = []

        for index, text in enumerate(pieces):

            chunks.append(

                self._build_chunk(
                    document=document,
                    chunk_index=index,
                    text=text,
                )
            )

        return tuple(chunks)
    
    def chunk_documents(
        self,
        documents: Iterable[CorpusDocument],
    ) -> tuple[Chunk, ...]:
        """
        Chunk multiple corpus documents.
        """

        if documents is None:
            raise TypeError(
                "documents must not be None."
            )
        
        documents = tuple(documents)

        if len(documents) == 0:
            raise ValueError(
                "At least one document is required."
            )
        
        chunks: list[Chunk] = []

        for document in documents:

            chunks.extend(
                self.chunk_document(document)
            )

        return tuple(chunks)
    
    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _split_text(
        self,
        text: str,
    ) -> tuple[str, ...]:
        """
        Split text using a deterministic sliding window.
        """

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string."
            )
        
        text = text.strip()

        if not text:
            return ()
        
        chunks: list[str] = []

        step = (
            self.chunk_size 
            - self.chunk_overlap 
        )

        start = 0 

        while start < len(text):

            end = start + self.chunk_size 

            piece = text[start:end]

            if (
                len(piece)
                < self.minimum_chunk_size
            ):
                break 

            chunks.append(piece)

            start += step 

        return tuple(chunks)
    
    def _build_chunk(
        self,
        *,
        document: CorpusDocument, 
        chunk_index: int, 
        text: str,
    ) -> Chunk: 
        """
        Build an immutable Chunk.
        """

        chunk_id = (
            f"{document.source}"
            f"_p{document.page:03d}"
            f"_c{chunk_index:03d}"
        )

        metadata = ChunkMetadata.build(
            source=document.source,
            page=document.page,
            chunk_index=chunk_index,
        )

        return Chunk(
            id=metadata["id"],
            source=document.source,
            page=document.page,
            chunk_index=chunk_index,
            text=text,
            metadata=metadata,
        )
    
    # ----------------------------------------------
    # Validation
    # ----------------------------------------------

    def _validate_constructor(self) -> None:

        if not isinstance(
            self.chunk_size,
            int,
        ):
            raise TypeError(
                "chunk_size must be an integer."
            )
        
        if self.chunk_size <= 0:
            raise ValueError(
                "chunk_size must be positive."
            )
        
        if not isinstance(
            self.chunk_overlap,
            int,
        ): 
            raise TypeError(
                "chunk_overlap must be an integer."
            )
        
        if self.chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )
        
        if (
            self.chunk_overlap
            >= self.chunk_size 
        ):
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )
        
        if not isinstance(
            self.minimum_chunk_size,
            int,
        ):
            raise TypeError(
                "minimum_chunk_size must be an integer."
            )
        
        if self.minimum_chunk_size <= 0:
            raise ValueError(
                "minimum_chunk_size must be positive."
            )
        

    def _validate_document(
        self,
        document: CorpusDocument,
    ) -> None:
        
        if document is None:
            raise TypeError(
                "document must not be None."
            )
        
        if not isinstance(
            document,
            CorpusDocument,
        ): 
            raise TypeError(
                "document must be a CorpusDocument."
            )
        
        if not document.source.strip():
            raise ValueError(
                "Document source cannot be empty."
            )
        
        if document.page <= 0:
            raise ValueError(
                "Document page must be positive."
            )
        
        if document.page <= 0:
            raise ValueError(
                "Document page must be positive."
            )
        
        if not document.text.strip():
            raise ValueError(
                "Document text cannot be empty."
            )