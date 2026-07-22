"""
representative_corpus.py

Loads the authoritative PDF corpus into immutable CorpusDocument
objects.

Responsibilities
----------------
* Validate the corpus directory.
* Discover PDF files. 
* Extract page text.
* Construct CorpusDocument objects.
* Return a deterministic ordering.

Never: 

* chunks documents
* embeds documents 
* creates vector stores 
* performs retrieval

Author: 
Retrieval as a Decision (TARG) Reproduction
"""

from __future__ import annotations 

from pathlib import Path
from typing import Sequence 

from pypdf import PdfReader 

from corpus.corpus_models import CorpusDocument 

class RepresentativeCorpusBuilder:

    def __init__(
        self,
        *,
        pdf_directory: Path,
    ) -> None: 
        
        self.pdf_directory = pdf_directory 

        self._validate_directory()

    # -------------------------------------------
    # Public API
    # -------------------------------------------

    def build(
        self,
    ) -> tuple[CorpusDocument, ...]:
        
        documents: list[CorpusDocument] = []

        for pdf_path in self._discover_pdfs():

            documents.extend(
                self._load_pdf(pdf_path)
            )

        return self._sort_documents(
            documents
        )
    
    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    def _validate_directory(
        self,
    ) -> None:
        
        if not isinstance(
            self.pdf_directory,
            Path,
        ): 
            raise TypeError(
                "pdf_directory must be a pathlib.Path."
            )
        
        if not self.pdf_directory.exists():
            raise FileNotFoundError(
                f"{self.pdf_directory} does not exist."
            )
        
        if not self.pdf_directory.is_dir():
            raise NotADirectoryError(
                f"{self.pdf_directory} is not a directory."
            )
        
    # ----------------------------------------------------------------
    # Discovery
    # ----------------------------------------------------------------

    def _discover_pdfs(
        self,
    ) -> tuple[Path, ...]:
        
        return tuple(
            sorted(
                self.pdf_directory.glob("*.pdf")
            )
        )
    
    # ----------------------------------------------------------------
    # Loading
    # ----------------------------------------------------------------

    def _load_pdf(
        self,
        pdf_path: Path, 
    ) -> tuple[CorpusDocument, ...]:
        
        reader = PdfReader(pdf_path)

        documents = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            
            text = page.extract_text()

            if text is None:
                continue 

            text = text.strip()

            if not text:
                continue 

            documents.append(
                self._build_document(
                    source=pdf_path.stem,
                    page=page_number,
                    text=text,
                )
            )

        return tuple(documents)
    
    # ----------------------------------------------------
    # Model construction
    # ----------------------------------------------------

    def _build_document(
        self,
        *,
        source: str,
        page: int,
        text: str,
    ) -> CorpusDocument:
        
        return CorpusDocument(
            source=source,
            page=page,
            text=text,
            metadata={
                "source": source,
                "page": page,
            },
        )
    
    # ------------------------------------------------
    # Ordering 
    # ------------------------------------------------

    def _sort_documents(
        self,
        documents: Sequence[CorpusDocument],
    ) -> tuple[CorpusDocument, ...]:
        
        return tuple(
            sorted(
                documents,
                key=lambda d: (
                    d.source,
                    d.page,
                ),
            )
        )