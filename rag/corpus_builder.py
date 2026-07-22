"""
corpus_builder.py

Deterministic document loading and chunking for the representative
TARG corpus. 

Responsibilities
----------------
1. Load every PDF from a fixed corpus directory.
2. Extract text page by page.
3. Normalize extracted text consistently.
4. Split documents into approximately 1,000-character passages.
5. Apply a 100-character overlap.
6. Discard passages shorter than 200 characters.
7. Attach stable provenance metadata to every chunk.
8. Produce an immutable corpus-build result.

The builder does NOT 

- create embeddings
- construct a vector index 
- perform retrieval 
- evaluate retrieval quality 
- execute the TARG gate

Author:
TARG Reproduction
"""

from __future__ import annotations

import hashlib 
import re 
from dataclasses import dataclass 
from pathlib import Path 
from types import MappingProxyType 
from typing import Iterable, Mapping, Sequence 

from langchain_core.documents import Document 
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 

# --------------------------------------------------------------------
# Corpus document specification 
# --------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CorpusDocumentSpec:
    """
    Metadata describing one source document in the corpus.

    Attributes
    ----------
    corpus_id:
        Stable machine-readable identifier. 

    filename: 
        Exact PDF filename within the corpus directory. 

    title: 
        Canonical document title. 

    category: 
        Broad document category, such as ``rag_research`` or 
        ``security_standard``.
    """

    corpus_id: str 
    filename: str 
    title: str 
    category: str 

    def __post_init__(self) -> None:

        object.__setattr__(
            self,
            "corpus_id",
            self._validate_text(
                self.corpus_id, 
                "corpus_id",
            ),
        )

        object.__setattr__(
            self,
            "filename",
            self._validate_filename(
                self.filename,
            ),
        )

        object.__setattr__(
            self,
            "title",
            self._validate_text(
                self.title, 
                "title",
            ),
        )

        object.__setattr__(
            self,
            "category",
            self._validate_text(
                self.category,
                "category",
            ),
        )

    @staticmethod 
    def _validate_text(
        value: str, 
        field_name: str,
    ) -> str: 
        
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be a string."
            )
        
        value = value.strip()

        if not value: 
            raise ValueError(
                f"{field_name} must not be empty."
            )
        
        return value 
    
    @classmethod 
    def _validate_filename(
        cls, 
        value: str, 
    ) -> str: 
        
        value = cls._validate_text(
            value,
            "filename",
        )

        path = Path(value)

        if path.name != value: 
            raise ValueError(
                "filename must contain only the file name, "
                "not a directory path."
            )
        
        if path.suffix.lower() != ".pdf":
            raise ValueError(
                "filename must identify a PDF file."
            )
        
        return value 
    

# -------------------------------------------------------
# Chunking configuration
# -------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CorpusChunkingConfig: 

    """
    Fixed chunking configuration for the representative corpus.
    """

    chunk_size: int = 1_000

    chunk_overlap: int = 100 

    minimum_chunk_size: int = 200

    separators: tuple[str, ...] = (
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    )

    def __post_init__(self) -> None: 

        if not isinstance(self.chunk_size, int):
            raise TypeError(
                "chunk_size must be an integer."
            )
        
        if self.chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )
        
        if not isinstance(self.chunk_overlap, int):
            raise TypeError(
                "chunk_overlap must be an integer."
            )
        
        if self.chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap must not be negative."
            )
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )
        
        if self.chunk_overlap >= self.chunk_size:
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
                "minimum_chunk_size must be greater than zero."
            )
        
        if self.minimum_chunk_size > self.chunk_size: 
            raise ValueError(
                "minimum_chunk_size must not exceed chunk_size."
            )
        
        if not self.separators:
            raise ValueError(
                "separators must not be empty."
            )
        

# ------------------------------------------------------------------
# Immutable output records
# ------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CorpusSourceSummary:
    """
    Build statistics for one source document.
    """

    corpus_id: str 

    source: str 

    title: str 

    page_count: int 

    extracted_character_count: int 

    chunk_count: int 

@dataclass(frozen=True, slots=True)
class CorpusBuildResult:
    """
    Complete result of a deterministic corpus build.
    """

    documents: tuple[Document, ...]

    source_summaries: tuple[CorpusSourceSummary, ...]

    total_source_count: int 

    total_page_count: int 

    total_chunk_count: int 

    total_character_count: int 

    corpus_fingerprint: str 

    chunking_config: CorpusChunkingConfig 

    def __post_init__(self) -> None: 

        if self.total_source_count != len(
            self.source_summaries 
        ):
            raise ValueError(
                "total_source_count does not match "
                "source_summaries."
            )
        
        if self.total_chunk_count != len(
            self.documents
        ):
            raise ValueError(
                "total_chunk_count does not match documents."
            )
        
        if not self.corpus_fingerprint:
            raise ValueError(
                "corpus_fingerprint must not be empty."
            )
        
# --------------------------------------------------------------
# Representative corpus manifest 
# --------------------------------------------------------------

REPRESENTATIVE_CORPUS: tuple[CorpusDocumentSpec, ...] = (

    CorpusDocumentSpec(
        corpus_id="self_rag",
        filename="self_rag.pdf",
        title=(
            "SELF-RAG: LEARNING TO RETRIEVE, GENERATE, AND "
            "CRITIQUE THROUGH SELF-REFLECTION"
        ),
        category="rag_research",
    ),

    CorpusDocumentSpec(
        corpus_id="adaptive_rag",
        filename="adaptive_rag.pdf",
        title=(
            "Adaptive-RAG: Learning to Adapt "
            "Retrieval-Augmented Large Language Models "
            "Through Question Complexity"
        ), 
        category="rag_research",
    ),

    CorpusDocumentSpec(
        corpus_id="rag_survey",
        filename="rag_survey.pdf",
        title=(
            "A Comprehensive Survey of Retrieval-Augmented "
            "Generation (RAG): Evolution, Current Landscape "
            "and Future Directions"
        ),
        category="rag_survey",
    ),

    CorpusDocumentSpec(
        corpus_id="targ",
        filename="targ.pdf",
        title=(
            "Retrieval as a Decision: Training-Free "
            "Adaptive Gating for Efficient RAG"
        ),
        category="rag_research",
    ),

    CorpusDocumentSpec(
        corpus_id="coala",
        filename="coala.pdf",
        title="Cognitive Architectures for Language Agents",
        category="agent_memory",
    ),

    CorpusDocumentSpec(
        corpus_id="nist_sp_800_115",
        filename="nist_sp_800_115.pdf",
        title=(
            "Technical Guide to Information Security "
            "Testing and Assessment"
        ),
        category="security_standard",
    ),

    CorpusDocumentSpec(
        corpus_id="ptes",
        filename="ptes.pdf",
        title=(
            "The Penetration Testing Execution "
            "Standard Documentation"
        ),
        category="security_methodology",
    ),
)

# ------------------------------------------------------------
# Corpus builder 
# ------------------------------------------------------------

class RepresentativeCorpusBuilder:
    """
    Load and consistently chunk the representative corpus.
    """

    def __init__(
        self,
        *,
        corpus_directory: str | Path,
        specifications: Sequence[
            CorpusDocumentSpec 
        ] = REPRESENTATIVE_CORPUS,
        config: CorpusChunkingConfig | None = None, 
    ) -> None: 
        
        self._corpus_directory = self._validate_directory(
            corpus_directory
        )

        self._specifications = self._validate_specifications(
            specifications
        )

        self._config = (
            config 
            if config is not None 
            else CorpusChunkingConfig()
        )

        if not isinstance(
            self._config, 
            CorpusChunkingConfig,
        ):
            raise TypeError(
                "config must be a CorpusChunkingConfig."
            )
        
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.chunk_size, 
            chunk_overlap=self._config.chunk_overlap,
            separators=list(
                self._config.separators
            ),
            length_function=len,
            is_separator_regex=False,
        )

    # --------------------------------------------------------------

    @property 
    def corpus_directory(self) -> Path:
        return self._corpus_directory 
    
    @property 
    def specifications(
        self, 
    ) -> tuple[CorpusDocumentSpec, ...]:
        return self._specifications 
    
    @property 
    def config(self) -> CorpusChunkingConfig:
        return self._config 
    
    # -------------------------------------------------------------

    def build(self) -> CorpusBuildResult:
        """
        Load, normalize, chunk, validate and summarize the corpus.
        """

        self._validate_source_files()

        all_chunks: list[Document] = []

        source_summaries: list[
            CorpusSourceSummary
        ] = []

        for specification in self._specifications:

            source_chunks, summary = (
                self._process_source(
                    specification 
                )
            )

            all_chunks.extend(
                source_chunks
            )

            source_summaries.append(
                summary
            )

        self._validate_chunk_ids(
            all_chunks
        )

        fingerprint = self._create_corpus_fingerprint(
            all_chunks
        )

        return CorpusBuildResult(
            documents=tuple(all_chunks),
            source_summaries=tuple(
                source_summaries
            ),
            total_source_count=len(
                source_summaries
            ),
            total_page_count=sum(
                item.page_count
                for item in source_summaries 
            ),
            total_chunk_count=len(
                all_chunks
            ),
            total_character_count=sum(
                len(document.page_content)
                for document in all_chunks 
            ),
            corpus_fingerprint=fingerprint,
            chunking_config=self._config, 
        )
    
    # ------------------------------------------------------

    def _process_source(
        self,
        specification: CorpusDocumentSpec,
    ) -> tuple[
        tuple[Document, ...],
        CorpusSourceSummary,
    ]:
        """
        Load and chunk one PDF source.
        """

        source_path = (
            self._corpus_directory 
            / specification.filename 
        )

        loader = PyPDFLoader(
            str(source_path)
        )

        pages = loader.load()

        if not pages: 
            raise ValueError(
                f"No pages were extracted from "
                f"{specification.filename!r}."
            )
        
        source_chunks: list[Document] = []

        extracted_character_count = 0

        source_chunk_index = 0 

        for fallback_page_index, page in enumerate(
            pages
        ):
            page_text = self._normalize_text(
                page.page_content
            )

            extracted_character_count += len(
                page_text 
            )

            if not page_text: 
                continue 

            page_number = self._resolve_page_number(
                page.metadata,
                fallback_page_index,
            )

            split_texts = (
                self._splitter.split_text(
                    page_text 
                )
            )

            for page_chunk_index, chunk_text in enumerate(
                split_texts
            ):
                
                chunk_text = self._normalize_text(
                    chunk_text 
                )

                if (
                    len(chunk_text)
                    < self._config.minimum_chunk_size
                ): 
                    continue 

                chunk_id = self._create_chunk_id(
                    corpus_id=specification.corpus_id,
                    page_number=page_number,
                    chunk_index=source_chunk_index,
                    text=chunk_text,
                )

                metadata = MappingProxyType(
                    {
                        "id": chunk_id,
                        "corpus_id": (
                            specification.corpus_id
                        ),
                        "source": (
                            specification.filename
                        ),
                        "source_path": str(
                            source_path.resolve()
                        ),
                        "title": specification.title,
                        "category": (
                            specification.category
                        ),
                        "page": page_number,
                        "page_chunk_index": (
                            page_chunk_index
                        ),
                        "chunk": source_chunk_index,
                        "character_count": len(
                            chunk_text
                        ),
                        "chunk_size": (
                            self._config.chunk_size
                        ),
                        "chunk_overlap": (
                            self._config.chunk_overlap
                        ),
                        "minimum_chunk_size": (
                            self._config
                            .minimum_chunk_size
                        ),
                    }
                )

                source_chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=dict(metadata),
                    )
                )

                source_chunk_index += 1

        if not source_chunks:
            raise ValueError(
                f"No valid chunks were produced from "
                f"{specification.filename!r}."
            )
        
        summary = CorpusSourceSummary(
            corpus_id=specification.corpus_id,
            source=specification.filename,
            title=specification.title,
            page_count=len(pages),
            extracted_character_count=(
                extracted_character_count
            ),
            chunk_count=len(source_chunks),
        )

        return (
            tuple(source_chunks),
            summary,
        )
    
    # -----------------------------------------------------

    def _validate_source_files(self) -> None:
        """
        Ensure every manifest entry resolves to a readable PDF.
        """

        missing: list[str] = []

        for specification in self._specifications:

            path = (
                self._corpus_directory 
                / specification.filename
            )

            if not path.is_file():
                missing.append(
                    specification.filename
                )

        if missing:
            formatted = ", ".join(
                sorted(missing)
            )

            raise FileNotFoundError(
                "The following corpus files are missing: "
                f"{formatted}"
            )
        
    # ------------------------------------------------------------

    @staticmethod
    def _normalize_text(
        text: str, 
    ) -> str: 
        """
        Normalize PDF-extracted text without changing its meaning.

        Normalization
        -------------
        1. Remove soft hyphens.
        2. Repair words split across line breaks. 
        3. Normalize line endings.
        4. Collapse repeated horizontal whitespace.
        5. Preserve paragraph boundaries.
        """

        if not isinstance(text, str):
            raise TypeError(
                "Extracted page text must be a string."
            )
        
        text = text.replace(
            "\u00ad",
            "",
        )

        text = text.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        text = re.sub(
            r"(?<=\w)-\n(?=\w)",
            "",
            text,
        )

        paragraphs = re.split(
            r"\n\s*\n",
            text,
        )

        normalized_paragraphs: list[str] = []

        for paragraph in paragraphs:

            paragraph = re.sub(
                r"[ \t]+",
                " ",
                paragraph,
            )

            paragraph = re.sub(
                r"\s*\n\s*",
                " ",
                paragraph,
            )

            paragraph = paragraph.strip()

            if paragraph:
                normalized_paragraphs.append(
                    paragraph
                )

        return "\n\n".join(
            normalized_paragraphs
        )
    
    # ---------------------------------------------------------

    @staticmethod
    def _resolve_page_number(
        metadata: Mapping[str, object],
        fallback_page_index: int,
    ) -> int:
        """
        Return a stable one-based page number.
        """

        raw_page = metadata.get(
            "page",
            fallback_page_index,
        )

        if isinstance(raw_page, bool):
            raw_page = fallback_page_index 

        try:
            page_index = int(raw_page)
        except (
            TypeError, 
            ValueError,
        ):
            page_index = fallback_page_index 

        return page_index + 1
    
    # ----------------------------------------------------

    @staticmethod 
    def _create_chunk_id(
        *,
        corpus_id: str,
        page_number: int,
        chunk_index: int, 
        text: str,
    ) -> str: 
        """
        Produce a stable, content-sensitive chunk identifier.
        """

        digest_input = (
            f"{corpus_id}|"
            f"{page_number}|"
            f"{chunk_index}|"
            f"{text}"
        )

        digest = hashlib.sha256(
            digest_input.encode(
                "utf-8"
            )
        ).hexdigest()[:16]

        return (
            f"{corpus_id}"
            f"_p{page_number:04d}"
            f"_c{chunk_index:05d}"
            f"_{digest}"
        )
    
    # -------------------------------------------------------------

    @staticmethod 
    def _create_corpus_fingerprint(
        documents: Iterable[Document],
    ) -> str: 
        """
        Hash all ordered chunk IDs and text. 

        The fingerprint changes when source content, chunking order, 
        or chunk boundaries change.
        """

        hasher = hashlib.sha256()

        for document in documents:

            chunk_id = document.metadata.get(
                "id"
            )

            hasher.update(
                str(chunk_id).encode(
                    "utf-8"
                )
            )

            hasher.update(
                b"\0"
            )

            hasher.update(
                document.page_content.encode(
                    "utf-8"
                )
            )

            hasher.update(
                b"\0"
            )
        
        return hasher.hexdigest()
    
    # ---------------------------------------------------

    @staticmethod 
    def _validate_chunk_ids(
        documents: Sequence[Document],
    ) -> None: 
        """
        Ensure every generated chunk ID is present and unique.
        """

        seen: set[str] = set()

        for document in documents:

            chunk_id = document.metadata.get(
                "id"
            )

            if (
                not isinstance(chunk_id, str)
                or not chunk_id
            ):
                raise ValueError(
                    "Every chunk must have a non-empty "
                    "metadata['id'] value."
                )
            
            if chunk_id in seen:
                raise ValueError(
                    f"Duplicate chunk ID: {chunk_id}"
                )
            
            seen.add(
                chunk_id
            )

    # --------------------------------------------------------

    @staticmethod 
    def _validate_directory(
        value: str | Path,
    ) -> Path:
        
        if not isinstance(
            value,
            (str, Path),
        ):
            raise TypeError(
                "corpus_directory must be a string or Path."
            )
        
        path = Path(value).expanduser()

        if not path.exists():
            raise FileNotFoundError(
                f"Corpus directory does not exist: {path}"
            )
        
        if not path.is_dir():
            raise NotADirectoryError(
                f"Corpus path is not a directory: {path}"
            )
        
        return path.resolve()
    
    # ---------------------------------------------------------------

    @staticmethod
    def _validate_specifications(
        specifications: Sequence[
            CorpusDocumentSpec
        ],
    ) -> tuple[CorpusDocumentSpec, ...]:
        
        if specifications is None:
            raise TypeError(
                "specifications must not be None."
            )
        
        specifications = tuple(
            specifications
        )

        if not specifications:
            raise ValueError(
                "At least one corpus specification is required."
            )
        
        corpus_ids: set[str] = set()

        filenames: set[str] = set()

        for specification in specifications:

            if not isinstance(
                specification,
                CorpusDocumentSpec,
            ):
                raise TypeError(
                    "Each specification must be a "
                    "CorpusDocumentSpec."
                )
            
            if specification.corpus_id in corpus_ids:
                raise ValueError(
                    "Duplicate corpus_id: "
                    f"{specification.corpus_id}"
                )
            
            if specification.filename in filenames:
                raise ValueError(
                    "Duplicate filename: "
                    f"{specification.filename}"
                )
            
            corpus_ids.add(
                specification.corpus_id
            )

            filenames.add(
                specification.filename 
            )

        return specifications 