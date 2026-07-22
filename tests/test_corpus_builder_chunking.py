"""
test_corpus_builder_chunking.py

Unit tests for deterministic chunk identifiers, chunk ID validation,
and corpus fingerprint generation.
"""

import hashlib

import pytest
from langchain_core.documents import Document

from rag.corpus_builder import RepresentativeCorpusBuilder


# ---------------------------------------------------------------------
# Chunk ID generation
# ---------------------------------------------------------------------


def test_create_chunk_id_is_deterministic():
    """
    Identical chunk inputs must always produce the same identifier.
    """

    first = RepresentativeCorpusBuilder._create_chunk_id(
        corpus_id="targ",
        page_number=3,
        chunk_index=2,
        text="Adaptive retrieval reduces unnecessary retrieval.",
    )

    second = RepresentativeCorpusBuilder._create_chunk_id(
        corpus_id="targ",
        page_number=3,
        chunk_index=2,
        text="Adaptive retrieval reduces unnecessary retrieval.",
    )

    assert first == second


def test_create_chunk_id_has_expected_structure():
    """
    The identifier should contain the corpus ID, padded page number,
    padded chunk index, and a 16-character SHA-256 prefix.
    """

    chunk_id = RepresentativeCorpusBuilder._create_chunk_id(
        corpus_id="targ",
        page_number=7,
        chunk_index=12,
        text="Example chunk text.",
    )

    assert chunk_id.startswith(
        "targ_p0007_c00012_"
    )

    digest = chunk_id.rsplit(
        "_",
        maxsplit=1,
    )[1]

    assert len(digest) == 16
    assert all(
        character in "0123456789abcdef"
        for character in digest
    )


def test_create_chunk_id_matches_expected_hash():
    """
    The digest should be derived from the exact canonical input format.
    """

    corpus_id = "targ"
    page_number = 2
    chunk_index = 4
    text = "Retrieval is treated as a decision."

    expected_digest_input = (
        f"{corpus_id}|"
        f"{page_number}|"
        f"{chunk_index}|"
        f"{text}"
    )

    expected_digest = hashlib.sha256(
        expected_digest_input.encode("utf-8")
    ).hexdigest()[:16]

    expected_id = (
        "targ"
        "_p0002"
        "_c00004"
        f"_{expected_digest}"
    )

    result = RepresentativeCorpusBuilder._create_chunk_id(
        corpus_id=corpus_id,
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
    )

    assert result == expected_id


@pytest.mark.parametrize(
    (
        "changed_field",
        "arguments",
    ),
    [
        (
            "corpus_id",
            {
                "corpus_id": "adaptive_rag",
                "page_number": 3,
                "chunk_index": 2,
                "text": "Identical text.",
            },
        ),
        (
            "page_number",
            {
                "corpus_id": "targ",
                "page_number": 4,
                "chunk_index": 2,
                "text": "Identical text.",
            },
        ),
        (
            "chunk_index",
            {
                "corpus_id": "targ",
                "page_number": 3,
                "chunk_index": 3,
                "text": "Identical text.",
            },
        ),
        (
            "text",
            {
                "corpus_id": "targ",
                "page_number": 3,
                "chunk_index": 2,
                "text": "Changed text.",
            },
        ),
    ],
)
def test_create_chunk_id_changes_when_any_input_changes(
    changed_field,
    arguments,
):
    """
    Chunk IDs must be sensitive to corpus identity, page, position,
    and content.
    """

    baseline = RepresentativeCorpusBuilder._create_chunk_id(
        corpus_id="targ",
        page_number=3,
        chunk_index=2,
        text="Identical text.",
    )

    changed = RepresentativeCorpusBuilder._create_chunk_id(
        **arguments,
    )

    assert changed != baseline, (
        f"Changing {changed_field} should change the chunk ID."
    )


def test_create_chunk_ids_are_unique_for_distinct_chunks():
    """
    Distinct chunk positions and content should produce unique IDs.
    """

    chunk_ids = {
        RepresentativeCorpusBuilder._create_chunk_id(
            corpus_id="targ",
            page_number=page_number,
            chunk_index=chunk_index,
            text=text,
        )
        for page_number, chunk_index, text in [
            (1, 0, "First chunk."),
            (1, 1, "Second chunk."),
            (2, 0, "Third chunk."),
            (2, 1, "Fourth chunk."),
        ]
    }

    assert len(chunk_ids) == 4


# ---------------------------------------------------------------------
# Chunk ID validation
# ---------------------------------------------------------------------


def test_validate_chunk_ids_accepts_unique_non_empty_ids():
    """
    Validation should accept documents whose IDs are present and unique.
    """

    documents = [
        Document(
            page_content="First chunk.",
            metadata={"id": "chunk-1"},
        ),
        Document(
            page_content="Second chunk.",
            metadata={"id": "chunk-2"},
        ),
    ]

    RepresentativeCorpusBuilder._validate_chunk_ids(
        documents
    )


def test_validate_chunk_ids_accepts_empty_document_sequence():
    """
    An empty collection contains no invalid or duplicate chunk IDs.
    """

    RepresentativeCorpusBuilder._validate_chunk_ids(
        []
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"id": None},
        {"id": ""},
        {"id": 42},
        {"id": False},
    ],
)
def test_validate_chunk_ids_rejects_missing_or_invalid_id(
    metadata,
):
    """
    Every document must contain a non-empty string metadata ID.
    """

    documents = [
        Document(
            page_content="Chunk text.",
            metadata=metadata,
        ),
    ]

    with pytest.raises(
        ValueError,
        match=r"Every chunk must have a non-empty metadata\['id'\] value",
    ):
        RepresentativeCorpusBuilder._validate_chunk_ids(
            documents
        )


def test_validate_chunk_ids_rejects_duplicate_ids():
    """
    Duplicate identifiers must be rejected.
    """

    documents = [
        Document(
            page_content="First chunk.",
            metadata={"id": "duplicate-id"},
        ),
        Document(
            page_content="Second chunk.",
            metadata={"id": "duplicate-id"},
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate chunk ID: duplicate-id",
    ):
        RepresentativeCorpusBuilder._validate_chunk_ids(
            documents
        )


# ---------------------------------------------------------------------
# Corpus fingerprint
# ---------------------------------------------------------------------


def make_documents() -> list[Document]:
    """
    Return a deterministic representative chunk sequence.
    """

    return [
        Document(
            page_content="First corpus chunk.",
            metadata={"id": "chunk-1"},
        ),
        Document(
            page_content="Second corpus chunk.",
            metadata={"id": "chunk-2"},
        ),
    ]


def test_create_corpus_fingerprint_is_deterministic():
    """
    Identical ordered documents must produce the same fingerprint.
    """

    first = (
        RepresentativeCorpusBuilder
        ._create_corpus_fingerprint(
            make_documents()
        )
    )

    second = (
        RepresentativeCorpusBuilder
        ._create_corpus_fingerprint(
            make_documents()
        )
    )

    assert first == second


def test_create_corpus_fingerprint_is_sha256_hex_digest():
    """
    A corpus fingerprint should be a complete SHA-256 hexadecimal digest.
    """

    fingerprint = (
        RepresentativeCorpusBuilder
        ._create_corpus_fingerprint(
            make_documents()
        )
    )

    assert len(fingerprint) == 64
    assert all(
        character in "0123456789abcdef"
        for character in fingerprint
    )


def test_create_corpus_fingerprint_matches_expected_hash():
    """
    The fingerprint should hash each ordered ID and chunk text,
    separated by null bytes.
    """

    documents = make_documents()

    hasher = hashlib.sha256()

    for document in documents:
        hasher.update(
            document.metadata["id"].encode("utf-8")
        )
        hasher.update(b"\0")
        hasher.update(
            document.page_content.encode("utf-8")
        )
        hasher.update(b"\0")

    expected = hasher.hexdigest()

    result = (
        RepresentativeCorpusBuilder
        ._create_corpus_fingerprint(
            documents
        )
    )

    assert result == expected


def test_create_corpus_fingerprint_changes_when_text_changes():
    """
    Changing chunk content must change the corpus fingerprint.
    """

    original = make_documents()

    modified = [
        Document(
            page_content="First corpus chunk.",
            metadata={"id": "chunk-1"},
        ),
        Document(
            page_content="Modified second corpus chunk.",
            metadata={"id": "chunk-2"},
        ),
    ]

    original_fingerprint = (
        RepresentativeCorpusBuilder
        ._create_corpus_fingerprint(
            original
        )
    )

    modified_fingerprint = (
        RepresentativeCorpusBuilder
        ._create_corpus_fingerprint(
            modified
        )
    )

    assert modified_fingerprint != original_fingerprint


def test_create_corpus_fingerprint_changes_when_id_changes():
    """
    Changing a chunk identifier must change the fingerprint.
    """

    original = make_documents()

    modified = [
        Document(
            page_content="First corpus chunk.",
            metadata={"id": "changed-id"},
        ),
        Document(
            page_content="Second corpus chunk.",
            metadata={"id": "chunk-2"},
        ),
    ]

    original_fingerprint = (
        RepresentativeCorpusBuilder
        ._create_corpus_fingerprint(
            original
        )
    )

    modified_fingerprint = (
        RepresentativeCorpusBuilder
        ._create_corpus_fingerprint(
            modified
        )
    )

    assert modified_fingerprint != original_fingerprint


def test_create_corpus_fingerprint_changes_when_order_changes():
    """
    The fingerprint must preserve the ordering of corpus chunks.
    """

    documents = make_documents()

    forward = (
        RepresentativeCorpusBuilder
        ._create_corpus_fingerprint(
            documents
        )
    )

    reversed_order = (
        RepresentativeCorpusBuilder
        ._create_corpus_fingerprint(
            list(reversed(documents))
        )
    )

    assert reversed_order != forward


def test_create_corpus_fingerprint_for_empty_sequence():
    """
    An empty corpus should produce the SHA-256 digest of empty input.
    """

    fingerprint = (
        RepresentativeCorpusBuilder
        ._create_corpus_fingerprint(
            []
        )
    )

    assert fingerprint == hashlib.sha256().hexdigest()


def test_create_corpus_fingerprint_handles_missing_id():
    """
    A missing ID is currently converted to the string 'None'.

    This test documents the current implementation. ID validation should
    normally run before fingerprint creation.
    """

    documents = [
        Document(
            page_content="Chunk without an ID.",
            metadata={},
        ),
    ]

    hasher = hashlib.sha256()
    hasher.update(b"None")
    hasher.update(b"\0")
    hasher.update(
        b"Chunk without an ID."
    )
    hasher.update(b"\0")

    result = (
        RepresentativeCorpusBuilder
        ._create_corpus_fingerprint(
            documents
        )
    )

    assert result == hasher.hexdigest()