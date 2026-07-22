"""
Tests for RepresentativeCorpusBuilder.
"""

from pathlib import Path

import pytest 
from reportlab.pdfgen import canvas 

from corpus.corpus_models import CorpusDocument
from corpus.representative_corpus import RepresentativeCorpusBuilder 


# ------------------------------------------------------
# Helpers
# ------------------------------------------------------

def create_pdf(
    path: Path,
    pages: list[str],
) -> None: 
    """
    Create a small PDF for testing
    """

    pdf = canvas.Canvas(str(path))

    for text in pages:

        pdf.drawString(
            72,
            720,
            text,
        )

        pdf.showPage()

    pdf.save()

# -------------------------------------------------------
# Constructor validation
# -------------------------------------------------------

def test_constructor(tmp_path):

    RepresentativeCorpusBuilder(
        pdf_directory=tmp_path,
    )

def test_constructor_missing_directory():

    with pytest.raises(
        FileNotFoundError,
    ):
        
        RepresentativeCorpusBuilder(
            pdf_directory=Path(
                "/does/not/exist"
            ),
        )

def test_constructor_not_directory(
    tmp_path,
): 
    
    file = tmp_path / "file.txt"

    file.write_text("test")

    with pytest.raises(
        NotADirectoryError,
    ):
        
        RepresentativeCorpusBuilder(
            pdf_directory=file,
        )

@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        [],
        {},
    ],
)
def test_constructor_invalid_directory_type(
    value,
):
    
    with pytest.raises(
        TypeError,
        match="pdf_directory must be a pathlib.Path.",
    ):
        
        RepresentativeCorpusBuilder(
            pdf_directory=value,
        )

# -------------------------------------------------------
# PDF discovery
# -------------------------------------------------------

def test_discover_pdfs(
    tmp_path,
):
    
    create_pdf(
        tmp_path / "b.pdf",
        ["Page"],
    )

    create_pdf(
        tmp_path / "a.pdf",
        ["Page"],
    )

    (tmp_path / "notes.txt").write_text(
        "ignore"
    )

    builder = RepresentativeCorpusBuilder(
        pdf_directory=tmp_path,
    )

    pdfs = builder._discover_pdfs()

    assert len(pdfs) == 2

    assert pdfs[0].name == "a.pdf"

    assert pdfs[1].name == "b.pdf"

# ----------------------------------------------
# Single-page loading
# ----------------------------------------------

def test_load_pdf_single_page(
    tmp_path,
): 
    
    pdf = tmp_path / "paper.pdf"

    create_pdf(
        pdf,
        [
            "Representative corpus.",
        ],
    )

    builder = RepresentativeCorpusBuilder(
        pdf_directory=tmp_path,
    )

    documents = builder._load_pdf(
        pdf,
    )

    assert len(documents) == 1

    document = documents[0]

    assert isinstance(
        document,
        CorpusDocument, 
    )

    assert document.source == "paper"

    assert document.page == 1

    assert "Representative" in document.text


# ------------------------------------------------------
# Multi-page loading
# ------------------------------------------------------


def test_load_pdf_multiple_pages(
    tmp_path,
):
    
    pdf = tmp_path / "paper.pdf"

    create_pdf(
        pdf,
        [
            "Page One",
            "Page Two",
            "Page Three",
        ],
    )

    builder = RepresentativeCorpusBuilder(
        pdf_directory=tmp_path,
    )

    documents = builder._load_pdf(
        pdf,
    )

    assert len(documents) == 3

    assert documents[0].page == 1

    assert documents[1].page == 2

    assert documents[2].page == 3


# --------------------------------------------------------
# Document construction
# --------------------------------------------------------

def test_build_document(
    tmp_path,
):
    
    builder = RepresentativeCorpusBuilder(
        pdf_directory=tmp_path,
    )

    document = builder._build_document(
        source="TARG",
        page=5,
        text="Evidence",
    )

    assert document.source == "TARG"

    assert document.page == 5

    assert document.text == "Evidence"

    assert document.metadata == {
        "source": "TARG",
        "page": 5,
    }


# --------------------------------------------------------
# Deterministic ordering 
# --------------------------------------------------------

def test_sort_documents(
    tmp_path,
):
    
    builder = RepresentativeCorpusBuilder(
        pdf_directory=tmp_path,
    )

    documents = (

        builder._build_document(
            source="PTES",
            page=2,
            text="A",
        ),

        builder._build_document(
            source="PTES",
            page=1,
            text="B",
        ),

        builder._build_document(
            source="Adaptive",
            page=3,
            text="C"
        ),
    )

    ordered = builder._sort_documents(
        documents,
    )

    assert ordered[0].source == "Adaptive"

    assert ordered[1].page == 1

    assert ordered[2].page == 2

# ----------------------------------------------------------
# Full build
# ----------------------------------------------------------

def test_build(
    tmp_path,
):
    
    create_pdf(
        tmp_path / "paper1.pdf",
        [
            "One",
            "Two",
        ],
    )

    create_pdf(
        tmp_path / "paper2.pdf",
        [
            "Three",
        ],
    )

    builder = RepresentativeCorpusBuilder(
        pdf_directory=tmp_path,
    )

    documents = builder.build()

    assert len(documents) == 3

    assert all(
        isinstance(
            d,
            CorpusDocument,
        )
        for d in documents
    )


# -------------------------------------------------
# Empty directory
# -------------------------------------------------

def test_build_empty_directory(
    tmp_path,
):
    
    builder = RepresentativeCorpusBuilder(
        pdf_directory=tmp_path,
    )

    documents = builder.build()

    assert documents == ()