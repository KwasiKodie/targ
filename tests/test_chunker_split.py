"""
Tests for Chunker._split_text().
"""

import pytest 

from corpus.chunker import Chunker 

def test_split_short_document():
    
    chunker = Chunker()

    text = "A" * 500

    chunks = chunker._split_text(text)

    assert len(chunks) == 1

    assert chunks[0] == text 

def test_split_exact_chunk_size():

    chunker = Chunker()

    text = "A" * chunker.chunk_size 

    chunks = chunker._split_text(text)

    assert len(chunks) == 1 

    assert len(chunks[0]) == chunker.chunk_size 

def test_split_multiple_chunks():

    chunker = Chunker()

    text = "A" * 2500

    chunks = chunker._split_text(text)

    assert len(chunks) == 3

    assert len(chunks[0]) == 1000

    assert len(chunks[1]) == 1000

    assert len(chunks[2]) == 700

def test_split_overlap_preserved():

    chunker = Chunker()

    alphabet = "".join(

        chr(65 + (i % 26))

        for i in range(2500)
    )

    chunks = chunker._split_text(alphabet)

    overlap = chunker.chunk_overlap 

    assert chunks[0][-overlap:] == chunks[1][:overlap]

def test_split_discards_small_final_chunk():

    chunker = Chunker()

    text = "A" * 1950

    chunks = chunker._split_text(text)

    assert len(chunks) == 2

    assert len(chunks[0]) == 1000

    assert len(chunks[1]) == 1000

def test_split_retains_minimum_final_chunk():

    chunker = Chunker()

    text = "A" * 2000

    chunks = chunker._split_text(text)

    assert len(chunks) == 3

    assert len(chunks[0]) == 1000

    assert len(chunks[1]) == 1000

    assert len(chunks[2]) == 200

def test_split_empty_string():

    chunker = Chunker()

    chunks = chunker._split_text("")

    assert chunks == ()

def test_split_whitespace_only():

    chunker = Chunker()

    chunks = chunker._split_text("    ")

    assert chunks == ()


@pytest.mark.parametrize(
    "value",
    [
        None,
        123,
        [],
        {},
    ],
)
def test_split_invalid_type(value):

    chunker = Chunker()

    with pytest.raises(
        TypeError,
        match="text must be a string.",
    ):
        
        chunker._split_text(value)