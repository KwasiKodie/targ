"""
Tests for Chunker constructor.
"""

import pytest 

from corpus.chunker import Chunker 

def test_constructor_valid():
    
    chunker = Chunker()

    assert chunker.chunk_size == 1000

    assert chunker.chunk_overlap == 100

    assert chunker.minimum_chunk_size == 200

@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -100,
    ],
)
def test_constructor_invalid_chunk_size(
    value,
):
    
    with pytest.raises(
        ValueError,
        match="chunk_size must be positive.",
    ):
        
        Chunker(
            chunk_size=value,
        )

def test_constructor_negative_overlap():

    with pytest.raises(
        ValueError,
        match="chunk_overlap cannot be negative.",
    ):
        
        Chunker(
            chunk_overlap=-1
        )

@pytest.mark.parametrize(
    "chunk_size, overlap",
    [
        (1000, 1000),
        (1000, 1200),
        (500, 500),
        (500, 900),
    ],
)
def test_constructor_overlap_not_smaller_than_chunk_size(
    chunk_size,
    overlap,
):
    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size.",
    ):
        
        Chunker(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )

@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -200,
    ],
)
def test_constructor_invalid_minimum_chunk_size(
    value,
):
    
    with pytest.raises(
        ValueError,
        match="minimum_chunk_size must be positive.",
    ):
        Chunker(
            minimum_chunk_size=value,
        )

@pytest.mark.parametrize(
    "value",
    [
        None,
        1.5,
        "1000",
        [],
    ],
)
def test_constructor_invalid_chunk_size_type(
    value,
):
    
    with pytest.raises(
        TypeError,
        match="chunk_size must be an integer.",
    ):
        
        Chunker(
            chunk_size=value,
        )

@pytest.mark.parametrize(
    "value",
    [
        None,
        50.5,
        "100",
        {},
    ],
)
def test_constructor_invalid_chunk_overlap_type(
    value,
):
    
    with pytest.raises(
        TypeError,
        match="chunk_overlap must be an integer.",
    ):
        
        Chunker(
            chunk_overlap=value,
        )

@pytest.mark.parametrize(
    "value",
    [
        None,
        200.5,
        "200",
        (),
    ],
)
def test_constructor_invalid_minimum_chunk_size_type(
    value,
):
    
    with pytest.raises(
        TypeError,
        match="minimum_chunk_size must be an integer.",
    ):
        
        Chunker(
            minimum_chunk_size=value,
        )