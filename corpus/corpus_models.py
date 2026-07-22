"""
corpus_models.py

Immutable document and chunk models used during corpus preprocessing.
"""

from __future__ import annotations 

from dataclasses import dataclass, field 
from types import MappingProxyType 
from typing import Any, Mapping 

@dataclass(frozen=True, slots=True)
class CorpusDocument:
    source: str 
    page: int 
    text: str 
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str):
            raise TypeError("source must be a string.")
        
        if not self.source.strip():
            raise ValueError("source must not be empty.")
        
        if not isinstance(self.page, int):
            raise TypeError("page must be an integer.")
        
        if self.page <= 0:
            raise ValueError("page must be positive.")
        
        if not isinstance(self.text, str):
            raise TypeError("text must be a string.")
        
        if not self.text.strip():
            raise ValueError("text must not be empty.")
        
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

@dataclass(frozen=True, slots=True)
class Chunk:
    id: str 
    source: str 
    page: int 
    chunk_index: int 
    text: str 
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None: 
        if not isinstance(self.id, str):
            raise TypeError("id must be a string.")
        
        if not self.id.strip():
            raise ValueError("id must not be empty.")
        
        if not isinstance(self.source, str):
            raise TypeError("source must be a string.")
        
        if not self.source.strip():
            raise ValueError("source must not be empty.")
        
        if not isinstance(self.page, int):
            raise TypeError("page must be an integer.")
        
        if self.page <= 0:
            raise ValueError("page must be positive.")
        
        if not isinstance(self.chunk_index, int):
            raise TypeError("chunk_index must be an integer.")
        
        if self.chunk_index < 0:
            raise ValueError("chunk_index cannot be negative.")
        
        if not isinstance(self.text, str):
            raise TypeError("text must be a string.")
        
        if not self.text.strip():
            raise ValueError("text must not be empty.")
        
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )