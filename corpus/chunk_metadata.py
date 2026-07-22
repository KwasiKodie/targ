"""
chunk_metadata.py

Deterministic metadata generation for corpus chunks.
"""

from __future__ import annotations 

from types import MappingProxyType 
from typing import Any, Mapping 

class ChunkMetadata: 

    """
    Utility for constructing deterministic chunk metadata.
    """

    @staticmethod 
    def chunk_id(
        *, 
        source: str, 
        page: int, 
        chunk_index: int, 
    ) -> str: 
        
        return(
            f"{source}"
            f"_p{page:03d}"
            f"_c{chunk_index:03d}"
        )
    
    @classmethod 
    def build(
        cls,
        *,
        source: str,
        page: int, 
        chunk_index: int, 
        extra: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        
        metadata: dict[str, Any] = {}

        if extra is not None:
            metadata.update(extra)

        metadata["id"] = cls.chunk_id(
            source=source,
            page=page,
            chunk_index=chunk_index,
        )

        metadata["source"] = source 
        metadata["page"] = page 
        metadata["chunk"] = chunk_index 

        return MappingProxyType(metadata)