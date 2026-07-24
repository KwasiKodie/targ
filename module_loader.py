"""
Centralized runtime loader.

Loads standalone module exactly once to avoid duplicate class identities.
"""

from __future__ import annotations 

import importlib.util 
import sys 
from pathlib import Path 
from types import ModuleType 

_ROOT = Path(__file__).resolve().parent 
_CACHE: dict[str, ModuleType] = {}

def load(module_name: str, filename: str) -> ModuleType:
    """
    Load a standalone module exactly once.

    Parameters
    ----------
    module_name:
        Unique runtime name. 

    filename:
        Filename relative to this directory.
    """

    if module_name in _CACHE:
        return _CACHE[module_name]
    
    module_path = _ROOT / filename 

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )

    if spec is None or spec.loader is None: 
        raise ImportError(f"Unable to load {filename}")
    
    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module 

    spec.loader.exec_module(module)

    _CACHE[module_name] = module 

    return module 