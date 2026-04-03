"""
Serialization utilities for JSON-safe conversion.

Provides functions to convert numpy types and other non-JSON-serializable
objects to JSON-safe Python types, and back.
"""

from typing import Any

import numpy as np


def _to_json_safe(obj: Any) -> Any:
    """
    Recursively converts numpy types to JSON-safe Python types.

    - ``np.ndarray`` → ``{"__ndarray__": True, "data": <list>}``
    - ``np.floating`` → ``float``
    - ``np.integer`` → ``int``
    - ``np.bool_`` → ``bool``
    """
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "data": obj.tolist()}
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


def _from_json_safe(obj: Any) -> Any:
    """
    Reverses the transformation applied by [_to_json_safe][SRToolkit.utils.serialization._to_json_safe].
    """
    if isinstance(obj, dict):
        if obj.get("__ndarray__"):
            return np.array(obj["data"])
        return {k: _from_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_json_safe(v) for v in obj]
    return obj
