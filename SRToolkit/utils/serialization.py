"""Which values can be saved in SRToolkit results and configs, and how they are stored as JSON.

SRToolkit saves results and configurations as JSON ([ModelResult][SRToolkit.utils.types.ModelResult],
[EvalResult][SRToolkit.utils.types.EvalResult],
[ApproachConfig][SRToolkit.approaches.sr_approach.ApproachConfig],
[HoldoutMetric][SRToolkit.evaluation.result_augmentation.HoldoutMetric]). Everything you put into
augmentation data, ranking data or an approach config must be one of the values in the table below,
nested as deeply as you like; anything else makes ``to_dict`` raise a ``TypeError`` that names the
value's type and its location (e.g. ``obj['augmentations'][2]``).

Internally, ``_to_json_safe`` encodes a value into JSON-compatible Python objects before
``json.dump``, and ``_from_json_safe`` decodes it after ``json.load``. The encoded form appears in
saved files, so changing it affects existing results.

**Encoding.**

| Value | Encoded as |
|---|---|
| ``None``, ``bool``, ``int``, ``float``, ``str`` | unchanged (``nan``/``inf`` stay floats) |
| ``list`` | new list of encoded elements |
| ``tuple`` | ``{"__tuple__": [encoded elements]}`` |
| ``dict`` whose keys are all exactly ``str``, none reserved | new dict of encoded values, key order kept |
| any other ``dict`` | ``{"__dict__": [[encoded key, encoded value], ...]}``, in order |
| ``np.ndarray``, dtype kind ``b``/``i``/``u``/``f``/``U`` | ``{"__ndarray__": True, "dtype": str(arr.dtype), "shape": list(arr.shape), "data": arr.tolist()}`` |
| ``np.ndarray``, ``object`` dtype | same marker with ``"dtype": "object"`` and elements encoded recursively |
| numpy scalar of kind ``b``/``i``/``u``/``f``/``U`` | ``obj.item()`` (Python ``bool``/``int``/``float``/``str``) |

- **Reserved keys** are ``"__ndarray__"``, ``"__tuple__"`` and ``"__dict__"``. A str-keyed dict
  containing one as a key is escaped through ``"__dict__"``. The decoder recognises a marker by the
  presence of a reserved key, regardless of its value, so user data is never mistaken for a marker.
  Field order inside a marker is not part of the format.
- **Dict keys** may be ``None``, ``bool``, ``int``, ``float`` (including ``nan`` and ``-0.0``),
  ``str``, numpy scalars of a supported kind, and tuples of supported keys (recursively); numpy
  scalars inside tuple keys are converted.
- For a 0-d array, ``data`` is a scalar. Any other value raises ``TypeError``: ``complex``,
  ``bytes``, ``set``, numpy scalars such as ``np.bytes_``, ``np.complex128`` and ``np.datetime64``,
  arrays of other dtype kinds, floats wider than 64 bits (``np.longdouble`` on most Linux
  systems; scalars and arrays), masked arrays (``np.ma.MaskedArray``), arbitrary objects.
  Self-referencing (cyclic) structures are not supported.
- ``nan`` and ``inf`` rely on Python's ``json`` writing and reading the non-standard tokens ``NaN`` and
  ``Infinity``; strict JSON parsers reject them.

**Round-trip guarantee.** For every supported ``x``, ``y =
_from_json_safe(json.loads(json.dumps(_to_json_safe(x))))`` equals ``x``: containers equal
element-wise with ``nan`` matching ``nan``, arrays with equal dtype and shape, and leaves of the same
type (``type(y) is type(x)``; ``True`` does not come back as ``1``). Two exceptions:

- numpy scalars come back as the equal Python scalar (``np.float64`` becomes exactly ``float``;
  ``np.float32(0.1)`` becomes ``0.10000000149011612``);
- subclasses of ``int``, ``float`` and ``str`` (e.g. ``IntEnum``, ``StrEnum``) come back as the plain
  base type. ``bool`` keeps its type.

Container subclasses are not preserved: ``OrderedDict`` and ``defaultdict`` come back as ``dict``,
list subclasses as ``list``, a namedtuple as a plain ``tuple``, an ndarray subclass as
``np.ndarray``; array memory order is not preserved.

**Legacy markers.** Files written by older versions contain ``{"__ndarray__": true, "data": ...}``
without ``dtype`` and ``shape``; these decode to ``np.array(data)`` (numpy's default dtype, shape of
``data``).
"""

from typing import Any

import numpy as np


def _to_json_safe(obj: Any) -> Any:
    """Encode ``obj`` into JSON-compatible Python objects, losslessly.

    Follows the encoding table in the [module docstring][SRToolkit.utils.serialization]. Never
    mutates ``obj``; every container in the result is new.

    Args:
        obj: Value to encode: nested ``dict``/``list``/``tuple`` of Python scalars, numpy scalars
            and numpy arrays.

    Returns:
        A value made only of ``None``, ``bool``, ``int``, ``float``, ``str``, ``list`` and
        str-keyed ``dict``, accepted by ``json.dumps``.

    Raises:
        TypeError: If ``obj`` contains an unsupported value or dict key. The message names the type
            (``type(x).__name__``, e.g. ``complex``; ``ndarray of dtype complex128`` for arrays) and
            the location, written as ``obj`` for the top level followed by ``[repr(key)]`` per dict
            value, ``[i]`` per list or tuple element and ``[i, j]`` per object-array element, e.g.
            ``obj['a'][1]``. An unsupported key is reported as ``dict key at <path of the dict>``.

    Examples:
        >>> _to_json_safe({"n": np.int64(3), "w": np.array([1.0, 2.0])})
        {'n': 3, 'w': {'__ndarray__': True, 'dtype': 'float64', 'shape': [2], 'data': [1.0, 2.0]}}
    """
    return _encode(obj, "obj")


def _from_json_safe(obj: Any) -> Any:
    """Decode a value produced by [_to_json_safe][SRToolkit.utils.serialization._to_json_safe].

    A dict containing a reserved key as a key is a marker and is restored: ``"__ndarray__"`` to an
    array with the stored dtype and shape (legacy markers without them use ``np.array(data)``),
    ``"__tuple__"`` to a ``tuple`` and ``"__dict__"`` to a ``dict`` with decoded keys and values.
    Other dicts and lists are rebuilt with decoded contents; all other values, including tuples
    (which JSON never produces), are returned unchanged. Never mutates ``obj``. Malformed markers
    (bad ``dtype`` or ``shape``, malformed ``"__tuple__"`` or ``"__dict__"`` payloads) have
    undefined behaviour.

    Args:
        obj: Encoded value, usually after a ``json.dumps``/``json.loads`` round trip.

    Returns:
        The decoded value.

    Raises:
        KeyError: ``'data'``, if a dict has an ``"__ndarray__"`` key but no ``"data"`` key.

    Examples:
        >>> import json
        >>> x = {"shape": (2, 3), "w": np.zeros(2, dtype=np.float32)}
        >>> y = _from_json_safe(json.loads(json.dumps(_to_json_safe(x))))
        >>> y["shape"], y["w"].dtype
        ((2, 3), dtype('float32'))
    """
    return _decode(obj)


_NO = object()
_RESERVED = ("__ndarray__", "__tuple__", "__dict__")
_ARRAY_KINDS = "biufU"


def _supported_dtype(dtype: np.dtype) -> bool:
    """Whether values of ``dtype`` convert to plain Python scalars (floats only up to 64 bits)."""
    return dtype.kind in _ARRAY_KINDS and not (dtype.kind == "f" and dtype.itemsize > 8)


def _unsupported(obj: Any, path: str) -> TypeError:
    """Return the TypeError for an unsupported value ``obj`` at ``path``."""
    if type(obj) is not np.ma.MaskedArray and isinstance(obj, np.ndarray):
        name = f"ndarray of dtype {obj.dtype}"
    else:
        name = type(obj).__name__
    return TypeError(f"[_to_json_safe] Unsupported type {name} at {path}.")


def _encode_scalar(obj: Any) -> Any:
    """Return ``obj`` as a plain Python scalar, or ``_NO`` if it is not a supported scalar."""
    if obj is None or type(obj) is bool:
        return obj
    if isinstance(obj, np.generic):
        return obj.item() if _supported_dtype(obj.dtype) else _NO
    # Read the base value directly: a subclass may override __str__ (e.g. ``class E(str, Enum)``
    # gives "E.A" for str()), __int__ or __float__.
    if isinstance(obj, int):
        return int.__int__(obj)
    if isinstance(obj, float):
        return float.__float__(obj)
    if isinstance(obj, str):
        return str.__str__(obj)
    return _NO


def _encode_key(key: Any, path: str) -> Any:
    """Encode a dict key of the dict at ``path``; tuples become ``__tuple__`` markers."""
    if isinstance(key, tuple):
        return {"__tuple__": [_encode_key(k, path) for k in key]}
    out = _encode_scalar(key)
    if out is _NO:
        raise TypeError(f"[_to_json_safe] Unsupported type {type(key).__name__} for dict key at {path}.")
    return out


def _encode(obj: Any, path: str) -> Any:
    """Encode ``obj``, located at ``path`` within the top-level value (for error messages)."""
    out = _encode_scalar(obj)
    if out is not _NO:
        return out
    if isinstance(obj, np.ma.MaskedArray):
        raise _unsupported(obj, path)
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "O":
            data = _nested(obj.shape, lambda idx: _encode(obj[idx], f"{path}[{', '.join(map(str, idx))}]"))
        elif _supported_dtype(obj.dtype):
            data = obj.tolist()
        else:
            raise _unsupported(obj, path)
        return {"__ndarray__": True, "dtype": str(obj.dtype), "shape": list(obj.shape), "data": data}
    if isinstance(obj, dict):
        if all(type(k) is str for k in obj) and not any(k in _RESERVED for k in obj):
            return {k: _encode(v, f"{path}[{k!r}]") for k, v in obj.items()}
        return {"__dict__": [[_encode_key(k, path), _encode(v, f"{path}[{k!r}]")] for k, v in obj.items()]}
    if isinstance(obj, tuple):
        return {"__tuple__": [_encode(v, f"{path}[{i}]") for i, v in enumerate(obj)]}
    if isinstance(obj, list):
        return [_encode(v, f"{path}[{i}]") for i, v in enumerate(obj)]
    raise _unsupported(obj, path)


def _decode_key(key: Any) -> Any:
    """Decode a dict key written by ``_encode_key``."""
    if isinstance(key, dict) and "__tuple__" in key:
        return tuple(_decode_key(k) for k in key["__tuple__"])
    return key


def _decode(obj: Any) -> Any:
    """Decode a value written by ``_encode``."""
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            data = obj["data"]
            if "dtype" not in obj:
                return np.array(data)
            if obj["dtype"] == "object":
                out = np.empty(tuple(obj["shape"]), dtype=object)
                # Filled element-wise: np.array would stack equal-length elements into extra axes.
                for idx in np.ndindex(out.shape):
                    value = data
                    for i in idx:
                        value = value[i]
                    out[idx] = _decode(value)
                return out
            return np.array(data, dtype=obj["dtype"]).reshape(obj["shape"])
        if "__tuple__" in obj:
            return tuple(_decode(v) for v in obj["__tuple__"])
        if "__dict__" in obj:
            return {_decode_key(k): _decode(v) for k, v in obj["__dict__"]}
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


def _nested(shape: tuple, element: Any, idx: tuple = ()) -> Any:
    """Return nested lists of ``shape`` whose entry at ``idx`` is ``element(idx)``."""
    if len(idx) == len(shape):
        return element(idx)
    return [_nested(shape, element, idx + (i,)) for i in range(shape[len(idx)])]
