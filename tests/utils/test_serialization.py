import copy
import enum
import json
import math
from collections import OrderedDict, defaultdict, namedtuple

import numpy as np
import pytest
from hypothesis import given, settings
from strategies import serializable_values

from SRToolkit.utils.serialization import _from_json_safe, _to_json_safe


def _rt(x):
    """The documented round trip: encode, JSON dump/load, decode."""
    return _from_json_safe(json.loads(json.dumps(_to_json_safe(x))))


def assert_identical(actual, expected, ordered_keys=True):
    """Equality as defined by the round-trip guarantee: same types, same values, nan matches nan,
    arrays with equal dtype and shape (object arrays compared element-wise), dict key order kept.
    ``ordered_keys=False`` compares dicts as mappings (marker field order is not part of the format)."""
    if isinstance(expected, np.ndarray):
        assert type(actual) is np.ndarray, f"{type(actual)} is not np.ndarray"
        assert actual.dtype == expected.dtype, f"dtype {actual.dtype} != {expected.dtype}"
        assert actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}"
        if expected.dtype == object:
            for a, e in zip(actual.flat, expected.flat):
                assert_identical(a, e, ordered_keys)
        else:
            np.testing.assert_array_equal(actual, expected)
        return
    assert type(actual) is type(expected), f"{actual!r} ({type(actual)}) vs {expected!r} ({type(expected)})"
    if isinstance(expected, dict):
        if ordered_keys:
            assert len(actual) == len(expected)
            for (ak, av), (ek, ev) in zip(actual.items(), expected.items()):
                assert_identical(ak, ek, ordered_keys)
                assert_identical(av, ev, ordered_keys)
        else:
            assert set(actual) == set(expected)
            for k in expected:
                assert_identical(actual[k], expected[k], ordered_keys)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_identical(a, e, ordered_keys)
    elif isinstance(expected, float) and math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == expected


def _obj_array(items, shape):
    """Object array of the given shape filled from ``items`` without numpy unpacking nested lists."""
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr.reshape(shape)


class _Color(enum.IntEnum):
    RED = 1


class _Name(str, enum.Enum):
    A = "a"


class _MyFloat(float):
    pass


class _MyList(list):
    pass


class _MyArray(np.ndarray):
    pass


_Point = namedtuple("_Point", "x y")
# np.longdouble is float64 on Windows and Apple silicon; only wider ones are unsupported.
_NEEDS_WIDE_LONGDOUBLE = pytest.mark.skipif(
    np.dtype(np.longdouble).itemsize <= 8, reason="np.longdouble is 64-bit on this platform"
)
NAN = float("nan")
INF = float("inf")


class TestRoundTrip:
    @pytest.mark.parametrize(
        ("x", "expected"),
        [
            (None, None),
            (True, True),
            (-3, -3),
            (2**70, 2**70),
            (1.5, 1.5),
            (NAN, NAN),
            (INF, INF),
            ("s", "s"),
            (np.float32(0.1), 0.10000000149011612),
            (np.uint64(2**64 - 1), 18446744073709551615),
            (_Color.RED, 1),
            (_Name.A, "a"),
            (_MyFloat(2.5), 2.5),
        ],
        ids=[
            "none",
            "bool",
            "int",
            "big-int",
            "float",
            "nan",
            "inf",
            "str",
            "np.float32-widens",
            "np.uint64-max",
            "IntEnum-to-int",
            "str-enum-to-str",
            "float-subclass-to-float",
        ],
    )
    def test_round_trip_scalars(self, x, expected):
        assert_identical(_rt(x), expected)

    @pytest.mark.parametrize(
        ("x", "expected"),
        [
            (np.array(2.5), np.array(2.5)),
            (np.zeros((0, 3)), np.zeros((0, 3))),
            (np.zeros((2, 0, 4), dtype=np.int64), np.zeros((2, 0, 4), dtype=np.int64)),
            (np.zeros((3, 0)), np.zeros((3, 0))),
            (np.array([0.1, -2.0], dtype=np.float32), np.array([0.1, -2.0], dtype=np.float32)),
            (np.array([2**64 - 1], dtype=np.uint64), np.array([2**64 - 1], dtype=np.uint64)),
            (np.array([-(2**63), 2**63 - 1]), np.array([-(2**63), 2**63 - 1])),
            (np.array([[True], [False]]), np.array([[True], [False]])),
            (np.array(["ab", ""], dtype="<U3"), np.array(["ab", ""], dtype="<U3")),
            (np.array([1.5, 2.0], dtype=">f8"), np.array([1.5, 2.0], dtype=">f8")),
            (np.asfortranarray(np.arange(6.0).reshape(2, 3)), np.arange(6.0).reshape(2, 3)),
            (np.array([NAN, INF, -INF]), np.array([NAN, INF, -INF])),
            (np.array([1.0, 2.0]).view(_MyArray), np.array([1.0, 2.0])),
        ],
        ids=[
            "0-d",
            "shape-(0,3)",
            "shape-(2,0,4)",
            "shape-(3,0)",
            "float32",
            "uint64-max",
            "int64-extremes",
            "bool-2d",
            "<U3",
            ">f8-byte-order",
            "fortran-order",
            "nan-inf",
            "ndarray-subclass",
        ],
    )
    def test_round_trip_arrays(self, x, expected):
        assert_identical(_rt(x), expected)

    @pytest.mark.parametrize(
        ("x", "expected"),
        [
            (
                _obj_array([np.array([1.0, 2.0]), np.array([3])], (2,)),
                _obj_array([np.array([1.0, 2.0]), np.array([3])], (2,)),
            ),
            (
                _obj_array([(1, 2), {"a": np.int64(1)}, None], (3,)),
                _obj_array([(1, 2), {"a": 1}, None], (3,)),
            ),
            (_obj_array([[1, 2], [3, 4]], (2,)), _obj_array([[1, 2], [3, 4]], (2,))),
            (_obj_array([1, "a"], (2, 1)), _obj_array([1, "a"], (2, 1))),
        ],
        ids=["ragged-arrays", "mixed-elements", "equal-length-lists-keep-shape-(2,)", "2d-object"],
    )
    def test_round_trip_object_arrays(self, x, expected):
        assert_identical(_rt(x), expected)

    @pytest.mark.parametrize(
        ("x", "expected"),
        [
            (((1, (2,)), [3]), ((1, (2,)), [3])),
            ([(), {}, []], [(), {}, []]),
            ({"b": 1, "a": 2}, {"b": 1, "a": 2}),
            ({"a": (np.int64(1), [np.float32(0.5)])}, {"a": (1, [0.5])}),
            (OrderedDict([("b", 1), ("a", 2)]), {"b": 1, "a": 2}),
            (defaultdict(list, {"x": [1]}), {"x": [1]}),
            (_MyList([1, (2,)]), [1, (2,)]),
            (_Point(1, 2), (1, 2)),
        ],
        ids=[
            "nested-tuple",
            "empties-in-list",
            "key-order",
            "nested-numpy-scalars",
            "OrderedDict-to-dict",
            "defaultdict-to-dict",
            "list-subclass-to-list",
            "namedtuple-to-tuple",
        ],
    )
    def test_round_trip_containers(self, x, expected):
        assert_identical(_rt(x), expected)

    @pytest.mark.parametrize(
        ("x", "expected"),
        [
            ({1: "a", 1.5: "b", None: "c"}, {1: "a", 1.5: "b", None: "c"}),
            ({True: "a"}, {True: "a"}),
            ({((1, 2), "x"): 1}, {((1, 2), "x"): 1}),
            ({(np.int64(1), "a"): 1}, {(1, "a"): 1}),
            ({1: "a", "1": "b"}, {1: "a", "1": "b"}),
            ({"a": 1, 2: 3}, {"a": 1, 2: 3}),
            ({np.int64(1): "a"}, {1: "a"}),
            ({np.str_("k"): 1}, {"k": 1}),
            ({1: np.array([1.0])}, {1: np.array([1.0])}),
            ({"__ndarray__": True, "data": [1]}, {"__ndarray__": True, "data": [1]}),
            ({"__tuple__": [1]}, {"__tuple__": [1]}),
            ({"__dict__": 1}, {"__dict__": 1}),
        ],
        ids=[
            "scalar-keys",
            "bool-key",
            "nested-tuple-key",
            "numpy-scalar-in-tuple-key",
            "int-and-str-keys-distinct",
            "mixed-keys-order",
            "np.int64-key",
            "np.str_-key",
            "array-value-under-int-key",
            "reserved-__ndarray__",
            "reserved-__tuple__",
            "reserved-__dict__",
        ],
    )
    def test_round_trip_dict_keys(self, x, expected):
        assert_identical(_rt(x), expected)

    @settings(max_examples=50)
    @given(serializable_values())
    def test_round_trip_property(self, x):
        enc = _to_json_safe(x)
        # Returns contract: JSON-native, so a JSON round trip of the encoded form changes nothing.
        assert_identical(json.loads(json.dumps(enc)), enc)
        assert_identical(_from_json_safe(json.loads(json.dumps(enc))), x)


class TestToJsonSafe:
    @pytest.mark.parametrize(
        ("x", "expected"),
        [
            (
                np.array([[1, 2], [3, 4]], dtype=np.int8),
                {"__ndarray__": True, "dtype": "int8", "shape": [2, 2], "data": [[1, 2], [3, 4]]},
            ),
            (np.array(2.5), {"__ndarray__": True, "dtype": "float64", "shape": [], "data": 2.5}),
            (
                np.array([NAN, INF]),
                {"__ndarray__": True, "dtype": "float64", "shape": [2], "data": [NAN, INF]},
            ),
            (
                _obj_array([np.array([1.0, 2.0]), np.array([3])], (2,)),
                {
                    "__ndarray__": True,
                    "dtype": "object",
                    "shape": [2],
                    "data": [
                        {"__ndarray__": True, "dtype": "float64", "shape": [2], "data": [1.0, 2.0]},
                        {"__ndarray__": True, "dtype": "int64", "shape": [1], "data": [3]},
                    ],
                },
            ),
        ],
        ids=["int8-2d", "0-d-scalar-data", "nan-inf-floats", "object-ragged"],
    )
    def test_array_marker_format(self, x, expected):
        assert_identical(_to_json_safe(x), expected, ordered_keys=False)

    @pytest.mark.parametrize(
        ("x", "expected"),
        [
            ((1, np.int64(2)), {"__tuple__": [1, 2]}),
            ({"a": 1, (1, 2): "b"}, {"__dict__": [["a", 1], [{"__tuple__": [1, 2]}, "b"]]}),
            ({"__tuple__": 5}, {"__dict__": [["__tuple__", 5]]}),
            ({"a": (1,)}, {"a": {"__tuple__": [1]}}),
        ],
        ids=["tuple", "mixed-and-tuple-keys", "reserved-key-escaped", "str-keyed-dict-plain"],
    )
    def test_container_marker_format(self, x, expected):
        assert_identical(_to_json_safe(x), expected)

    @pytest.mark.parametrize(
        ("x", "expected"),
        [
            (np.float64(1.5), 1.5),
            (np.int64(3), 3),
            (np.bool_(False), False),
            (np.str_("a"), "a"),
        ],
        ids=["float64", "int64", "bool_", "str_"],
    )
    def test_numpy_scalars_encode_to_exact_python_types(self, x, expected):
        # A JSON round trip cannot see np.float64/np.str_ left in place (they subclass float/str).
        assert_identical(_to_json_safe(x), expected)

    @pytest.mark.parametrize(
        ("x", "type_name", "location"),
        [
            (1j, "complex", "obj"),
            ({"augmentations": [0, 0, b"x"]}, "bytes", "obj['augmentations'][2]"),
            (np.datetime64("2020-01-01"), "datetime64", "obj"),
            ({"o": object()}, "object", "obj['o']"),
            ({"t": (1, 1j)}, "complex", "obj['t'][1]"),
            ({1: {2}}, "set", "obj[1]"),
            ({"a": np.array([1j])}, "ndarray of dtype complex128", "obj['a']"),
            (_obj_array([1, {2}], (1, 2)), "set", "obj[0, 1]"),
            ({"d": {frozenset(): 1}}, "frozenset", "dict key at obj['d']"),
            ({"d": {(1, (b"x",)): 1}}, "bytes", "dict key at obj['d']"),
            ({"m": np.ma.array([1.0, 2.0], mask=[0, 1])}, "MaskedArray", "obj['m']"),
            pytest.param(np.longdouble(1.5), "longdouble", "obj", marks=_NEEDS_WIDE_LONGDOUBLE),
            pytest.param(
                {"a": np.array([1.5], dtype=np.longdouble)},
                f"ndarray of dtype {np.dtype(np.longdouble)}",
                "obj['a']",
                marks=_NEEDS_WIDE_LONGDOUBLE,
            ),
        ],
        ids=[
            "complex",
            "bytes-docstring-example",
            "datetime64",
            "object",
            "in-tuple",
            "under-int-key",
            "complex-array",
            "object-array-element",
            "frozenset-key-nested",
            "bad-element-in-nested-tuple-key",
            "masked-array",
            "longdouble-scalar",
            "longdouble-array",
        ],
    )
    def test_unsupported_value_raises_type_error_with_location(self, x, type_name, location):
        with pytest.raises(TypeError) as excinfo:
            _to_json_safe(x)
        assert type_name in str(excinfo.value)
        assert location in str(excinfo.value)

    def test_does_not_mutate_input_and_returns_new_containers(self):
        x = {
            "l": [1, [2]],
            "t": (1, [3]),
            "e": {},
            "a": np.array([1.0, 2.0]),
            "o": _obj_array([[1]], (1,)),
        }
        snapshot = copy.deepcopy(x)
        out = _to_json_safe(x)
        assert_identical(x, snapshot)
        assert out is not x
        assert out["l"] is not x["l"]
        assert out["l"][1] is not x["l"][1]
        assert out["e"] is not x["e"]
        assert out["o"]["data"][0] is not x["o"][0]
        out["a"]["data"][0] = 99.0
        out["t"]["__tuple__"][1].append(4)
        assert_identical(x, snapshot)


class TestFromJsonSafe:
    @pytest.mark.parametrize(
        ("marker", "expected"),
        [
            ({"__ndarray__": True, "data": [[1, 2], [3, 4]]}, np.array([[1, 2], [3, 4]])),
            ({"__ndarray__": True, "data": []}, np.array([])),
            ({"__ndarray__": False, "data": [1.5]}, np.array([1.5])),
        ],
        ids=["int-matrix", "empty", "recognised-by-key-presence"],
    )
    def test_legacy_marker_decodes_with_numpy_defaults(self, marker, expected):
        assert_identical(_from_json_safe(marker), expected)

    @pytest.mark.parametrize(
        "obj",
        [
            {"__ndarray__": True},
            {"__ndarray__": False},
            [{"a": {"__ndarray__": True}}],
        ],
        ids=["no-data", "falsy-value-no-data", "nested"],
    )
    def test_marker_without_data_raises_key_error(self, obj):
        with pytest.raises(KeyError, match="data"):
            _from_json_safe(obj)

    @pytest.mark.parametrize(
        "obj",
        [1, None, (1, {"__tuple__": [1]})],
        ids=["int", "none", "tuple-not-decoded"],
    )
    def test_non_container_values_returned_unchanged(self, obj):
        assert _from_json_safe(obj) is obj

    def test_does_not_mutate_input_and_returns_new_containers(self):
        enc = {
            "d": {"__dict__": [[1, [1, 2]]]},
            "t": {"__tuple__": [[1]]},
            "w": {"__ndarray__": True, "dtype": "float64", "shape": [2], "data": [1.0, 2.0]},
            "l": [[1]],
        }
        snapshot = copy.deepcopy(enc)
        out = _from_json_safe(enc)
        assert enc == snapshot
        assert out is not enc
        assert out["l"] is not enc["l"]
        assert out["l"][0] is not enc["l"][0]
        out["w"][0] = 99.0
        out["d"][1].append(3)
        out["t"][0].append(2)
        assert enc == snapshot
