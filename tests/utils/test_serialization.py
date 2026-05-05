import numpy as np

from SRToolkit.utils.serialization import _from_json_safe, _to_json_safe


class TestToJsonSafe:
    def test_ndarray(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _to_json_safe(arr)
        assert result == {"__ndarray__": True, "data": [1.0, 2.0, 3.0]}

    def test_numpy_scalars(self):
        assert _to_json_safe(np.bool_(True)) is True
        assert isinstance(_to_json_safe(np.bool_(True)), bool)
        assert _to_json_safe(np.float64(1.5)) == 1.5
        assert isinstance(_to_json_safe(np.float64(1.5)), float)
        assert _to_json_safe(np.int64(42)) == 42
        assert isinstance(_to_json_safe(np.int64(42)), int)
        assert _to_json_safe(np.str_("hello")) == "hello"
        assert isinstance(_to_json_safe(np.str_("hello")), str)

    def test_dict_recursive(self):
        d = {"a": np.int64(1), "b": np.float64(2.0)}
        result = _to_json_safe(d)
        assert result == {"a": 1, "b": 2.0}
        assert isinstance(result["a"], int)
        assert isinstance(result["b"], float)

    def test_list_recursive(self):
        lst = [np.int64(1), np.float64(2.0)]
        result = _to_json_safe(lst)
        assert result == [1, 2.0]
        assert isinstance(result[0], int)

    def test_tuple_converted_to_list(self):
        result = _to_json_safe((np.int64(1), np.int64(2)))
        assert result == [1, 2]
        assert isinstance(result, list)

    def test_passthrough(self):
        assert _to_json_safe(1) == 1
        assert _to_json_safe(1.5) == 1.5
        assert _to_json_safe("hello") == "hello"
        assert _to_json_safe(None) is None


class TestFromJsonSafe:
    def test_ndarray_dict(self):
        d = {"__ndarray__": True, "data": [1.0, 2.0, 3.0]}
        result = _from_json_safe(d)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_plain_dict_recursive(self):
        d = {"nested": {"__ndarray__": True, "data": [1.0]}}
        result = _from_json_safe(d)
        assert isinstance(result["nested"], np.ndarray)

    def test_list_recursive(self):
        lst = [{"__ndarray__": True, "data": [1.0]}, 42]
        result = _from_json_safe(lst)
        assert isinstance(result[0], np.ndarray)
        assert result[1] == 42

    def test_passthrough(self):
        assert _from_json_safe(1) == 1
        assert _from_json_safe("hello") == "hello"
        assert _from_json_safe(None) is None
