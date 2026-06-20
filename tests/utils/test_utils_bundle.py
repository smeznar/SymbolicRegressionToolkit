from pathlib import Path

import pytest

from SRToolkit.bundle._relocate import _relocate_class_paths as relocate_class_paths


def _make_bundle(root: Path, layout: dict) -> None:
    """Create files under ``root`` from ``{relative_path: file_text}``."""
    for rel, content in layout.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


class TestRelocateClassPaths:
    def test_unique_class_is_rewritten(self, tmp_path):
        _make_bundle(
            tmp_path,
            {"constraints.py": "class MyConstraint:\n    pass\n"},
        )
        config = {"constraint_class": "alice.pkg.constraints.MyConstraint"}
        out = relocate_class_paths(config, "SRToolkit.user_bundles.alice", tmp_path)
        assert out == {"constraint_class": "SRToolkit.user_bundles.alice.constraints.MyConstraint"}

    def test_srtoolkit_paths_left_untouched(self, tmp_path):
        config = {
            "constraint_class": "SRToolkit.utils.grammar.constraints.MaxDepth",
            "limit": 5,
        }
        out = relocate_class_paths(config, "SRToolkit.user_bundles.alice", tmp_path)
        assert out == config

    def test_ambiguous_resolved_by_module_match(self, tmp_path):
        _make_bundle(
            tmp_path,
            {
                "grammar.py": "class MyThing:\n    pass\n",
                "other.py": "class MyThing:\n    pass\n",
            },
        )
        config = {"approach_class": "alice.deep.grammar.MyThing"}
        out = relocate_class_paths(config, "B", tmp_path)
        assert out["approach_class"] == "B.grammar.MyThing"

    def test_nested_dict_and_list(self, tmp_path):
        _make_bundle(
            tmp_path,
            {
                "samp.py": "class S:\n    pass\n",
                "cb.py": "class C:\n    pass\n",
            },
        )
        config = {
            "approach_class": "alice.something.S",
            "callbacks": [
                {"callback_class": "alice.cb.C", "weight": 1},
            ],
            "nested": {"sampler_class": "alice.samp.S"},
            "irrelevant_class_name": "not.a.class.path",  # not *_class
            "other": 42,
        }
        out = relocate_class_paths(config, "B", tmp_path)
        assert out["approach_class"] == "B.samp.S"
        assert out["callbacks"][0]["callback_class"] == "B.cb.C"
        assert out["nested"]["sampler_class"] == "B.samp.S"
        # non-*_class keys must be left alone
        assert out["irrelevant_class_name"] == "not.a.class.path"
        assert out["other"] == 42

    def test_does_not_mutate_input(self, tmp_path):
        _make_bundle(tmp_path, {"m.py": "class K:\n    pass\n"})
        config = {"approach_class": "alice.m.K"}
        snapshot = dict(config)
        relocate_class_paths(config, "B", tmp_path)
        assert config == snapshot

    def test_missing_class_raises(self, tmp_path):
        _make_bundle(tmp_path, {"m.py": "class K:\n    pass\n"})
        config = {"approach_class": "alice.m.Missing"}
        with pytest.raises(LookupError):
            relocate_class_paths(config, "B", tmp_path)

    def test_init_files_skipped(self, tmp_path):
        _make_bundle(
            tmp_path,
            {
                "__init__.py": "class K:\n    pass\n",
                "m.py": "class K:\n    pass\n",
            },
        )
        config = {"approach_class": "alice.m.K"}
        out = relocate_class_paths(config, "B", tmp_path)
        assert out["approach_class"] == "B.m.K"
