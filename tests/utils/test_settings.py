"""Spec tests for SRToolkit.utils.settings (sections, Settings, saving and loading)."""

import copy
import json
import pickle
import sys
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

from SRToolkit.bundle import _store, install, pack
from SRToolkit.utils.expression_simplifier import DefaultRules
from SRToolkit.utils.settings import (
    BEDSettings,
    EstimationSettings,
    EvaluatorSettings,
    RMSESettings,
    Settings,
    SettingsSection,
)


@dataclass(frozen=True, eq=False)
class _MySection(SettingsSection):
    name = "my_ranking"
    alpha: float = 0.5


@dataclass(frozen=True, eq=False)
class _OtherSection(SettingsSection):
    name = "my_ranking"
    alpha: float = 0.5


@dataclass(frozen=True, eq=False)
class _BEDChild(BEDSettings):
    pass


_MY_PATH = f"{_MySection.__module__}.{_MySection.__qualname__}"
_OTHER_PATH = f"{_OtherSection.__module__}.{_OtherSection.__qualname__}"
_RULES = DefaultRules(exclude=["inverses"])
_RULES_SAVED = {"rules_class": "SRToolkit.utils.expression_simplifier.DefaultRules", "exclude": ["inverses"]}
_GT = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


# --------------------------------------------------------------------------- #
# Sections: names, defaults, validation
# --------------------------------------------------------------------------- #
class TestSections:
    @pytest.mark.parametrize(
        "cls, name",
        [
            (EvaluatorSettings, "evaluator"),
            (EstimationSettings, "estimation"),
            (RMSESettings, "rmse"),
            (BEDSettings, "bed"),
        ],
        ids=["evaluator", "estimation", "rmse", "bed"],
    )
    def test_builtin_section_names(self, cls, name):
        assert cls.name == name

    @pytest.mark.parametrize(
        "cls, field, default",
        [
            (EvaluatorSettings, "simplify_expr", False),
            (EvaluatorSettings, "canonicalise_constants", False),
            (EvaluatorSettings, "fold_constants", "none"),
            (EvaluatorSettings, "simplify_max_iterations", 30),
            (EvaluatorSettings, "simplify_max_nodes", 10_000),
            (EvaluatorSettings, "simplify_max_time", None),
            (EvaluatorSettings, "simplify_fold_aware", True),
            (EvaluatorSettings, "max_consecutive_skips", 10000),
            (EvaluatorSettings, "max_constants", 8),
            (EvaluatorSettings, "max_expr_length", -1),
            (EvaluatorSettings, "backend", "stack"),
            (EvaluatorSettings, "ranking_seed", None),
            (EstimationSettings, "method", "L-BFGS-B"),
            (EstimationSettings, "tol", 1e-6),
            (EstimationSettings, "gtol", 1e-3),
            (EstimationSettings, "max_iter", 100),
            (EstimationSettings, "initialization", "random"),
            (BEDSettings, "num_points_sampled", 64),
            (BEDSettings, "num_consts_sampled", 32),
            (BEDSettings, "bed_gt", None),
        ],
        ids=lambda v: v if isinstance(v, str) else None,
    )
    def test_sections_have_documented_defaults(self, cls, field, default):
        assert getattr(cls(), field) == default

    @pytest.mark.parametrize(
        "kwargs, exc, match",
        [
            ({"simplify_expr": 1}, TypeError, "evaluator.simplify_expr"),
            ({"canonicalise_constants": "yes"}, TypeError, "evaluator.canonicalise_constants"),
            ({"simplify_fold_aware": None}, TypeError, "evaluator.simplify_fold_aware"),
            ({"simplify_rules": "default"}, TypeError, "evaluator.simplify_rules"),
            ({"simplify_max_time": "5"}, TypeError, "evaluator.simplify_max_time"),
            ({"simplify_max_time": True}, TypeError, "evaluator.simplify_max_time"),
            ({"simplify_max_iterations": 3.0}, TypeError, "evaluator.simplify_max_iterations"),
            ({"simplify_max_nodes": "10"}, TypeError, "evaluator.simplify_max_nodes"),
            ({"max_consecutive_skips": 1.5}, TypeError, "evaluator.max_consecutive_skips"),
            ({"max_constants": None}, TypeError, "evaluator.max_constants"),
            ({"max_expr_length": 2.0}, TypeError, "evaluator.max_expr_length"),
            ({"ranking_seed": 1.5}, TypeError, "evaluator.ranking_seed"),
            ({"fold_constants": "all"}, ValueError, r"^\[Settings\] evaluator.fold_constants"),
            ({"backend": "numba"}, ValueError, "evaluator.backend"),
            ({"simplify_max_iterations": -1}, ValueError, "evaluator.simplify_max_iterations"),
            ({"simplify_max_nodes": -1}, ValueError, "evaluator.simplify_max_nodes"),
            ({"ranking_seed": -1}, ValueError, "evaluator.ranking_seed"),
            ({"simplify_max_time": 0}, ValueError, "evaluator.simplify_max_time"),
            ({"simplify_max_time": float("inf")}, ValueError, "evaluator.simplify_max_time"),
            ({"max_consecutive_skips": 0}, ValueError, "evaluator.max_consecutive_skips"),
        ],
        ids=lambda v: "-".join(f"{k}={x}" for k, x in v.items()) if isinstance(v, dict) else "",
    )
    def test_evaluator_settings_rejects_invalid(self, kwargs, exc, match):
        with pytest.raises(exc, match=match):
            EvaluatorSettings(**kwargs)

    @pytest.mark.parametrize(
        "kwargs, exc, match",
        [
            ({"method": 5}, TypeError, "estimation.method"),
            ({"tol": "1e-6"}, TypeError, "estimation.tol"),
            ({"tol": True}, TypeError, "estimation.tol"),
            ({"gtol": None}, TypeError, "estimation.gtol"),
            ({"max_iter": 1.5}, TypeError, "estimation.max_iter"),
            ({"max_iter": True}, TypeError, "estimation.max_iter"),
            ({"method": "foo"}, ValueError, "estimation.method"),
            ({"tol": 0}, ValueError, "estimation.tol"),
            ({"tol": float("nan")}, ValueError, "estimation.tol"),
            ({"gtol": -1}, ValueError, "estimation.gtol"),
            ({"gtol": float("inf")}, ValueError, "estimation.gtol"),
            ({"max_iter": -1}, ValueError, "estimation.max_iter"),
            ({"initialization": "zeros"}, ValueError, "estimation.initialization"),
        ],
        ids=lambda v: "-".join(f"{k}={x}" for k, x in v.items()) if isinstance(v, dict) else "",
    )
    def test_estimation_settings_rejects_invalid(self, kwargs, exc, match):
        with pytest.raises(exc, match=match):
            EstimationSettings(**kwargs)

    @pytest.mark.parametrize(
        "kwargs, exc, match",
        [
            ({"num_points_sampled": 64.0}, TypeError, "bed.num_points_sampled"),
            ({"num_consts_sampled": "32"}, TypeError, "bed.num_consts_sampled"),
            ({"num_points_sampled": 0}, ValueError, "bed.num_points_sampled"),
            ({"num_points_sampled": -2}, ValueError, "bed.num_points_sampled"),
            ({"num_consts_sampled": 0}, ValueError, "bed.num_consts_sampled"),
            ({"bed_gt": np.zeros(32)}, ValueError, "bed.bed_gt"),
            ({"bed_gt": np.zeros((5, 31))}, ValueError, "bed.bed_gt"),
            ({"bed_gt": np.zeros((5, 4)), "num_consts_sampled": 5}, ValueError, "bed.bed_gt"),
        ],
        ids=[
            "points_float",
            "consts_str",
            "points_0",
            "points_-2",
            "consts_0",
            "gt_1d",
            "gt_31_cols",
            "gt_4_of_5_cols",
        ],
    )
    def test_bed_settings_rejects_invalid(self, kwargs, exc, match):
        with pytest.raises(exc, match=match):
            BEDSettings(**kwargs)

    @pytest.mark.parametrize(
        "cls, field, value",
        [
            (EvaluatorSettings, "simplify_max_iterations", 0),
            (EvaluatorSettings, "simplify_max_nodes", 0),
            (EvaluatorSettings, "ranking_seed", 0),
            (EvaluatorSettings, "max_consecutive_skips", -1),
            (EvaluatorSettings, "max_constants", -1),
            (EvaluatorSettings, "simplify_max_time", 1),
            (EvaluatorSettings, "simplify_max_time", 0.5),
            (EstimationSettings, "max_iter", 0),
            (EstimationSettings, "tol", 1),
            (BEDSettings, "num_points_sampled", -1),
            (BEDSettings, "num_points_sampled", 1),
            (EvaluatorSettings, "backend", "codegen"),
            (EvaluatorSettings, "backend", "stack_py"),
            (EvaluatorSettings, "fold_constants", "default"),
            (EvaluatorSettings, "fold_constants", "keep"),
            (EstimationSettings, "initialization", "mean"),
        ],
        ids=lambda v: v if isinstance(v, str) else None,
    )
    def test_sections_accept_domain_boundaries(self, cls, field, value):
        assert getattr(cls(**{field: value}), field) == value

    # Reason: clause (method "stored in SciPy's spelling").
    @pytest.mark.parametrize("given", ["nelder-mead", "NELDER-MEAD", "Nelder-Mead"])
    def test_method_is_stored_in_scipy_spelling(self, given):
        assert EstimationSettings(method=given).method == "Nelder-Mead"

    def test_method_in_other_letter_case_equals_default(self):
        assert Settings(estimation={"method": "l-bfgs-b"}) == Settings()

    def test_bed_accepts_one_constant_column(self):
        gt = np.ones((3, 1))
        np.testing.assert_array_equal(BEDSettings(num_consts_sampled=1, bed_gt=gt).bed_gt, gt)

    @pytest.mark.parametrize(
        "cls, field",
        [
            (EvaluatorSettings, "simplify_max_iterations"),
            (EvaluatorSettings, "simplify_max_nodes"),
            (EvaluatorSettings, "max_consecutive_skips"),
            (EvaluatorSettings, "max_constants"),
            (EvaluatorSettings, "max_expr_length"),
            (EvaluatorSettings, "ranking_seed"),
            (EstimationSettings, "max_iter"),
            (BEDSettings, "num_points_sampled"),
            (BEDSettings, "num_consts_sampled"),
        ],
        ids=lambda v: v if isinstance(v, str) else None,
    )
    def test_numpy_ints_are_converted_to_int(self, cls, field):
        value = getattr(cls(**{field: np.int64(5)}), field)
        assert (type(value), value) == (int, 5)

    @pytest.mark.parametrize(
        "make",
        [
            lambda: BEDSettings(num_consts_sampled=1, bed_gt=[[1.0], [2.0]]),
            lambda: BEDSettings.from_dict(BEDSettings(num_consts_sampled=1, bed_gt=[[1.0], [2.0]]).to_dict()),
        ],
        ids=["list_input", "from_dict"],
    )
    def test_bed_gt_is_stored_as_ndarray(self, make):
        gt = make().bed_gt
        assert isinstance(gt, np.ndarray)
        np.testing.assert_array_equal(gt, [[1.0], [2.0]])

    # Reason: clause (bed_gt "stored as a read-only copy").
    def test_bed_gt_is_a_read_only_copy(self):
        given = _GT.copy()
        stored = BEDSettings(num_consts_sampled=2, bed_gt=given).bed_gt
        given[0, 0] = 99.0
        assert stored[0, 0] == 1.0
        with pytest.raises(ValueError):
            stored[0, 0] = 99.0

    @pytest.mark.parametrize(
        "obj, attr, value",
        [
            (EstimationSettings(), "max_iter", 5),
            (Settings(), "estimation", EstimationSettings(max_iter=5)),
            (Settings(), "foo", 1),
        ],
        ids=["section_field", "settings_property", "settings_new_attribute"],
    )
    def test_settings_and_sections_are_immutable(self, obj, attr, value):
        with pytest.raises(AttributeError):
            setattr(obj, attr, value)


# --------------------------------------------------------------------------- #
# SettingsSection: changed, to_dict, from_dict, equality
# --------------------------------------------------------------------------- #
class TestSettingsSection:
    @pytest.mark.parametrize(
        "section, expected",
        [
            (EstimationSettings(), {}),
            (EstimationSettings(max_iter=100, tol=1e-6), {}),
            (EstimationSettings(max_iter=5), {"max_iter": 5}),
            (EvaluatorSettings(simplify_rules=DefaultRules()), {}),
            (EvaluatorSettings(simplify_rules=_RULES), {"simplify_rules": _RULES}),
            (BEDSettings(), {}),
            (RMSESettings(), {}),
        ],
        ids=["defaults", "given_at_default", "changed", "fresh_default_rules", "changed_rules", "bed", "rmse"],
    )
    def test_changed_lists_fields_differing_from_defaults(self, section, expected):
        assert section.changed() == expected

    def test_section_to_dict_saves_codec_field_through_to_dict(self):
        section = EvaluatorSettings(simplify_rules=_RULES, max_constants=3)
        assert section.to_dict() == {"simplify_rules": _RULES_SAVED, "max_constants": 3}

    def test_section_from_dict_fills_missing_fields_with_defaults(self):
        assert EstimationSettings.from_dict({"max_iter": 5}) == EstimationSettings(max_iter=5)

    @pytest.mark.parametrize(
        "data, exc, match",
        [
            ({"maxiter": 5}, TypeError, "maxiter"),
            ({"tol": -1}, ValueError, "estimation.tol"),
            ({"max_iter": "5"}, TypeError, "estimation.max_iter"),
        ],
        ids=["unknown_field", "out_of_range", "wrong_type"],
    )
    def test_section_from_dict_rejects_invalid(self, data, exc, match):
        with pytest.raises(exc, match=match):
            EstimationSettings.from_dict(data)

    @pytest.mark.parametrize(
        "left, right, expected",
        [
            (EvaluatorSettings(), EvaluatorSettings(), True),
            (EstimationSettings(max_iter=5), EstimationSettings(max_iter=6), False),
            (BEDSettings(num_consts_sampled=2, bed_gt=_GT), BEDSettings(num_consts_sampled=2, bed_gt=_GT.copy()), True),
            (BEDSettings(num_consts_sampled=2, bed_gt=_GT), BEDSettings(num_consts_sampled=2, bed_gt=_GT + 1), False),
            (BEDSettings(num_consts_sampled=2, bed_gt=_GT), BEDSettings(num_consts_sampled=2), False),
            (BEDSettings(), _BEDChild(), False),
            (_MySection(alpha=0.3), _OtherSection(alpha=0.3), False),
            (BEDSettings(), None, False),
        ],
        ids=[
            "evaluator_defaults",
            "diff_value",
            "equal_arrays",
            "diff_arrays",
            "array_vs_none",
            "subclass",
            "other_class",
            "not_section",
        ],
    )
    def test_section_equality(self, left, right, expected):
        assert (left == right) is expected


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
class TestSettingsConstruction:
    @pytest.mark.parametrize(
        "kwargs, prop, expected",
        [
            ({}, "evaluator", EvaluatorSettings()),
            ({"estimation": {"max_iter": 200}}, "estimation", EstimationSettings(max_iter=200)),
            ({"rmse": None}, "rmse", RMSESettings()),
            ({"rmse": RMSESettings()}, "rmse", RMSESettings()),
            ({"bed": BEDSettings(num_consts_sampled=4)}, "bed", BEDSettings(num_consts_sampled=4)),
            ({"estimation": {}}, "estimation", EstimationSettings()),
        ],
        ids=["not_given", "dict_over_defaults", "none", "rmse_instance", "instance", "empty_dict"],
    )
    def test_section_properties_return_given_or_default(self, kwargs, prop, expected):
        assert getattr(Settings(**kwargs), prop) == expected

    @pytest.mark.parametrize(
        "kwargs, exc, match",
        [
            ({"other": _MySection()}, ValueError, "other"),
            ({"estimation": EvaluatorSettings()}, ValueError, "estimation"),
            ({"bed": _BEDChild()}, ValueError, "bed"),
            ({"my_ranking": {"alpha": 0.3}}, ValueError, "my_ranking"),
            ({"estimation": {"tol": 0}}, ValueError, "estimation.tol"),
            ({"estimation": 5}, TypeError, "estimation"),
            ({"my_ranking": "x"}, TypeError, "my_ranking"),
            ({"estimation": {"maxiter": 5}}, TypeError, "maxiter"),
            ({"my_ranking": _MySection(alpha=1 + 2j)}, TypeError, "complex"),
            ({"evaluator": {"simplify_expr": "yes"}}, TypeError, "evaluator.simplify_expr"),
        ],
        ids=[
            "key_differs_from_name",
            "builtin_key_other_class",
            "builtin_key_subclass",
            "dict_for_custom",
            "dict_out_of_range",
            "not_section_or_dict",
            "custom_not_section",
            "dict_unknown_field",
            "unsaveable_value",
            "dict_wrong_type",
        ],
    )
    def test_constructor_rejects_invalid_sections(self, kwargs, exc, match):
        with pytest.raises(exc, match=match):
            Settings(**kwargs)


class TestSettingsGetReplace:
    @pytest.mark.parametrize(
        "settings, cls, expected",
        [
            (Settings(my_ranking=_MySection(alpha=0.3)), _MySection, _MySection(alpha=0.3)),
            (Settings(), _MySection, _MySection()),
            (Settings(estimation={"max_iter": 5}), EstimationSettings, EstimationSettings(max_iter=5)),
        ],
        ids=["stored_custom", "absent_custom_defaults", "builtin"],
    )
    def test_get_returns_stored_section_or_defaults(self, settings, cls, expected):
        assert settings.get(cls) == expected

    def test_get_rejects_section_of_another_class(self):
        with pytest.raises(TypeError):
            Settings(my_ranking=_MySection(alpha=0.3)).get(_OtherSection)

    @pytest.mark.parametrize(
        "start, kwargs, expected",
        [
            (
                Settings(estimation={"max_iter": 200}),
                {"estimation": {"tol": 1e-8}},
                Settings(estimation={"max_iter": 200, "tol": 1e-8}),
            ),
            (
                Settings(estimation={"max_iter": 200}),
                {"estimation": EstimationSettings(tol=1e-8)},
                Settings(estimation={"tol": 1e-8}),
            ),
            (
                Settings(my_ranking=_MySection(alpha=0.3)),
                {"my_ranking": {"alpha": 0.7}},
                Settings(my_ranking=_MySection(alpha=0.7)),
            ),
            (Settings(), {"my_ranking": _MySection(alpha=0.3)}, Settings(my_ranking=_MySection(alpha=0.3))),
            (
                Settings(estimation={"max_iter": 200}, bed={"num_consts_sampled": 4}),
                {"estimation": {"tol": 1e-8}},
                Settings(estimation={"max_iter": 200, "tol": 1e-8}, bed={"num_consts_sampled": 4}),
            ),
        ],
        ids=["dict_updates_current", "instance_replaces", "custom_dict_updates", "custom_added", "unnamed_kept"],
    )
    def test_replace_updates_or_replaces_sections(self, start, kwargs, expected):
        assert start.replace(**kwargs) == expected

    def test_replace_leaves_self_unchanged(self):
        settings = Settings(estimation={"max_iter": 200}, my_ranking=_MySection(alpha=0.3))
        settings.replace(estimation={"tol": 1e-8}, my_ranking={"alpha": 0.7})
        assert settings.to_dict() == Settings(estimation={"max_iter": 200}, my_ranking=_MySection(alpha=0.3)).to_dict()

    @pytest.mark.parametrize(
        "kwargs, exc, match",
        [
            ({"estimation": None}, TypeError, "estimation"),
            ({"my_ranking": None}, TypeError, "my_ranking"),
            ({"estimation": {"maxiter": 1}}, TypeError, "maxiter"),
            ({"my_ranking": {"alpha": 0.7}}, ValueError, "my_ranking"),
            ({"estimation": {"tol": -1}}, ValueError, "estimation.tol"),
            ({"bed": _BEDChild()}, ValueError, "bed"),
        ],
        ids=["builtin_none", "custom_none", "unknown_field", "dict_for_absent_custom", "out_of_range", "subclass"],
    )
    def test_replace_rejects_invalid(self, kwargs, exc, match):
        with pytest.raises(exc, match=match):
            Settings().replace(**kwargs)


class TestSettingsSaving:
    @pytest.mark.parametrize(
        "settings, expected",
        [
            (Settings(), {"format_version": 1, "sections": {}}),
            (
                Settings(estimation={"max_iter": 200, "tol": 1e-6}, bed={"num_consts_sampled": 4}, rmse=None),
                {"format_version": 1, "sections": {"estimation": {"max_iter": 200}, "bed": {"num_consts_sampled": 4}}},
            ),
        ],
        ids=["defaults", "changed_fields_only"],
    )
    def test_to_dict_saves_only_changed_fields(self, settings, expected):
        assert settings.to_dict() == expected

    def test_to_dict_custom_section_has_import_path(self):
        saved = Settings(my_ranking=_MySection(alpha=0.3)).to_dict()
        assert saved["sections"]["my_ranking"] == {"section_class": _MY_PATH, "alpha": 0.3}

    @pytest.mark.parametrize(
        "settings",
        [
            Settings(),
            Settings(evaluator={"simplify_rules": _RULES, "simplify_expr": True}),
            Settings(estimation={"max_iter": 200, "method": "Nelder-Mead"}),
            Settings(bed={"num_consts_sampled": 2, "num_points_sampled": -1, "bed_gt": _GT}),
            Settings(my_ranking=_MySection(alpha=0.3)),
        ],
        ids=["defaults", "evaluator_rules", "estimation", "bed_gt", "custom"],
    )
    def test_from_dict_round_trips_through_json(self, settings):
        assert Settings.from_dict(json.loads(json.dumps(settings.to_dict()))) == settings

    # Reason: clause ("Copying and pickling go through to_dict / from_dict").
    @pytest.mark.parametrize(
        "copy_fn", [copy.deepcopy, lambda s: pickle.loads(pickle.dumps(s))], ids=["deepcopy", "pickle"]
    )
    def test_settings_copy_and_pickle(self, copy_fn):
        settings = Settings(
            estimation={"max_iter": 200},
            bed={"num_consts_sampled": 2, "bed_gt": _GT},
            my_ranking=_MySection(alpha=0.3),
        )
        copied = copy_fn(settings)
        assert copied == settings and copied is not settings

    def test_from_dict_does_not_mutate_input(self):
        settings = Settings(estimation={"max_iter": 200}, my_ranking=_MySection(alpha=0.3))
        data = settings.to_dict()
        Settings.from_dict(data)
        assert data == settings.to_dict()

    @pytest.mark.parametrize(
        "sections, version, exc, match",
        [
            ({}, 2, ValueError, "format_version"),
            ({"foo": {"alpha": 1}}, 1, ValueError, "foo"),
            ({"other": {"section_class": _MY_PATH}}, 1, ValueError, "other"),
            ({"estimation": {"tol": -1}}, 1, ValueError, "estimation.tol"),
            ({"my_ranking": {"section_class": "no_such_module.X"}}, 1, ImportError, "no_such_module"),
            ({"my_ranking": {"section_class": "collections.OrderedDict"}}, 1, TypeError, "OrderedDict"),
            (
                {"my_ranking": {"section_class": "SRToolkit.utils.settings.SettingsSection"}},
                1,
                TypeError,
                "SettingsSection",
            ),
            ({"estimation": {"maxiter": 1}}, 1, TypeError, "maxiter"),
            (None, 1, TypeError, "sections"),
            ({"estimation": 5}, 1, TypeError, "sections"),
        ],
        ids=[
            "format_version",
            "unknown_builtin",
            "key_differs_from_name",
            "out_of_range",
            "unimportable",
            "not_section_class",
            "base_class",
            "unknown_field",
            "sections_none",
            "section_not_dict",
        ],
    )
    def test_from_dict_rejects_invalid(self, sections, version, exc, match):
        with pytest.raises(exc, match=match):
            Settings.from_dict({"format_version": version, "sections": sections})


@pytest.fixture
def installed_section_bundle(tmp_path, monkeypatch):
    """Install a bundle holding ``bundled_section.BundledSection`` into an isolated store.

    The source module is not importable afterwards, so only a bound path resolves.
    """
    monkeypatch.setattr(_store, "bundles_root", lambda: tmp_path / "store" / "srtk_bundles")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    src = tmp_path / "src" / "bundled_section.py"
    src.parent.mkdir()
    src.write_text(
        "from dataclasses import dataclass\n"
        "from SRToolkit.utils.settings import SettingsSection\n\n"
        "@dataclass(frozen=True, eq=False)\n"
        "class BundledSection(SettingsSection):\n"
        "    name = 'bundled'\n"
        "    alpha: float = 0.5\n"
    )
    out = tmp_path / "bundled-1.0.0.srtk"
    pack(files=[str(src)], out_path=out, name="settings_bundle", version="1.0.0")
    with patch("SRToolkit.bundle._install._confirm", return_value=True):
        install(out)
    return "settings_bundle"


def test_from_dict_binds_bundle_section_classes(installed_section_bundle):
    data = {
        "_bundle": installed_section_bundle,
        "_version": "1.0.0",
        "format_version": 1,
        "sections": {"bundled": {"section_class": "bundled_section.BundledSection", "alpha": 0.3}},
    }
    section = Settings.from_dict(data).to_dict()["sections"]["bundled"]
    assert section["section_class"].startswith("srtk_bundles.")
    assert section["alpha"] == 0.3


class TestSettingsEquality:
    @pytest.mark.parametrize(
        "left, right, expected",
        [
            (Settings(estimation={"max_iter": 100}), Settings(), True),
            (Settings(my_ranking=_MySection()), Settings(), True),
            (Settings(estimation={"max_iter": 5}), Settings(estimation={"max_iter": 6}), False),
            (Settings(my_ranking=_MySection(alpha=0.3)), Settings(my_ranking=_OtherSection(alpha=0.3)), False),
            (Settings(), Settings().to_dict(), False),
        ],
        ids=["given_at_default", "custom_at_default", "diff_value", "diff_custom_class", "not_settings"],
    )
    def test_settings_equality(self, left, right, expected):
        assert (left == right) is expected
