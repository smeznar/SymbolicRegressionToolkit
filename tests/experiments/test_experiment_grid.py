"""Tests for ExperimentGrid serialization (to_dict / from_dict) and the
results_dir anchoring fix in load()."""

import json
import os
import shutil
import warnings
import zipfile

import pytest

from SRToolkit.approaches import ProGED
from SRToolkit.dataset import SR_dataset, data_cache
from SRToolkit.dataset.sampling import UniformSampling
from SRToolkit.experiments import ExperimentGrid
from SRToolkit.experiments.experiment_grid import (
    _annotate_bundles,
    _classify_dataset_source,
    _custom_class_paths,
)


def _make_dataset(name="ds1", benchmark="unit_test", version="1.0.0", max_evaluations=-1):
    """A small, network-free sample-source dataset (no arrays embedded in config)."""
    return SR_dataset.from_samplers(
        ground_truth=["X_0", "*", "X_0"],
        samplers=[UniformSampling(0.5, 5.0, uses_negative=False)],
        n_samples=64,
        seed=1,
        dataset_name=name,
        benchmark=benchmark,
        version=version,
        max_evaluations=max_evaluations,
    )


def _named_approach(name):
    ap = ProGED()
    ap.config.name = name
    return ap


def _make_grid(results_dir, **kwargs):
    defaults = dict(
        datasets=[_make_dataset()],
        approaches=[ProGED()],
        num_experiments=2,
        results_dir=results_dir,
        initial_seed=7,
        top_k=5,
    )
    defaults.update(kwargs)
    return ExperimentGrid(**defaults)


# --------------------------------------------------------------------------- #
# to_dict
# --------------------------------------------------------------------------- #
def test_to_dict_shape(tmp_path):
    grid = _make_grid(str(tmp_path / "run"))
    d = grid.to_dict()

    assert d["format_version"] == 2
    assert d["type"] == "ExperimentGrid"
    assert "srtk_version" in d
    assert d["num_experiments"] == 2
    assert d["initial_seed"] == 7
    assert d["top_k"] == 5
    assert list(d["datasets"]) == ["ds1"]
    assert d["approaches"][0]["approach_class"].endswith(".ProGED")
    assert d["callbacks"] is None


def test_to_dict_is_json_safe_and_carries_no_results_dir(tmp_path):
    grid = _make_grid(str(tmp_path / "run"))
    d = grid.to_dict()
    # Round-trips through JSON unchanged.
    assert json.loads(json.dumps(d)) == d
    # The recipe is location-independent: no results_dir baked in.
    assert "results_dir" not in d
    assert "adapted_states" not in d


# --------------------------------------------------------------------------- #
# from_dict
# --------------------------------------------------------------------------- #
def test_from_dict_roundtrip_in_memory(tmp_path):
    grid = _make_grid(str(tmp_path / "run"))
    rebuilt = ExperimentGrid.from_dict(grid.to_dict(), results_dir=str(tmp_path / "run"))

    assert rebuilt.num_experiments == grid.num_experiments
    assert rebuilt.initial_seed == grid.initial_seed
    assert rebuilt.top_k == grid.top_k
    assert list(rebuilt.datasets) == list(grid.datasets)
    assert rebuilt.approach_configs == grid.approach_configs


def test_from_dict_requires_results_dir_for_dict():
    grid = _make_grid("/tmp/whatever")
    with pytest.raises(ValueError, match="results_dir is required"):
        ExperimentGrid.from_dict(grid.to_dict())


def test_from_dict_anchors_results_dir_to_file_location(tmp_path):
    grid = _make_grid(str(tmp_path / "orig"))
    grid_json = tmp_path / "orig" / "grid_recipe.json"
    grid_json.parent.mkdir(parents=True)
    grid_json.write_text(json.dumps(grid.to_dict()))

    # Move the file elsewhere, then load from the NEW location.
    moved_dir = tmp_path / "moved"
    moved_dir.mkdir()
    moved_json = moved_dir / "grid_recipe.json"
    shutil.move(str(grid_json), str(moved_json))

    rebuilt = ExperimentGrid.from_dict(str(moved_json))
    assert rebuilt.results_dir == str(moved_dir)


def test_from_dict_explicit_results_dir_overrides_file_anchor(tmp_path):
    grid = _make_grid(str(tmp_path / "orig"))
    grid_json = tmp_path / "grid.json"
    grid_json.write_text(json.dumps(grid.to_dict()))

    rebuilt = ExperimentGrid.from_dict(str(grid_json), results_dir=str(tmp_path / "elsewhere"))
    assert rebuilt.results_dir == str(tmp_path / "elsewhere")


def test_from_dict_rejects_wrong_type(tmp_path):
    with pytest.raises(ValueError, match="not an ExperimentGrid recipe"):
        ExperimentGrid.from_dict({"type": "SomethingElse", "format_version": 2}, results_dir=str(tmp_path))


def test_from_dict_rejects_unsupported_format_version(tmp_path):
    with pytest.raises(ValueError, match="Unsupported format_version"):
        ExperimentGrid.from_dict({"type": "ExperimentGrid", "format_version": 1}, results_dir=str(tmp_path))


def test_from_dict_warns_on_missing_class(tmp_path):
    grid = _make_grid(str(tmp_path / "run"))
    d = grid.to_dict()
    d["approaches"][0]["approach_class"] = "nonexistent_module.NotAClass"

    with pytest.warns(UserWarning, match="could not be imported"):
        ExperimentGrid.from_dict(d, results_dir=str(tmp_path / "run"))


def test_from_dict_no_warning_when_all_importable(tmp_path, recwarn):
    grid = _make_grid(str(tmp_path / "run"))
    ExperimentGrid.from_dict(grid.to_dict(), results_dir=str(tmp_path / "run"))
    assert not [w for w in recwarn.list if "could not be imported" in str(w.message)]


# --------------------------------------------------------------------------- #
# create_jobs after from_dict (results land under the anchored dir)
# --------------------------------------------------------------------------- #
def test_jobs_after_from_dict_write_under_anchored_dir(tmp_path):
    grid = _make_grid(str(tmp_path / "orig"))
    grid_json = tmp_path / "run" / "grid.json"
    grid_json.parent.mkdir(parents=True)
    grid_json.write_text(json.dumps(grid.to_dict()))

    rebuilt = ExperimentGrid.from_dict(str(grid_json))
    jobs = rebuilt.create_jobs(skip_completed=False)
    assert jobs, "expected at least one job"
    for job in jobs:
        assert job.result_path.startswith(str(tmp_path / "run"))


# --------------------------------------------------------------------------- #
# add_approach / add_dataset
# --------------------------------------------------------------------------- #
def test_add_approach_appends(tmp_path):
    grid = _make_grid(str(tmp_path / "run"))
    assert [c["name"] for c in grid.approach_configs] == ["ProGED"]
    grid.add_approach(_named_approach("ProGED_B"))
    assert [c["name"] for c in grid.approach_configs] == ["ProGED", "ProGED_B"]


def test_add_approach_records_adaptation_scope(tmp_path):
    grid = _make_grid(str(tmp_path / "run"))
    grid.add_approach(_named_approach("ProGED_B"))
    assert grid.approach_configs[-1]["adaptation_scope"] == "experiment"


def test_add_approach_rejects_duplicate_name(tmp_path):
    grid = _make_grid(str(tmp_path / "run"))
    with pytest.raises(ValueError, match="Duplicate approach name 'ProGED'"):
        grid.add_approach(ProGED())


def test_add_dataset_appends(tmp_path):
    grid = _make_grid(str(tmp_path / "run"))
    grid.add_dataset(_make_dataset(name="ds2"))
    assert list(grid.datasets) == ["ds1", "ds2"]


def test_add_dataset_rejects_duplicate_name(tmp_path):
    grid = _make_grid(str(tmp_path / "run"))
    with pytest.raises(ValueError, match="Duplicate dataset name 'ds1'"):
        grid.add_dataset(_make_dataset(name="ds1"))


def test_add_dataset_rejects_wrong_type(tmp_path):
    grid = _make_grid(str(tmp_path / "run"))
    with pytest.raises(ValueError, match="must be an SR_dataset or SR_benchmark"):
        grid.add_dataset("not a dataset")


def test_extend_loaded_grid(tmp_path):
    """add_* work on a grid reconstructed via from_dict."""
    grid = _make_grid(str(tmp_path / "run"))
    rebuilt = ExperimentGrid.from_dict(grid.to_dict(), results_dir=str(tmp_path / "run"))
    rebuilt.add_dataset(_make_dataset(name="ds2"))
    rebuilt.add_approach(_named_approach("ProGED_B"))
    assert list(rebuilt.datasets) == ["ds1", "ds2"]
    assert [c["name"] for c in rebuilt.approach_configs] == ["ProGED", "ProGED_B"]


def test_constructor_still_rejects_duplicate_dataset(tmp_path):
    """The duplicate-name guard survives the refactor through add_dataset."""
    with pytest.raises(ValueError, match="Duplicate dataset name 'ds1'"):
        _make_grid(str(tmp_path / "run"), datasets=[_make_dataset("ds1"), _make_dataset("ds1")])


# --------------------------------------------------------------------------- #
# build_job
# --------------------------------------------------------------------------- #
def test_build_job_matches_create_jobs(tmp_path):
    grid = _make_grid(str(tmp_path / "run"))
    job = grid.build_job("ds1", "ProGED", grid.initial_seed)
    assert job.dataset_name == "ds1"
    assert job.approach_name == "ProGED"
    assert job.seed == grid.initial_seed
    assert job.result_path == os.path.join(str(tmp_path / "run"), "ds1", "ProGED", "exp_7.json")


def test_build_job_unknown_approach_raises(tmp_path):
    grid = _make_grid(str(tmp_path / "run"))
    with pytest.raises(ValueError, match="No approach named 'Nope'"):
        grid.build_job("ds1", "Nope", 0)


def test_build_job_unknown_dataset_raises(tmp_path):
    grid = _make_grid(str(tmp_path / "run"))
    with pytest.raises(KeyError, match="No dataset named 'nope'"):
        grid.build_job("nope", "ProGED", 0)


# --------------------------------------------------------------------------- #
# save() / load() — single grid.json, results_dir anchoring, adapted_states
# --------------------------------------------------------------------------- #
def test_save_writes_single_grid_json(tmp_path):
    run = tmp_path / "run"
    grid = _make_grid(str(run))
    grid.save()

    assert (run / "grid.json").exists()
    # No scattered spec directories any more.
    assert not (run / "_datasets").exists()
    assert not (run / "_approaches").exists()
    assert not (run / "_callbacks.json").exists()

    d = json.loads((run / "grid.json").read_text())
    assert d["format_version"] == 2
    assert d["type"] == "ExperimentGrid"


def test_save_load_follows_moved_directory(tmp_path):
    grid = _make_grid(str(tmp_path / "orig_run"))
    grid.save()

    moved = tmp_path / "moved_run"
    shutil.move(str(tmp_path / "orig_run"), str(moved))

    loaded = ExperimentGrid.load(str(moved / "grid.json"))
    assert loaded.results_dir == str(moved)
    assert list(loaded.datasets) == ["ds1"]
    jobs = loaded.create_jobs(skip_completed=False)
    assert all(job.result_path.startswith(str(moved)) for job in jobs)


def test_save_load_preserves_adapted_states(tmp_path):
    run = tmp_path / "run"
    grid = _make_grid(str(run), adapted_states={"ProGED": {"ds1": "/tmp/state.pt"}})
    grid.save()

    # adapted_states is local-only: in the shareable recipe it is dropped...
    assert "adapted_states" not in grid.to_dict()
    # ...but save()/load() round-trips it for local persistence.
    loaded = ExperimentGrid.load(str(run / "grid.json"))
    assert loaded._adapted_states == {"ProGED": {"ds1": "/tmp/state.pt"}}


# --------------------------------------------------------------------------- #
# save_commands — grid-driven command lines
# --------------------------------------------------------------------------- #
def test_save_commands_grid_driven(tmp_path):
    run = tmp_path / "run"
    grid = _make_grid(str(run), num_experiments=2)
    cmd_file = tmp_path / "commands.sh"
    grid.save_commands(str(cmd_file), python_executable="python3", skip_completed=False)

    text = cmd_file.read_text()
    grid_json = os.path.join(str(run), "grid.json")
    # One line per (dataset, approach, seed); all reference the single grid.json.
    job_lines = [ln for ln in text.splitlines() if "run_job" in ln]
    assert len(job_lines) == 2  # 1 dataset × 1 approach × 2 seeds
    for ln in job_lines:
        assert grid_json in ln
        assert "--dataset" in ln and "--approach" in ln and "--seed" in ln
    # No scattered per-job files written.
    assert not (run / "_datasets").exists()
    # No once-scope approaches → no prepare file.
    assert not (tmp_path / "prepare_commands.sh").exists()


class _OnceProGED(ProGED):
    """ProGED-backed approach forced to ``adaptation_scope='once'`` for prepare tests."""

    @property
    def adaptation_scope(self):
        return "once"


def _isolate_cache(monkeypatch, tmp_path):
    """Point the data cache at a throwaway dir so materialise tests don't touch the real one."""
    root = tmp_path / "cache"
    monkeypatch.setattr(data_cache, "data_root", lambda: root / "data")
    return root


# --------------------------------------------------------------------------- #
# save_commands — data materialisation + prepare file
# --------------------------------------------------------------------------- #
def test_save_commands_materializes_data(tmp_path, monkeypatch):
    _isolate_cache(monkeypatch, tmp_path)
    grid = _make_grid(str(tmp_path / "run"))
    assert not data_cache.dataset_path("unit_test", "1.0.0", "ds1").exists()

    grid.save_commands(str(tmp_path / "commands.sh"))
    # Data was resolved into the (isolated) cache by save_commands itself.
    assert data_cache.dataset_path("unit_test", "1.0.0", "ds1").exists()


def test_save_commands_skip_materialize(tmp_path, monkeypatch):
    _isolate_cache(monkeypatch, tmp_path)
    grid = _make_grid(str(tmp_path / "run"))
    grid.save_commands(str(tmp_path / "commands.sh"), materialize_data=False)
    assert not data_cache.dataset_path("unit_test", "1.0.0", "ds1").exists()


def test_save_commands_writes_prepare_for_once_scope(tmp_path, monkeypatch):
    _isolate_cache(monkeypatch, tmp_path)
    state = tmp_path / "state.pt"  # absent → adaptation needed
    ap = _OnceProGED()
    ap.config.name = "OnceAp"
    grid = _make_grid(
        str(tmp_path / "run"),
        approaches=[ap],
        adapted_states={"OnceAp": {"ds1": str(state)}},
    )
    cmd_file = tmp_path / "commands.sh"
    prepare_path = grid.save_commands(str(cmd_file), python_executable="python3", skip_completed=False)

    assert prepare_path == str(tmp_path / "prepare_commands.sh")
    prepare_text = open(prepare_path).read()
    adapt_lines = [ln for ln in prepare_text.splitlines() if " adapt " in ln]
    assert len(adapt_lines) == 1
    assert "--dataset ds1" in adapt_lines[0] and "--approach OnceAp" in adapt_lines[0]
    # The experiments file points the user at the prepare file first.
    assert "prepare_commands.sh" in cmd_file.read_text()


def test_save_commands_no_prepare_when_state_exists(tmp_path, monkeypatch):
    _isolate_cache(monkeypatch, tmp_path)
    state = tmp_path / "state.pt"
    state.write_text("x")  # present → no adaptation needed
    ap = _OnceProGED()
    ap.config.name = "OnceAp"
    grid = _make_grid(str(tmp_path / "run"), approaches=[ap], adapted_states={"OnceAp": {"ds1": str(state)}})
    prepare_path = grid.save_commands(str(tmp_path / "commands.sh"))
    assert prepare_path is None
    assert not (tmp_path / "prepare_commands.sh").exists()


def test_shared_state_path_deduped_with_warning(tmp_path):
    """Two datasets sharing one state_path → a single adapt pair, plus an info warning."""
    state = tmp_path / "shared.pt"  # absent → adaptation needed
    ap = _OnceProGED()
    ap.config.name = "OnceAp"
    grid = _make_grid(
        str(tmp_path / "run"),
        datasets=[_make_dataset("ds1"), _make_dataset("ds2")],
        approaches=[ap],
        adapted_states={"OnceAp": {"ds1": str(state), "ds2": str(state)}},
    )

    with pytest.warns(UserWarning, match="share an adapted-state path"):
        pairs = grid._pairs_needing_adaptation()

    # Only the first-listed dataset is returned; ds2 reuses the same state.
    assert pairs == [("OnceAp", "ds1")]


def test_distinct_state_paths_no_warning(tmp_path):
    """Distinct paths per dataset → both pairs returned, no warning."""
    ap = _OnceProGED()
    ap.config.name = "OnceAp"
    grid = _make_grid(
        str(tmp_path / "run"),
        datasets=[_make_dataset("ds1"), _make_dataset("ds2")],
        approaches=[ap],
        adapted_states={"OnceAp": {"ds1": str(tmp_path / "a.pt"), "ds2": str(tmp_path / "b.pt")}},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        pairs = grid._pairs_needing_adaptation()

    assert sorted(pairs) == [("OnceAp", "ds1"), ("OnceAp", "ds2")]


# --------------------------------------------------------------------------- #
# adapt_one — no-op branches and validation
# --------------------------------------------------------------------------- #
def test_adapt_one_noop_for_experiment_scope(tmp_path, monkeypatch):
    _isolate_cache(monkeypatch, tmp_path)
    grid = _make_grid(str(tmp_path / "run"))  # ProGED → "experiment" scope
    grid.adapt_one("ProGED", "ds1")  # returns before reconstructing/adapting; no error


def test_adapt_one_noop_without_state_path(tmp_path):
    ap = _OnceProGED()
    ap.config.name = "OnceAp"
    grid = _make_grid(str(tmp_path / "run"), approaches=[ap])  # no adapted_states entry
    grid.adapt_one("OnceAp", "ds1")  # state_path is None → no-op


def test_adapt_one_unknown_approach_raises(tmp_path):
    grid = _make_grid(str(tmp_path / "run"))
    with pytest.raises(ValueError, match="No approach named 'Nope'"):
        grid.adapt_one("Nope", "ds1")


def test_adapt_one_force_overwrites_existing_state(tmp_path, monkeypatch):
    """Without force an existing state is left alone; with force it is re-adapted."""
    from SRToolkit.experiments import experiment_grid as eg

    state = tmp_path / "state.pt"
    state.write_text("OLD")
    ap = _OnceProGED()
    ap.config.name = "OnceAp"
    grid = _make_grid(str(tmp_path / "run"), approaches=[ap], adapted_states={"OnceAp": {"ds1": str(state)}})

    calls = []

    class _FakeApproach:
        def prepare(self):
            pass

        def adapt(self, X, symbol_library):
            calls.append("adapt")

        def save_adapted_state(self, path):
            with open(path, "w") as fh:
                fh.write("NEW")

    class _FakeDataset:
        X = None
        symbol_library = None

    monkeypatch.setattr(eg.SR_approach, "from_config_dict", staticmethod(lambda cfg: _FakeApproach()))
    monkeypatch.setattr(eg.SR_dataset, "from_dict", staticmethod(lambda d: _FakeDataset()))

    # Default: existing file is a no-op — not re-adapted, contents untouched.
    grid.adapt_one("OnceAp", "ds1")
    assert calls == []
    assert state.read_text() == "OLD"

    # force=True: re-adapts and overwrites.
    grid.adapt_one("OnceAp", "ds1", force=True)
    assert calls == ["adapt"]
    assert state.read_text() == "NEW"


def test_cli_adapt_force_requires_single_pair():
    """--force without --dataset/--approach is rejected before the grid is loaded."""
    import argparse

    from SRToolkit.experiments.__main__ import _cmd_adapt

    args = argparse.Namespace(grid="ignored.json", dataset=None, approach=None, force=True)
    with pytest.raises(SystemExit, match="force requires"):
        _cmd_adapt(args)


# --------------------------------------------------------------------------- #
# End-to-end: CLI run_job reconstructs the job from grid.json alone
# --------------------------------------------------------------------------- #
def test_cli_run_job_end_to_end(tmp_path):
    import subprocess
    import sys

    run = tmp_path / "run"
    grid = _make_grid(str(run), datasets=[_make_dataset(max_evaluations=200)])
    grid.save()
    grid_json = run / "grid.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "SRToolkit.experiments",
            "run_job",
            "--grid",
            str(grid_json),
            "--dataset",
            "ds1",
            "--approach",
            "ProGED",
            "--seed",
            "7",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (run / "ds1" / "ProGED" / "exp_7.json").exists()


def test_cli_adapt_requires_both_dataset_and_approach(tmp_path):
    import subprocess
    import sys

    run = tmp_path / "run"
    _make_grid(str(run)).save()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "SRToolkit.experiments",
            "adapt",
            "--grid",
            str(run / "grid.json"),
            "--dataset",
            "ds1",
        ],  # --approach omitted
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "must be given together" in (result.stderr + result.stdout)


def test_cli_progress(tmp_path):
    import subprocess
    import sys

    run = tmp_path / "run"
    grid = _make_grid(str(run), num_experiments=2)
    grid.save()
    # Fabricate one completed result so the table shows 1/2.
    res = run / "ds1" / "ProGED" / "exp_7.json"
    res.parent.mkdir(parents=True)
    res.write_text("{}")

    result = subprocess.run(
        [sys.executable, "-m", "SRToolkit.experiments", "progress", "--grid", str(run / "grid.json")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ds1" in result.stdout and "ProGED" in result.stdout
    assert "1/2" in result.stdout


# --------------------------------------------------------------------------- #
# export / from_export
# --------------------------------------------------------------------------- #
def test_classify_dataset_source():
    assert _classify_dataset_source({"data_source": None})[0] is True
    assert _classify_dataset_source({"data_source": {"source_class": "x.SampleSource", "seed": None}})[0] is True
    assert _classify_dataset_source({"data_source": {"source_class": "x.SampleSource", "seed": 4}})[0] is False
    assert _classify_dataset_source({"data_source": {"source_class": "x.UrlSource", "url": "u"}})[0] is False
    # A fallback chain is downloadable → referenced, not archived.
    assert _classify_dataset_source({"data_source": {"source_class": "x.FallbackSource", "sources": []}})[0] is False


def test_custom_class_paths():
    cfgs = [
        {"approach_class": "SRToolkit.approaches.ProGED.ProGED"},
        {"sampler_class": "mypkg.Foo", "nested": {"source_class": "SRToolkit.X"}},
    ]
    assert _custom_class_paths(cfgs) == ["mypkg.Foo"]


def _export_grid(results_dir, benchmark):
    """Grid with one archived (seedless) and one referenced (seeded) dataset."""
    archived = SR_dataset.from_samplers(
        ground_truth=["X_0", "*", "X_0"],
        samplers=[UniformSampling(0.5, 5.0, uses_negative=False)],
        n_samples=16,
        seed=None,
        dataset_name="vol",
        benchmark=benchmark,
        version="1.0.0",
    )
    referenced = SR_dataset.from_samplers(
        ground_truth=["X_0", "+", "X_0"],
        samplers=[UniformSampling(0.5, 5.0, uses_negative=False)],
        n_samples=16,
        seed=5,
        dataset_name="rep",
        benchmark=benchmark,
        version="1.0.0",
    )
    return ExperimentGrid([archived, referenced], [ProGED()], num_experiments=1, results_dir=results_dir)


def test_export_writes_zip_and_archives_only_unreachable(tmp_path):
    benchmark = "exp_test_writes"
    try:
        grid = _export_grid(str(tmp_path / "run"), benchmark)
        out = tmp_path / "export.zip"
        grid.export(str(out))

        assert out.is_file()
        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
            manifest = zf.read("MANIFEST.md").decode()
        assert "grid.json" in names
        assert "MANIFEST.md" in names
        # Only the seedless dataset is archived; the seeded one is referenced.
        assert "data/vol.zip" in names
        assert "data/rep.zip" not in names

        assert "`vol` — **archived**" in manifest
        assert "`rep` — referenced" in manifest
        assert "None — every referenced class is built into SRToolkit." in manifest
    finally:
        data_cache.remove(benchmark)


def test_export_warns_on_non_zip_suffix(tmp_path):
    benchmark = "exp_test_suffix"
    try:
        grid = _export_grid(str(tmp_path / "run"), benchmark)
        out = tmp_path / "export.tar"
        with pytest.warns(UserWarning, match="does not end in '.zip'"):
            grid.export(str(out))
        assert zipfile.is_zipfile(out)  # a zip is written regardless of the suffix
    finally:
        data_cache.remove(benchmark)


def test_export_from_export_roundtrip_on_wiped_cache(tmp_path):
    benchmark = "exp_test_roundtrip"
    try:
        grid = _export_grid(str(tmp_path / "run"), benchmark)
        out = tmp_path / "export.zip"
        grid.export(str(out))

        # Wipe the cache: archived data must come from the zip, referenced from its seed.
        data_cache.remove(benchmark)

        reloaded = ExperimentGrid.from_export(str(out), results_dir=str(tmp_path / "recipient"))
        assert list(reloaded.datasets) == ["vol", "rep"]
        assert reloaded.results_dir == str(tmp_path / "recipient")
        assert SR_dataset.from_dict(reloaded.datasets["vol"]).X.shape == (16, 1)
        assert SR_dataset.from_dict(reloaded.datasets["rep"]).X.shape == (16, 1)
    finally:
        data_cache.remove(benchmark)


def test_from_export_defaults_results_dir_to_zip_stem(tmp_path):
    benchmark = "exp_test_default_rd"
    try:
        grid = _export_grid(str(tmp_path / "run"), benchmark)
        out = tmp_path / "export.zip"
        grid.export(str(out))
        reloaded = ExperimentGrid.from_export(str(out))
        # Defaults to the archive path minus '.zip'.
        assert reloaded.results_dir == os.path.abspath(str(tmp_path / "export"))
    finally:
        data_cache.remove(benchmark)


def test_from_export_missing_archive_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No export archive"):
        ExperimentGrid.from_export(str(tmp_path / "nope.zip"))


def test_from_export_archive_without_grid_json_raises(tmp_path):
    bogus = tmp_path / "bogus.zip"
    with zipfile.ZipFile(bogus, "w") as zf:
        zf.writestr("readme.txt", "not an export")
    with pytest.raises(FileNotFoundError, match="no grid.json"):
        ExperimentGrid.from_export(str(bogus))


# --------------------------------------------------------------------------- #
# Bundle annotation (_annotate_bundles) — granularity & multi-bundle handling
# --------------------------------------------------------------------------- #
def test_annotate_bundles_stamps_innermost_dicts():
    # A dataset-shaped config: sampler from bundle A, source from bundle B.
    cfg = {
        "dataset_name": "ds",
        "samplers": [{"sampler_class": "a.Sampler", "lo": 0}],
        "data_source": {"source_class": "b.Source", "url": "x"},
    }
    mapping = {"a.Sampler": ("bundle_a", "1.0.0"), "b.Source": ("bundle_b", "2.0.0")}
    out = _annotate_bundles(cfg, mapping)

    # The enclosing dataset dict holds no *_class directly → not annotated.
    assert "_bundle" not in out
    # Each nested dict binds to its own bundle.
    assert out["samplers"][0]["_bundle"] == "bundle_a" and out["samplers"][0]["_version"] == "1.0.0"
    assert out["data_source"]["_bundle"] == "bundle_b" and out["data_source"]["_version"] == "2.0.0"


def test_annotate_bundles_single_class_dict():
    cfg = {"approach_class": "pkg.Approach", "hp": 1}
    out = _annotate_bundles(cfg, {"pkg.Approach": ("appbundle", "0.1.0")})
    assert out["_bundle"] == "appbundle" and out["_version"] == "0.1.0"


def test_annotate_bundles_multi_bundle_one_dict_warns():
    cfg = {"a_class": "x.A", "b_class": "y.B"}
    mapping = {"x.A": ("bundle_x", "1.0.0"), "y.B": ("bundle_y", "1.0.0")}
    with pytest.warns(UserWarning, match="multiple bundles"):
        out = _annotate_bundles(cfg, mapping)
    assert "_bundle" not in out  # left unannotated — handle manually


def test_annotate_bundles_leaves_builtin_alone():
    cfg = {"approach_class": "SRToolkit.approaches.ProGED.ProGED"}
    out = _annotate_bundles(cfg, {})  # nothing custom
    assert "_bundle" not in out


# --------------------------------------------------------------------------- #
# export packs custom code + annotates; full install round-trip
# --------------------------------------------------------------------------- #
@pytest.fixture
def custom_approach(tmp_path_factory):
    """Create an importable module with a custom SR_approach; clean up afterward."""
    import sys

    mod_dir = tmp_path_factory.mktemp("custom_mod")
    (mod_dir / "grid_custom_ap.py").write_text(
        "from SRToolkit.approaches import ApproachConfig, SR_approach\n\n"
        "class CustomAp(SR_approach):\n"
        "    def __init__(self):\n"
        "        super().__init__(ApproachConfig(name='CustomAp'))\n"
        "    def prepare(self):\n"
        "        pass\n"
        "    def search(self, sr_evaluator, seed=None):\n"
        "        pass\n"
        "    @classmethod\n"
        "    def from_config(cls, config):\n"
        "        return cls()\n"
    )
    sys.path.insert(0, str(mod_dir))
    import grid_custom_ap

    yield grid_custom_ap.CustomAp
    # Some tests remove this path themselves to simulate a fresh machine.
    if str(mod_dir) in sys.path:
        sys.path.remove(str(mod_dir))
    sys.modules.pop("grid_custom_ap", None)


def _pack_custom(cls, tmp_path, name="custom_ap_bundle", version="1.2.0"):
    """Pack the source file defining ``cls`` into a ``.srtk`` and return its path."""
    import inspect

    from SRToolkit.bundle import pack

    out = tmp_path / f"{name}.srtk"
    pack(files=[inspect.getsourcefile(cls)], out_path=out, name=name, version=version)
    return out


def test_export_packs_and_annotates_custom_class(tmp_path, custom_approach):
    grid = ExperimentGrid([_make_dataset()], [custom_approach()], num_experiments=1, results_dir=str(tmp_path / "run"))
    srtk = _pack_custom(custom_approach, tmp_path)
    out = tmp_path / "export.zip"
    grid.export(str(out), additional_bundles=[srtk])

    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        recipe = json.loads(zf.read("grid.json"))
        manifest = zf.read("MANIFEST.md").decode()
    assert "bundles/custom_ap_bundle.srtk" in names
    ap_cfg = recipe["approaches"][0]
    # Path is unchanged; resolution is deferred to the _bundle annotation on load.
    assert ap_cfg["approach_class"] == "grid_custom_ap.CustomAp"
    assert ap_cfg["_bundle"] == "custom_ap_bundle" and ap_cfg["_version"] == "1.2.0"

    assert "install" in manifest.lower()
    assert "custom_ap_bundle.srtk" in manifest


def test_export_warns_on_uncovered_custom_class(tmp_path, custom_approach):
    """No installed bundle, no additional_bundles → warn naming the class and its file."""
    grid = ExperimentGrid([_make_dataset()], [custom_approach()], num_experiments=1, results_dir=str(tmp_path / "run"))
    out = tmp_path / "export.zip"
    with pytest.warns(UserWarning, match="grid_custom_ap.CustomAp"):
        grid.export(str(out))
    with zipfile.ZipFile(out) as zf:
        manifest = zf.read("MANIFEST.md").decode()
        names = set(zf.namelist())
    # Flagged for manual handling; nothing was packed for it.
    assert "grid_custom_ap.CustomAp" in manifest
    assert not any(n.startswith("bundles/") for n in names)


def test_export_strict_raises_on_uncovered_custom_class(tmp_path, custom_approach):
    grid = ExperimentGrid([_make_dataset()], [custom_approach()], num_experiments=1, results_dir=str(tmp_path / "run"))
    out = tmp_path / "export.zip"
    with pytest.raises(ValueError, match="grid_custom_ap.CustomAp"):
        grid.export(str(out), strict=True)


def test_export_warns_on_unused_additional_bundle(tmp_path, custom_approach):
    """A supplied bundle that matches no referenced class warns."""
    grid = ExperimentGrid([_make_dataset()], [custom_approach()], num_experiments=1, results_dir=str(tmp_path / "run"))
    # Pack the right code, but the grid still references it, so this matches; add a decoy too.
    srtk = _pack_custom(custom_approach, tmp_path)
    decoy = tmp_path / "decoy.py"
    decoy.write_text("class Unrelated:\n    pass\n")
    from SRToolkit.bundle import pack

    decoy_srtk = tmp_path / "decoy.srtk"
    pack(files=[decoy], out_path=decoy_srtk, name="decoy", version="0.1.0")
    out = tmp_path / "export.zip"
    with pytest.warns(UserWarning, match="did not match"):
        grid.export(str(out), additional_bundles=[srtk, decoy_srtk])


def test_export_install_from_export_roundtrip(tmp_path, custom_approach, monkeypatch):
    import sys
    from unittest.mock import patch

    from SRToolkit.approaches.sr_approach import SR_approach
    from SRToolkit.bundle import _store, install

    grid = ExperimentGrid([_make_dataset()], [custom_approach()], num_experiments=1, results_dir=str(tmp_path / "run"))
    srtk = _pack_custom(custom_approach, tmp_path)
    out = tmp_path / "export.zip"
    grid.export(str(out), additional_bundles=[srtk])

    # Isolated bundle store so we don't touch the user's real one.
    bundle_root = tmp_path / "store" / "srtk_bundles"
    monkeypatch.setattr(_store, "bundles_root", lambda: bundle_root)

    # from_export extracts bundles/ into results_dir so they can be installed afterwards.
    recipient = tmp_path / "recipient"
    reloaded = ExperimentGrid.from_export(str(out), results_dir=str(recipient))
    assert (recipient / "bundles" / "custom_ap_bundle.srtk").is_file()
    assert (recipient / "MANIFEST.md").is_file()

    with patch("SRToolkit.bundle._install._confirm", return_value=True):
        install(str(recipient / "bundles" / "custom_ap_bundle.srtk"))

    # Drop the original loose module: the installed bundle is now the only source.
    sys.path[:] = [p for p in sys.path if "custom_mod" not in p]
    sys.modules.pop("grid_custom_ap", None)

    # Binding is deferred to run time, so reconstructing now resolves to the bundle.
    approach = SR_approach.from_config_dict(reloaded.approach_configs[0])
    assert type(approach).__name__ == "CustomAp"
    assert type(approach).__module__.startswith("srtk_bundles.")


def test_from_export_unresolved_bundle_soft_warns(tmp_path, custom_approach):
    """If the bundle isn't installed, from_export warns instead of crashing."""
    grid = ExperimentGrid([_make_dataset()], [custom_approach()], num_experiments=1, results_dir=str(tmp_path / "run"))
    srtk = _pack_custom(custom_approach, tmp_path)
    out = tmp_path / "export.zip"
    grid.export(str(out), additional_bundles=[srtk])

    # Drop the module so the class is unresolvable and the bundle is not installed.
    import sys

    sys.path[:] = [p for p in sys.path if "custom_mod" not in p]
    sys.modules.pop("grid_custom_ap", None)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        reloaded = ExperimentGrid.from_export(str(out), results_dir=str(tmp_path / "recipient"))
    # Loads without raising; warns about the missing bundle/class.
    assert list(reloaded.datasets) == ["ds1"]
    assert any("bundle" in str(x.message).lower() or "imported" in str(x.message).lower() for x in w)


def test_export_include_results_roundtrips(tmp_path):
    benchmark = "exp_test_results"
    try:
        run = tmp_path / "run"
        grid = _export_grid(str(run), benchmark)
        # Fabricate a completed result file.
        res = run / "vol" / "ProGED" / "exp_0.json"
        res.parent.mkdir(parents=True)
        res.write_text("{}")

        out = tmp_path / "export.zip"
        grid.export(str(out), include_results=True)
        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
            manifest = zf.read("MANIFEST.md").decode()
        assert "results/vol/ProGED/exp_0.json" in names
        assert "Included in `results/`." in manifest

        data_cache.remove(benchmark)
        ExperimentGrid.from_export(str(out), results_dir=str(tmp_path / "recipient"))
        # Shipped results are copied into the recipient's results_dir.
        assert (tmp_path / "recipient" / "vol" / "ProGED" / "exp_0.json").exists()
    finally:
        data_cache.remove(benchmark)
