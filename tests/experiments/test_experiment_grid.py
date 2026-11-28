"""Tests for ExperimentGrid serialization (to_dict / from_dict) and the
results_dir anchoring fix in load()."""

import contextlib
import fcntl
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pytest
from strategies import (
    EventLog,
    ScriptedApproach,
    SignalAfter,
    SteppedApproach,
    result_answers,
)

from SRToolkit.approaches import Approach, ApproachConfig, ProGED
from SRToolkit.dataset import Dataset, data_cache
from SRToolkit.dataset.sampling import UniformSampling
from SRToolkit.evaluation.results import Results
from SRToolkit.experiments import ExperimentGrid, ExperimentInfo, ExperimentJob
from SRToolkit.utils.expression_constraints import MaxConstants, MaxTokens, MaxTreeDepth


def _make_dataset(name="ds1", benchmark="unit_test", version="1.0.0", max_evaluations=-1):
    """A small, network-free sample-source dataset (no arrays embedded in config)."""
    return Dataset.from_samplers(
        ground_truth=["X_0", "*", "X_0"],
        samplers=[UniformSampling(0.5, 5.0)],
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


@pytest.mark.parametrize(
    "given, expected",
    [
        (None, lambda tmp: {"ProGED": {"ds1": "/from/file.pt"}}),
        ({"ProGED": {"ds1": "/given.pt"}}, lambda tmp: {"ProGED": {"ds1": "/given.pt"}}),
        ({"ProGED": {"ds1": "rel/given.pt"}}, lambda tmp: {"ProGED": {"ds1": str(tmp / "rel" / "given.pt")}}),
    ],
    ids=["read_from_file", "explicit_argument_wins", "relative_made_absolute"],
)
def test_from_dict_path_restores_adapted_states(tmp_path, monkeypatch, given, expected):
    monkeypatch.chdir(tmp_path)
    grid_json = tmp_path / "run" / "grid.json"
    grid_json.parent.mkdir()
    cfg = _make_grid(str(tmp_path / "run")).to_dict()
    cfg["adapted_states"] = {"ProGED": {"ds1": "/from/file.pt"}}
    grid_json.write_text(json.dumps(cfg))

    loaded = ExperimentGrid.from_dict(str(grid_json), adapted_states=given)

    assert loaded._adapted_states == expected(tmp_path)


def test_from_dict_explicit_results_dir_overrides_file_anchor(tmp_path):
    grid = _make_grid(str(tmp_path / "orig"))
    grid_json = tmp_path / "grid.json"
    grid_json.write_text(json.dumps(grid.to_dict()))

    rebuilt = ExperimentGrid.from_dict(str(grid_json), results_dir=str(tmp_path / "elsewhere"))
    assert rebuilt.results_dir == str(tmp_path / "elsewhere")


def test_from_dict_rejects_zip_path():
    with pytest.raises(ValueError, match="share.load"):
        ExperimentGrid.from_dict("run.zip")


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
    with pytest.raises(ValueError, match="must be a Dataset or Benchmark"):
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
# checkpoint() — single grid.json
# --------------------------------------------------------------------------- #
def test_checkpoint_writes_single_grid_json(tmp_path):
    run = tmp_path / "run"
    grid = _make_grid(str(run))
    grid.checkpoint()

    assert (run / "grid.json").exists()
    # No scattered spec directories any more.
    assert not (run / "_datasets").exists()
    assert not (run / "_approaches").exists()
    assert not (run / "_callbacks.json").exists()

    d = json.loads((run / "grid.json").read_text())
    assert d["format_version"] == 2
    assert d["type"] == "ExperimentGrid"


def test_checkpoint_round_trips_through_from_dict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "new" / "run"
    grid = _make_grid(str(run), adapted_states={"ProGED": {"ds1": "states/ds1.pt"}})
    grid.checkpoint()
    grid.add_approach(_named_approach("Second"))
    grid.checkpoint()  # overwrites the first grid.json

    loaded = ExperimentGrid.from_dict(str(run / "grid.json"))

    assert loaded.to_dict() == grid.to_dict()
    assert loaded._adapted_states == {"ProGED": {"ds1": str(tmp_path / "states" / "ds1.pt")}}


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


def test_adapt_one_force_readapts_and_passes_task(tmp_path, monkeypatch):
    """Without force an existing state is left alone; with force adapt gets the dataset's task."""
    _isolate_cache(monkeypatch, tmp_path)
    state = tmp_path / "state"
    state.write_text("OLD")
    ds = _task_dataset()
    grid = _make_grid(
        str(tmp_path / "run"),
        datasets=[ds],
        approaches=[_GridRecorder("once")],
        adapted_states={"recorder": {"ds1": str(state)}},
    )

    grid.adapt_one("recorder", "ds1")
    assert state.read_text() == "OLD"

    grid.adapt_one("recorder", "ds1", force=True)
    record = json.loads(state.read_text())
    assert record["X_shape"] == list(ds.X.shape)
    assert record["X_sum"] == pytest.approx(float(ds.X.sum()))
    assert record["background_knowledge"] == json.loads(json.dumps(ds.background_knowledge.to_dict()))
    assert record["required"] == [c.to_dict() for c in _TASK_REQUIRED]
    assert record["checked"] == [True, True, True]


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
    grid.checkpoint()
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
    _make_grid(str(run)).checkpoint()
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
    grid.checkpoint()
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
    lines = result.stdout.splitlines()
    assert (lines[0].split(), lines[2].split()) == (["Dataset", "ProGED"], ["ds1", "1/2"])
    assert lines[3:] == []  # nothing running, interrupted or crashed: no summary


# --------------------------------------------------------------------------- #
# What adapt receives (stage D)
# --------------------------------------------------------------------------- #
_TASK_REQUIRED = [MaxConstants(8), MaxTokens(20), MaxTreeDepth(5)]


def _task_dataset():
    """Grid dataset whose required list is _TASK_REQUIRED, with a hint and a description."""
    return Dataset.from_samplers(
        ground_truth=["X_0", "*", "X_0"],
        samplers=[UniformSampling(0.5, 5.0)],
        n_samples=16,
        seed=1,
        dataset_name="ds1",
        benchmark="unit_test",
        version="1.0.0",
        max_evaluations=5,
        required_constraints=[MaxTreeDepth(5)],
        max_expr_length=20,
        advisory_constraints=[MaxTreeDepth(3)],
        description="Toy task",
    )


@dataclass
class _RecorderConfig(ApproachConfig):
    name: str = "recorder"
    scope: str = "experiment"


class _GridRecorder(Approach):
    """Logs calls; its saved state is a JSON record of what the last ``adapt`` received."""

    def __init__(self, scope="experiment"):
        super().__init__(_RecorderConfig(scope=scope))
        self.events = []
        self.adapt_args = []
        self.loaded_from = []

    @property
    def adaptation_scope(self):
        return self.config.scope

    def adapt(self, X, background_knowledge, required_constraints):
        self.events.append("adapt")
        required = tuple(required_constraints)
        checked = [c.pre_check(["X_0"]).passed for c in required]  # raises unless set up
        self.adapt_args.append((X, background_knowledge, required, checked))

    def search(self, evaluator, seed=None):
        self.events.append("search")
        evaluator.evaluate_expr(["X_0"])

    def save_adapted_state(self, path):
        self.events.append("save")
        X, bk, required, checked = self.adapt_args[-1]
        record = {
            "X_shape": list(X.shape),
            "X_sum": float(X.sum()),
            "background_knowledge": bk.to_dict(),
            "required": [c.to_dict() for c in required],
            "checked": checked,
        }
        with open(path, "w") as fh:
            json.dump(record, fh)

    def load_adapted_state(self, path):
        self.events.append("load")
        self.loaded_from.append(path)

    @classmethod
    def from_config(cls, config):
        return cls(config["scope"])


@pytest.mark.parametrize("scope", ["experiment", "once"])
def test_job_run_passes_task_to_adapt(tmp_path, monkeypatch, scope):
    _isolate_cache(monkeypatch, tmp_path)
    ds = _task_dataset()
    approach = _GridRecorder(scope)
    ExperimentJob(ds, approach, ExperimentInfo(seed=0, result_path=str(tmp_path / "r.json"))).run()
    [(X, bk, required, checked)] = approach.adapt_args
    assert X.tolist() == ds.X.tolist()
    assert bk is ds.background_knowledge
    assert [c.to_dict() for c in required] == [c.to_dict() for c in _TASK_REQUIRED]
    assert checked == [True, True, True]
    assert len(required) == len(ds.required_constraints)
    assert all(a is b for a, b in zip(required, ds.required_constraints))


def _run_once_job(tmp_path, seed, state_path):
    approach = _GridRecorder("once")
    info = ExperimentInfo(seed=seed, result_path=str(tmp_path / f"r{seed}.json"), adapted_state_path=state_path)
    ExperimentJob(_task_dataset(), approach, info).run()
    return approach


@pytest.mark.parametrize(
    "with_path, expected",
    [(False, (1, 1, 0)), (True, (1, 0, 1))],
    ids=["without-path-adapts-every-job", "with-path-loads-saved-state"],
)
def test_job_run_once_reuses_state_only_with_path(tmp_path, monkeypatch, with_path, expected):
    _isolate_cache(monkeypatch, tmp_path)
    # caller wiring: "once" goes through adapt_or_load with the job's state path (pinned in test_sr_dataset)
    state = tmp_path / "state"
    path = str(state) if with_path else None
    first = _run_once_job(tmp_path, 0, path)
    second = _run_once_job(tmp_path, 1, path)
    assert (first.events.count("adapt"), second.events.count("adapt"), len(second.loaded_from)) == expected
    assert second.loaded_from == [path] * expected[2]
    assert state.exists() is with_path


# --------------------------------------------------------------------------- #
# Checkpoints of jobs (stage 2b)
# --------------------------------------------------------------------------- #
def _journal_mode(path):
    with sqlite3.connect(path) as con:
        return con.execute("PRAGMA journal_mode").fetchone()[0]


def _job(tmp_path, approach, dataset=None, keep_checkpoint=False, name="r.json", callbacks=None, **info_kwargs):
    info = ExperimentInfo(seed=7, result_path=str(tmp_path / name), keep_checkpoint=keep_checkpoint, **info_kwargs)
    return ExperimentJob(dataset or _make_dataset(), approach, info, callbacks=callbacks)


def _result(path):
    return {**Results.load(str(path))[0].to_dict(), "wall_time": None, "limits": None}


@pytest.mark.parametrize(
    "d, expected",
    [
        (
            ExperimentInfo(
                seed=1, result_path="/r.json", keep_checkpoint=True, snapshot_interval=None, keep_all_models=False
            ).to_dict(),
            (True, None, False),
        ),
        (
            {"seed": 1, "result_path": "/r.json", "top_k": 20, "adapted_state_path": None},
            (False, 300.0, True),
        ),  # an older dict
    ],
    ids=["round-trip", "without-keys"],
)
def test_experiment_info_checkpoint_settings_round_trip(d, expected):
    info = ExperimentInfo.from_dict(d)
    keys = ("keep_checkpoint", "snapshot_interval", "keep_all_models")
    assert (info.keep_checkpoint, info.snapshot_interval, info.keep_all_models) == expected
    assert tuple(info.to_dict()[k] for k in keys) == expected
    assert ExperimentInfo.from_dict(info.to_dict()) == info


def test_job_passes_snapshot_interval(tmp_path):
    approach = SteppedApproach(stop_after_asks=1)
    _job(tmp_path, approach, snapshot_interval=60.0).run()
    assert approach.evaluator.snapshot_interval == 60.0


@pytest.mark.parametrize("as_dir, expected", [(False, "r.sqlite"), (True, "exp_7.sqlite")], ids=["file", "directory"])
def test_checkpoint_path_naming(tmp_path, monkeypatch, as_dir, expected):
    _isolate_cache(monkeypatch, tmp_path)
    info = ExperimentInfo(seed=7, result_path=str(tmp_path) if as_dir else str(tmp_path / "r.json"))
    job = ExperimentJob(_make_dataset(), ScriptedApproach(), info)
    assert os.fspath(job.checkpoint_path) == str(tmp_path / expected)


class TestJobCheckpoints:
    @pytest.mark.parametrize(
        "keep, files, journal",
        [(False, ["r.json"], None), (True, ["r.json", "r.sqlite"], "delete")],
        ids=["delete", "keep"],
    )
    def test_finished_job_deletes_or_compacts_checkpoint(self, tmp_path, monkeypatch, keep, files, journal):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        job = _job(tmp_path / "out", ScriptedApproach(), keep_checkpoint=keep)
        os.makedirs(tmp_path / "out")
        job.run()
        assert sorted(os.listdir(tmp_path / "out")) == files  # compacted: no -wal or -shm
        checkpoint = tmp_path / "out" / "r.sqlite"
        assert (_journal_mode(checkpoint) if checkpoint.exists() else None) == journal

    def test_failed_result_save_keeps_checkpoint(self, tmp_path, monkeypatch):
        _isolate_cache(monkeypatch, tmp_path / "cache")

        def disk_full(*args, **kwargs):
            raise OSError("No space left on device")

        job = _job(tmp_path, ScriptedApproach())
        monkeypatch.setattr(os, "replace", disk_full)  # the last step of an atomic save
        monkeypatch.setattr(os, "rename", disk_full)
        with pytest.raises(OSError, match="No space"):
            job.run()
        assert (job.is_complete, (tmp_path / "r.sqlite").exists()) == (False, True)

    def test_interrupted_job_keeps_checkpoint_and_raises(self, tmp_path, monkeypatch):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        EventLog.ended.clear()
        EventLog.references.clear()
        job = _job(tmp_path, ScriptedApproach(kill_seed=7), callbacks=EventLog())
        with pytest.raises(KeyboardInterrupt):
            job.run()
        assert EventLog.ended == ["interrupted"]
        assert [r["path"] for r in EventLog.references] == [str(tmp_path / "r.sqlite")]  # the file remains
        assert not job.is_complete  # no result
        assert (tmp_path / "r.lock").exists()  # deleted only once a result is saved
        assert not (tmp_path / "r.sqlite-wal").exists()  # closed
        saved = Results.from_checkpoint(tmp_path / "r.sqlite")[0]
        assert (saved.trace, saved.stop_reason) == ([0, 1, 0], "interrupted")

    def test_resume_of_interrupted_job_equals_uninterrupted_job(self, tmp_path, monkeypatch):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        with pytest.raises(KeyboardInterrupt):  # in the second step: 4 calls and a snapshot are saved
            _job(tmp_path, SteppedApproach(stop_after_asks=3), callbacks=SignalAfter(3)).run()
        EventLog.events.clear()
        _job(tmp_path, SteppedApproach(stop_after_asks=3), callbacks=EventLog()).run()
        assert len(EventLog.events) == 2  # continued from the snapshot
        _job(tmp_path, SteppedApproach(stop_after_asks=3), name="straight.json").run()
        assert _result(tmp_path / "r.json") == _result(tmp_path / "straight.json")
        assert not (tmp_path / "r.sqlite").exists()

    def test_resume_of_search_only_job_restarts_with_warning(self, tmp_path, monkeypatch):
        # S4: an approach that cannot continue from a snapshot cannot resume; its job starts afresh.
        _isolate_cache(monkeypatch, tmp_path / "cache")
        with pytest.raises(KeyboardInterrupt):
            _job(tmp_path, ScriptedApproach(kill_seed=7)).run()
        with pytest.warns(UserWarning, match="snapshot"):
            _job(tmp_path, ScriptedApproach()).run()
        _job(tmp_path, ScriptedApproach(), name="straight.json").run()
        assert _result(tmp_path / "r.json") == _result(tmp_path / "straight.json")
        assert not (tmp_path / "r.sqlite").exists()

    def test_job_search_exception_keeps_checkpoint(self, tmp_path, monkeypatch):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        job = _job(tmp_path, ScriptedApproach(raise_seed=7))
        with pytest.raises(RuntimeError, match="boom"):
            job.run()
        assert (job.is_complete, (tmp_path / "r.sqlite").exists(), (tmp_path / "r.lock").exists()) == (
            False,
            True,
            True,
        )

    def test_job_resume_with_other_settings_raises(self, tmp_path, monkeypatch):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        with pytest.raises(KeyboardInterrupt):
            _job(tmp_path, SteppedApproach(), callbacks=SignalAfter(3)).run()
        other = Dataset.from_samplers(
            ground_truth=["X_0", "+", "X_0"],
            samplers=[UniformSampling(0.5, 5.0)],
            n_samples=64,
            seed=1,
            dataset_name="ds1",
            benchmark="unit_test",
            version="1.0.0",
        )
        with pytest.raises(ValueError, match=r"\by\b"):
            _job(tmp_path, SteppedApproach(), dataset=other).run()

    @pytest.mark.parametrize("keep_all_models", [True, False], ids=["full", "compact"])
    def test_finished_checkpoint_without_result_gives_the_result(self, tmp_path, monkeypatch, keep_all_models):
        # E.g. killed while saving its result: the result comes from the checkpoint and nothing runs,
        # also for an approach that cannot continue from a snapshot. The checkpoint is then deleted,
        # so the result has no reference, as one from an uninterrupted job.
        _isolate_cache(monkeypatch, tmp_path / "cache")
        _job(tmp_path, ScriptedApproach(), keep_checkpoint=True).run()
        os.remove(tmp_path / "r.json")
        approach = ScriptedApproach()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _job(tmp_path, approach, keep_all_models=keep_all_models).run()
        assert not [w for w in caught if "snapshot" in str(w.message)]  # not restarted
        _job(tmp_path, ScriptedApproach(), name="straight.json", keep_all_models=keep_all_models).run()
        assert (approach.seeds, _result(tmp_path / "r.json")) == ([], _result(tmp_path / "straight.json"))
        saved = Results.load(str(tmp_path / "r.json"))[0]
        assert ((saved.summary is None), saved.checkpoint) == (keep_all_models, None)
        assert not (tmp_path / "r.sqlite").exists()


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["sigint", "sigterm"])
def test_cli_run_job_interrupted_exits_130(tmp_path, monkeypatch, capsys, sig):
    from SRToolkit.experiments.__main__ import main

    _isolate_cache(monkeypatch, tmp_path / "cache")
    run = tmp_path / "run"
    grid = _make_grid(str(run), approaches=[ScriptedApproach(kill_seed=7, sig=int(sig))], num_experiments=1)
    grid.checkpoint()
    argv = ["srtk", "run_job", "--grid", str(run / "grid.json"), "--dataset", "ds1", "--approach", "scripted"]
    monkeypatch.setattr(sys, "argv", argv + ["--seed", "7"])
    with pytest.raises(SystemExit) as exit_info:
        main()
    checkpoint = run / "ds1" / "scripted" / "exp_7.sqlite"
    assert exit_info.value.code == 130
    assert str(checkpoint) in capsys.readouterr().out
    assert checkpoint.exists()


# --------------------------------------------------------------------------- #
# Job lock and status (stage 5)
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _held_lock(path):
    """Hold the job lock at ``path`` as another process would (flock locks are per open file)."""
    with open(path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _lock_taken(path):
    """``True`` if some open file holds an exclusive flock on ``path``."""
    with open(path, "a") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def _files(directory):
    return sorted(set(os.listdir(directory)) - {"cache"})


def _no_files(tmp_path):
    pass


def _finished(tmp_path):
    _job(tmp_path, ScriptedApproach(), keep_checkpoint=True).run()


def _finished_unsaved(tmp_path):  # killed while saving its result
    _finished(tmp_path)
    os.remove(tmp_path / "r.json")


def _interrupted(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        _job(tmp_path, ScriptedApproach(kill_seed=7)).run()


def _raised(tmp_path):  # a checkpoint without a stop reason, as after SIGKILL
    with pytest.raises(RuntimeError, match="boom"):
        _job(tmp_path, ScriptedApproach(raise_seed=7)).run()


def _garbage(tmp_path):
    (tmp_path / "r.sqlite").write_text("not a database")


class _LockProbe(_GridRecorder):
    """Records, in ``adapt`` and ``search``, whether the job's lock file is locked."""

    def __init__(self, lock_path):
        super().__init__("experiment")
        self.lock_path = lock_path
        self.locked = []

    def adapt(self, X, background_knowledge, required_constraints):
        self.locked.append(("adapt", _lock_taken(self.lock_path)))

    def search(self, evaluator, seed=None):
        self.locked.append(("search", _lock_taken(self.lock_path)))
        evaluator.evaluate_expr(["X_0"])


def _bad_header(tmp_path):  # a checkpoint whose stop reason cannot be read
    _interrupted(tmp_path)
    with contextlib.closing(sqlite3.connect(tmp_path / "r.sqlite")) as con, con:
        con.execute("UPDATE header SET value = 'not json' WHERE key = 'stop_reason'")


class _StatusProbe(_GridRecorder):
    """Records ``job.status`` during its search."""

    def search(self, evaluator, seed=None):
        self.seen = self.job.status
        evaluator.evaluate_expr(["X_0"])


class _FailingSetup(_GridRecorder):
    def adapt(self, X, background_knowledge, required_constraints):
        raise RuntimeError("setup failed")


def _before_lock(monkeypatch, action):
    """Run ``action`` between ``run()`` opening the lock file and locking it, as another process
    finishing the job at that moment would."""
    real = fcntl.flock

    def flock(file, operation):
        action()
        monkeypatch.setattr(fcntl, "flock", real)
        return real(file, operation)

    monkeypatch.setattr(fcntl, "flock", flock)


class TestJobStatus:
    # Reason: clause (one row per status; a result wins over a held lock; an unreadable checkpoint
    # is "crashed"). Implied: reading the status creates no lock file.
    @pytest.mark.parametrize(
        "setup, locked, expected",
        [
            (_no_files, False, "not started"),
            (_finished, False, "done"),
            (_no_files, True, "running"),
            (_finished, True, "done"),
            (_interrupted, False, "interrupted"),
            (_raised, False, "crashed"),
            (_finished_unsaved, False, "crashed"),
            (_garbage, False, "crashed"),
            (_bad_header, False, "crashed"),
        ],
        ids=[
            "not-started",
            "done",
            "running",
            "done-locked",
            "interrupted",
            "raised",
            "finished-unsaved",
            "garbage",
            "bad-header",
        ],
    )
    def test_status_from_files(self, tmp_path, monkeypatch, setup, locked, expected):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        setup(tmp_path)
        job = _job(tmp_path, ScriptedApproach())
        with _held_lock(tmp_path / "r.lock") if locked else contextlib.nullcontext():
            lock_file = (tmp_path / "r.lock").exists()
            assert job.status == expected
            assert (tmp_path / "r.lock").exists() is lock_file

    # Reason: clause (another process holds the lock -> RuntimeError before anything is changed;
    # rerun deletes only after taking the lock).
    def test_locked_job_raises_and_touches_nothing(self, tmp_path, monkeypatch):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        _finished(tmp_path)
        before = {name: (tmp_path / name).read_bytes() for name in ("r.json", "r.sqlite")}
        approach = ScriptedApproach()
        with _held_lock(tmp_path / "r.lock"):
            with pytest.raises(RuntimeError, match="another process"):
                _job(tmp_path, approach).run(rerun=True)
        assert approach.seeds == []
        assert {name: (tmp_path / name).read_bytes() for name in ("r.json", "r.sqlite")} == before
        assert _files(tmp_path) == ["r.json", "r.lock", "r.sqlite"]

    # Reason: clause (the lock is held while the job runs, adaptation included).
    def test_lock_is_held_during_adaptation_and_search(self, tmp_path, monkeypatch):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        approach = _LockProbe(tmp_path / "r.lock")
        _job(tmp_path, approach).run()
        assert approach.locked == [("adapt", True), ("search", True)]

    # Reason: implied (the OS frees the lock of a process that dies, so its job shows as crashed
    # and runs again).
    def test_lock_of_a_killed_process_is_free(self, tmp_path, monkeypatch):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        _raised(tmp_path)
        holder = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import fcntl, sys, time; f = open(sys.argv[1], 'a'); fcntl.flock(f, fcntl.LOCK_EX); "
                "print('locked', flush=True); time.sleep(60)",
                str(tmp_path / "r.lock"),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert holder.stdout.readline().strip() == "locked"
            assert _job(tmp_path, ScriptedApproach()).status == "running"
        finally:
            holder.kill()
            holder.wait()
        job = _job(tmp_path, ScriptedApproach())
        assert job.status == "crashed"
        with pytest.warns(UserWarning, match="snapshot"):  # S4: it cannot resume, so it restarts
            job.run()
        assert job.status == "done"

    # Reason: clause (where files cannot be locked, status cannot see a running job, and run warns
    # and runs unlocked).
    @pytest.mark.parametrize("no_lock", ["no-fcntl", "flock-fails"])
    def test_without_locking_warns_and_runs(self, tmp_path, monkeypatch, no_lock):
        import SRToolkit.experiments.experiment_grid as eg

        _isolate_cache(monkeypatch, tmp_path / "cache")
        job = _job(tmp_path, ScriptedApproach())
        with _held_lock(tmp_path / "r.lock"):
            if no_lock == "no-fcntl":
                monkeypatch.setattr(eg, "fcntl", None)
            else:

                def unsupported(*args):
                    raise OSError(38, "Function not implemented")

                monkeypatch.setattr(fcntl, "flock", unsupported)
            assert job.status == "not started"  # a running job is not seen
        with pytest.warns(UserWarning, match="cannot be locked"):
            job.run()
        assert _files(tmp_path) == ["r.json"]

    # Reason: clause (a done job does nothing, even while its lock is held; rerun deletes the
    # result and checkpoint and runs it again; the lock file is deleted once the result is saved).
    def test_done_job_runs_again_only_with_rerun(self, tmp_path, monkeypatch):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        _job(tmp_path, ScriptedApproach(), keep_checkpoint=True).run()  # a finished checkpoint is not reused
        assert _files(tmp_path) == ["r.json", "r.sqlite"]
        approach = ScriptedApproach()
        with _held_lock(tmp_path / "r.lock"):
            _job(tmp_path, approach).run()
        assert approach.seeds == []
        _job(tmp_path, approach, keep_checkpoint=True).run(rerun=True)
        assert approach.seeds == [7]
        assert _files(tmp_path) == ["r.json", "r.sqlite"]

    # Reason: clause (rerun runs afresh: an unfinished checkpoint is not resumed).
    def test_rerun_discards_an_unfinished_checkpoint(self, tmp_path, monkeypatch):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        with pytest.raises(KeyboardInterrupt):
            _job(tmp_path, SteppedApproach(stop_after_asks=3), callbacks=SignalAfter(3)).run()
        EventLog.events.clear()
        _job(tmp_path, SteppedApproach(stop_after_asks=3), callbacks=EventLog()).run(rerun=True)
        rerun_events = list(EventLog.events)
        EventLog.events.clear()
        _job(tmp_path, SteppedApproach(stop_after_asks=3), name="straight.json", callbacks=EventLog()).run()
        assert rerun_events == EventLog.events  # from the start; a resume evaluates only the last 2
        assert _result(tmp_path / "r.json") == _result(tmp_path / "straight.json")

    # Regression (code review): a rerun whose setup fails keeps the old result.
    def test_rerun_with_failing_setup_keeps_the_result(self, tmp_path, monkeypatch):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        _finished(tmp_path)
        before = {name: (tmp_path / name).read_bytes() for name in ("r.json", "r.sqlite")}
        with pytest.raises(RuntimeError, match="setup failed"):
            _job(tmp_path, _FailingSetup()).run(rerun=True)
        assert {name: (tmp_path / name).read_bytes() for name in ("r.json", "r.sqlite")} == before

    # Regression (code review): status inside the running process neither opens nor locks the lock
    # file (on NFS that would release the run's lock).
    def test_status_during_the_run(self, tmp_path, monkeypatch):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        approach = _StatusProbe()
        job = _job(tmp_path, approach)
        approach.job = job
        calls = []
        real = fcntl.flock
        monkeypatch.setattr(fcntl, "flock", lambda file, operation: (calls.append(operation), real(file, operation)))
        job.run()
        assert (approach.seen, calls) == ("running", [fcntl.LOCK_EX | fcntl.LOCK_NB])

    # Regression (code review): another process finishes the job between run() opening the lock
    # file and locking it. Without rerun the job is done and nothing runs; with rerun the lock file
    # was replaced, so run() raises and changes nothing. No lock file is left behind.
    @pytest.mark.parametrize("deleted, rerun", [(True, False), (False, False), (True, True)])
    def test_job_finished_while_taking_the_lock(self, tmp_path, monkeypatch, deleted, rerun):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        _finished(tmp_path)
        if not rerun:
            os.replace(tmp_path / "r.json", tmp_path / "done.json")

        def finish():
            if not rerun:
                os.replace(tmp_path / "done.json", tmp_path / "r.json")
            if deleted:
                os.remove(tmp_path / "r.lock")

        _before_lock(monkeypatch, finish)
        approach = ScriptedApproach()
        if rerun:
            with pytest.raises(RuntimeError, match="just finished"):
                _job(tmp_path, approach).run(rerun=True)
        else:
            _job(tmp_path, approach).run()
        assert approach.seeds == []
        assert _files(tmp_path) == ["r.json", "r.sqlite"]


def _grid_with_statuses(tmp_path, monkeypatch):
    """A grid of seeds 7-11: 7 done, 8 crashed, 9 running (lock held by the caller), 10 interrupted,
    11 not started."""
    _isolate_cache(monkeypatch, tmp_path / "cache")
    run = tmp_path / "run"
    grid = _make_grid(str(run), approaches=[ScriptedApproach(raise_seed=8, kill_seed=10)], num_experiments=5)
    grid.build_job("ds1", "scripted", 7).run()
    with pytest.raises(RuntimeError, match="boom"):
        grid.build_job("ds1", "scripted", 8).run()
    with pytest.raises(KeyboardInterrupt):
        grid.build_job("ds1", "scripted", 10).run()
    return grid, run / "ds1" / "scripted" / "exp_9.lock"


def _command_seeds(path):
    return [line.split("--seed ")[1] for line in path.read_text().splitlines() if not line.startswith("#")]


# Reason: clause (skip_completed skips done and running jobs, keeps interrupted and crashed ones;
# save_commands(skip_completed=False) adds --rerun to done jobs only).
def test_grid_job_selection_skips_done_and_running(tmp_path, monkeypatch):
    grid, lock = _grid_with_statuses(tmp_path, monkeypatch)
    commands = tmp_path / "commands.sh"
    with _held_lock(lock):
        assert [job.seed for job in grid.create_jobs()] == [8, 10, 11]
        assert [job.seed for job in grid.create_jobs(skip_completed=False)] == [7, 8, 9, 10, 11]
        grid.save_commands(str(commands), materialize_data=False)
        assert _command_seeds(commands) == ["8", "10", "11"]
        grid_json = tmp_path / "run" / "grid.json"
        assert (
            f"python -m SRToolkit.experiments run_job --grid {grid_json} --dataset ds1 --approach scripted --seed 8"
            in (commands.read_text().splitlines())
        )
        grid.save_commands(str(commands), materialize_data=False, skip_completed=False)
        assert _command_seeds(commands) == ["7 --rerun", "8", "9", "10", "11"]


# Regression (code review): a job that finishes while the commands are written gets no --rerun
# line when done jobs are skipped.
def test_save_commands_skipping_done_jobs_never_reruns(tmp_path, monkeypatch):
    _isolate_cache(monkeypatch, tmp_path / "cache")
    grid = _make_grid(str(tmp_path / "run"), num_experiments=1)
    reads = iter(["not started", "done"])  # finished between the two reads
    monkeypatch.setattr(ExperimentJob, "status", property(lambda job: next(reads, "done")))
    commands = tmp_path / "commands.sh"
    grid.save_commands(str(commands), materialize_data=False)
    assert _command_seeds(commands) == ["7"]


# Reason: clause (progress: done/total table, counts of running, interrupted and crashed jobs,
# and the interrupted and crashed ones in dataset, approach, seed order).
def test_progress_lists_unfinished_jobs(tmp_path, monkeypatch, capsys):
    grid, lock = _grid_with_statuses(tmp_path, monkeypatch)
    with _held_lock(lock):
        grid.progress()
    out = capsys.readouterr().out.splitlines()
    assert (out[0].split(), set(out[1]), out[2].split()) == (["Dataset", "scripted"], {"-"}, ["ds1", "1/5"])
    assert out[2].endswith("1/5")  # counts are right-aligned, as in the example
    assert out[3:5] == ["", "running 1, interrupted 1, crashed 1"]
    assert [line.split() for line in out[5:]] == [
        ["crashed", "ds1", "/", "scripted", "/", "seed", "8"],
        ["interrupted", "ds1", "/", "scripted", "/", "seed", "10"],
    ]


def _cli(monkeypatch, *argv):
    """Run the CLI with ``argv`` in-process and return its exit code."""
    from SRToolkit.experiments.__main__ import main

    monkeypatch.setattr(sys, "argv", ["srtk", *argv])
    try:
        main()
    except SystemExit as exit_info:
        return exit_info.code or 0
    return 0


def _cli_run_job(monkeypatch, run, *extra):
    """Run ``run_job`` for seed 7 of the grid at ``run`` in-process and return its exit code."""
    grid = ["--grid", str(run / "grid.json"), "--dataset", "ds1", "--approach", "scripted", "--seed", "7"]
    return _cli(monkeypatch, "run_job", *grid, *extra)


def _cli_grid(tmp_path, monkeypatch, result):
    """A checkpointed one-job grid; ``result`` (if not None) is written as the job's result."""
    _isolate_cache(monkeypatch, tmp_path / "cache")
    run = tmp_path / "run"
    _make_grid(str(run), approaches=[ScriptedApproach()], num_experiments=1).checkpoint()
    (run / "ds1" / "scripted").mkdir(parents=True)
    if result is not None:
        (run / "ds1" / "scripted" / "exp_7.json").write_text(result)
    return run


# Reason: clause (run_job: a job running elsewhere prints one line and exits 1; a done job, even
# with its lock held, prints "Already done" and exits 0; both on stdout, leaving the result as is).
@pytest.mark.parametrize(
    "result, locked, code, message",
    [(None, True, 1, "another process"), ("{}", False, 0, "Already done"), ("{}", True, 0, "Already done")],
    ids=["running", "done", "done-locked"],
)
def test_cli_run_job_running_or_done(tmp_path, monkeypatch, capsys, result, locked, code, message):
    run = _cli_grid(tmp_path, monkeypatch, result)
    job_dir = run / "ds1" / "scripted"
    with _held_lock(job_dir / "exp_7.lock") if locked else contextlib.nullcontext():
        assert _cli_run_job(monkeypatch, run) == code
    out, err = capsys.readouterr()
    assert (len(out.splitlines()), message in out, err) == (1, True, "")
    assert sorted(set(os.listdir(job_dir)) - {"exp_7.lock"}) == (["exp_7.json"] if result else [])
    assert result is None or (job_dir / "exp_7.json").read_text() == result


# Reason: clause (run_job --rerun runs a done job again).
def test_cli_run_job_rerun(tmp_path, monkeypatch, capsys):
    run = _cli_grid(tmp_path, monkeypatch, "{}")  # a placeholder result: the job counts as done
    assert _cli_run_job(monkeypatch, run, "--rerun") == 0
    assert "Already done" not in capsys.readouterr().out
    assert len(Results.load(str(run / "ds1" / "scripted" / "exp_7.json"))) == 1


# Reason: clause (commands skips done jobs; --all lists them with --rerun; --python sets the executable).
@pytest.mark.parametrize("flags, seeds", [([], []), (["--all"], ["7 --rerun"])], ids=["default", "all"])
def test_cli_commands_all_reruns_done_jobs(tmp_path, monkeypatch, flags, seeds):
    run = _cli_grid(tmp_path, monkeypatch, "{}")
    commands = tmp_path / "commands.sh"
    argv = ["commands", "--grid", str(run / "grid.json"), "--out", str(commands), "--python", "py3", *flags]
    assert _cli(monkeypatch, *argv) == 0
    assert _command_seeds(commands) == seeds
    assert all(line.startswith("py3 -m ") for line in commands.read_text().splitlines()[1:])


# --------------------------------------------------------------------------- #
# keep_all_models
# --------------------------------------------------------------------------- #
class TestKeepAllModels:
    # Reason: clause (a job with keep_all_models=False saves the top_k models and a summary, which
    # answer like a full job; with keep_checkpoint the result refers to the compacted file).
    def test_job_saves_compact_result_with_checkpoint_reference(self, tmp_path, monkeypatch):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        # top_k=1: the smallest allowed; the job evaluates 5 distinct models.
        _job(tmp_path, ScriptedApproach(), keep_checkpoint=True, keep_all_models=False, top_k=1).run()
        _job(tmp_path, ScriptedApproach(), name="full.json").run()
        saved = Results.load(str(tmp_path / "r.json"))[0]
        full = Results.load(str(tmp_path / "full.json"))[0]
        assert saved.summary is not None
        assert saved.all_models == full.top_models(1)
        np.testing.assert_equal(result_answers(saved, 1), result_answers(full, 1))
        assert saved.checkpoint["path"] == str(tmp_path / "r.sqlite")
        store = saved.open_checkpoint()
        try:
            assert list(store.trace) == full.trace
        finally:
            store.close()

    # Reason: clause (ExperimentJob.run Raises: keep_all_models False and top_k <= 0, before
    # anything runs).
    def test_job_compact_top_k_not_positive_raises_first(self, tmp_path, monkeypatch):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        approach = ScriptedApproach()
        with pytest.raises(ValueError, match="top_k"):
            _job(tmp_path, approach, keep_all_models=False, top_k=0).run()
        assert approach.seeds == []
        assert ((tmp_path / "r.json").exists(), (tmp_path / "r.sqlite").exists()) == (False, False)

    # Reason: clause (the ValueError comes before anything is changed: rerun keeps the old result).
    def test_rerun_with_compact_top_k_not_positive_keeps_the_result(self, tmp_path, monkeypatch):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        _job(tmp_path, ScriptedApproach(), keep_checkpoint=True).run()
        before = {name: (tmp_path / name).read_bytes() for name in ("r.json", "r.sqlite")}
        with pytest.raises(ValueError, match="top_k"):
            _job(tmp_path, ScriptedApproach(), keep_all_models=False, top_k=0).run(rerun=True)
        assert {name: (tmp_path / name).read_bytes() for name in ("r.json", "r.sqlite")} == before

    # Reason: clause (ExperimentGrid passes keep_all_models, keep_checkpoint and snapshot_interval to
    # its jobs and saves them in to_dict; a recipe without one loads its default).
    @pytest.mark.parametrize(
        "flag, default, other",
        [
            ("keep_all_models", True, False),
            ("keep_checkpoint", False, True),
            ("snapshot_interval", 300.0, None),
            ("snapshot_interval", 300.0, 0.5),  # any positive value is allowed
        ],
    )
    def test_grid_job_flag_round_trip(self, tmp_path, flag, default, other):
        run = str(tmp_path / "run")

        def job_flag(grid):
            return getattr(grid.build_job("ds1", "ProGED", 7).info, flag)

        assert job_flag(_make_grid(run)) == default
        grid = _make_grid(run, top_k=1, **{flag: other})  # top_k=1: the smallest allowed when compact
        assert job_flag(grid) == other
        d = grid.to_dict()
        assert d[flag] == other
        assert job_flag(ExperimentGrid.from_dict(d, results_dir=run)) == other
        del d[flag]  # a recipe written before the flag existed
        assert job_flag(ExperimentGrid.from_dict(d, results_dir=run)) == default

    # Reason: clause (ExperimentGrid and from_dict Raise: keep_all_models False and top_k <= 0).
    @pytest.mark.parametrize("top_k", [0, -1])
    def test_grid_compact_top_k_not_positive_raises(self, tmp_path, top_k):
        run = str(tmp_path / "run")
        with pytest.raises(ValueError, match="top_k"):
            _make_grid(run, keep_all_models=False, top_k=top_k)
        d = _make_grid(run, keep_all_models=False, top_k=1).to_dict() | {"top_k": top_k}
        with pytest.raises(ValueError, match="top_k"):
            ExperimentGrid.from_dict(d, results_dir=run)

    # Reason: clause (ExperimentGrid and from_dict Raise: snapshot_interval not positive and not None).
    @pytest.mark.parametrize("interval", [0, -1.0])
    def test_grid_snapshot_interval_not_positive_raises(self, tmp_path, interval):
        run = str(tmp_path / "run")
        with pytest.raises(ValueError, match="snapshot_interval"):
            _make_grid(run, snapshot_interval=interval)
        d = _make_grid(run).to_dict() | {"snapshot_interval": interval}
        with pytest.raises(ValueError, match="snapshot_interval"):
            ExperimentGrid.from_dict(d, results_dir=run)


# --------------------------------------------------------------------------- #
# Extending jobs (stage 6)
# --------------------------------------------------------------------------- #
def _extension_grid(tmp_path, monkeypatch, name="run", budget=5, keep_checkpoint=True):
    _isolate_cache(monkeypatch, tmp_path / "cache")
    return _make_grid(
        str(tmp_path / name),
        datasets=[_make_dataset(max_evaluations=budget)],
        approaches=[SteppedApproach()],
        num_experiments=1,
        keep_checkpoint=keep_checkpoint,
        snapshot_interval=1e-9,
    )


class TestExtendJobs:
    # Reason: clause (extend() continues a done job with the given limits, replaces its result, keeps the
    # checkpoint compacted and releases the lock).
    def test_done_job_extends_to_the_given_limits(self, tmp_path, monkeypatch):
        grid = _extension_grid(tmp_path, monkeypatch)
        job = grid.build_job("ds1", "stepped", 7)
        job.run()  # 5 calls: the budget cuts the third step
        job.extend(max_evaluations=10, success_threshold=5e-8)  # below RMSE's default 1e-7, met by no candidate
        straight = _extension_grid(tmp_path, monkeypatch, name="straight", budget=10, keep_checkpoint=False)
        straight.build_job("ds1", "stepped", 7).run()
        extended = Results.load(job.result_path)[0]
        assert {**_result(job.result_path), "checkpoint": None, "success_threshold": None} == {
            **_result(straight.build_job("ds1", "stepped", 7).result_path),
            "success_threshold": None,
        }
        assert [(s["max_evaluations"], s["success_threshold"]) for s in extended.limits] == [(5, 1e-7), (10, 5e-8)]
        assert (_journal_mode(job.checkpoint_path), os.path.exists(job._lock_path)) == ("delete", False)
        assert job.status == "done"

    # Reason: clause (errors: a job without a checkpoint, never started or done without keep_checkpoint,
    # raises ValueError; a job whose lock another process holds raises RuntimeError; the result stays).
    @pytest.mark.parametrize(
        "run_first, keep, locked, error, match",
        [
            (False, False, False, ValueError, "never started"),
            (True, False, False, ValueError, "rerun"),
            (True, True, True, RuntimeError, "another process"),
        ],
        ids=["new", "done", "locked"],
    )
    def test_job_that_cannot_be_extended(self, tmp_path, monkeypatch, run_first, keep, locked, error, match):
        grid = _extension_grid(tmp_path, monkeypatch, keep_checkpoint=keep)
        job = grid.build_job("ds1", "stepped", 7)
        if run_first:
            job.run()
        before = job.is_complete and open(job.result_path).read()
        with _held_lock(job._lock_path) if locked else contextlib.nullcontext():
            with pytest.raises(error, match=match):
                job.extend(max_evaluations=10)
        assert (job.is_complete and open(job.result_path).read()) == before

    # Reason: clause (an interrupt raises KeyboardInterrupt and keeps the checkpoint; a done job keeps its
    # old result and status "done" until an extend finishes, which continues with the new limits).
    def test_interrupted_extension_keeps_the_done_result(self, tmp_path, monkeypatch):
        grid = _extension_grid(tmp_path, monkeypatch)
        job = grid.build_job("ds1", "stepped", 7)
        job.run()
        before = open(job.result_path).read()
        with pytest.raises(KeyboardInterrupt), pytest.warns(UserWarning, match="interrupted"):
            job.extend(max_evaluations=10, callbacks=SignalAfter(1))
        assert (open(job.result_path).read(), job.status, os.path.exists(job.checkpoint_path)) == (before, "done", True)
        job.extend()  # None keeps the new budget
        straight = _extension_grid(tmp_path, monkeypatch, name="straight", budget=10, keep_checkpoint=False)
        straight.build_job("ds1", "stepped", 7).run()
        assert {**_result(job.result_path), "checkpoint": None} == _result(
            straight.build_job("ds1", "stepped", 7).result_path
        )
