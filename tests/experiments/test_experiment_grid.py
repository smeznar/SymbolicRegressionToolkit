"""Tests for ExperimentGrid serialization (to_dict / from_dict) and the
results_dir anchoring fix in load()."""

import json
import os
import shutil
import signal
import sqlite3
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pytest
from strategies import (
    EventLog,
    ScriptedApproach,
    SignalAfter,
    StatelessSteppedApproach,
    SteppedApproach,
    result_answers,
)

from SRToolkit.approaches import ApproachConfig, ProGED, SR_approach
from SRToolkit.dataset import SR_dataset, data_cache
from SRToolkit.dataset.sampling import UniformSampling
from SRToolkit.evaluation.results import SR_results
from SRToolkit.experiments import ExperimentGrid, ExperimentInfo, ExperimentJob
from SRToolkit.utils.expression_constraints import MaxConstants, MaxTokens, MaxTreeDepth


def _make_dataset(name="ds1", benchmark="unit_test", version="1.0.0", max_evaluations=-1):
    """A small, network-free sample-source dataset (no arrays embedded in config)."""
    return SR_dataset.from_samplers(
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
    assert "ds1" in result.stdout and "ProGED" in result.stdout
    assert "1/2" in result.stdout


# --------------------------------------------------------------------------- #
# What adapt receives (stage D)
# --------------------------------------------------------------------------- #
_TASK_REQUIRED = [MaxConstants(8), MaxTokens(20), MaxTreeDepth(5)]


def _task_dataset():
    """Grid dataset whose required list is _TASK_REQUIRED, with a hint and a description."""
    return SR_dataset.from_samplers(
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


class _GridRecorder(SR_approach):
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

    def search(self, sr_evaluator, seed=None):
        self.events.append("search")
        sr_evaluator.evaluate_expr(["X_0"])

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
    return {**SR_results.load(str(path))[0].to_dict(), "wall_time": None}


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
        assert not (tmp_path / "r.sqlite-wal").exists()  # closed
        saved = SR_results.from_checkpoint(tmp_path / "r.sqlite")[0]
        assert (saved.trace, saved.stop_reason) == ([0, 1, 0], "interrupted")

    def test_rerun_of_interrupted_job_equals_uninterrupted_job(self, tmp_path, monkeypatch):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        with pytest.raises(KeyboardInterrupt):  # in the second step: 4 calls and a snapshot are saved
            _job(tmp_path, SteppedApproach(stop_after_asks=3), callbacks=SignalAfter(3)).run()
        EventLog.events.clear()
        _job(tmp_path, SteppedApproach(stop_after_asks=3), callbacks=EventLog()).run()
        assert len(EventLog.events) == 2  # continued from the snapshot
        _job(tmp_path, SteppedApproach(stop_after_asks=3), name="straight.json").run()
        assert _result(tmp_path / "r.json") == _result(tmp_path / "straight.json")
        assert not (tmp_path / "r.sqlite").exists()

    def test_rerun_of_search_only_job_restarts_with_warning(self, tmp_path, monkeypatch):
        # S4: an approach that cannot continue from a snapshot cannot resume; its job starts afresh.
        _isolate_cache(monkeypatch, tmp_path / "cache")
        with pytest.raises(KeyboardInterrupt):
            _job(tmp_path, ScriptedApproach(kill_seed=7)).run()
        with pytest.warns(UserWarning, match="snapshot"):
            _job(tmp_path, ScriptedApproach()).run()
        _job(tmp_path, ScriptedApproach(), name="straight.json").run()
        assert _result(tmp_path / "r.json") == _result(tmp_path / "straight.json")
        assert not (tmp_path / "r.sqlite").exists()

    def test_rerun_of_stateless_step_job_restarts_with_warning(self, tmp_path, monkeypatch):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        with pytest.raises(KeyboardInterrupt):
            _job(tmp_path, StatelessSteppedApproach(stop_after_asks=3), callbacks=SignalAfter(3)).run()
        with pytest.warns(UserWarning, match="snapshot"):
            _job(tmp_path, StatelessSteppedApproach(stop_after_asks=3)).run()
        _job(tmp_path, StatelessSteppedApproach(stop_after_asks=3), name="straight.json").run()
        assert _result(tmp_path / "r.json") == _result(tmp_path / "straight.json")

    def test_job_search_exception_keeps_checkpoint(self, tmp_path, monkeypatch):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        job = _job(tmp_path, ScriptedApproach(raise_seed=7))
        with pytest.raises(RuntimeError, match="boom"):
            job.run()
        assert (job.is_complete, (tmp_path / "r.sqlite").exists()) == (False, True)

    def test_job_resume_with_other_settings_raises(self, tmp_path, monkeypatch):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        with pytest.raises(KeyboardInterrupt):
            _job(tmp_path, SteppedApproach(), callbacks=SignalAfter(3)).run()
        other = SR_dataset.from_samplers(
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
        saved = SR_results.load(str(tmp_path / "r.json"))[0]
        assert ((saved.summary is None), saved.checkpoint) == (keep_all_models, None)
        assert not (tmp_path / "r.sqlite").exists()

    def test_rerunning_a_kept_finished_job_gives_the_same_result(self, tmp_path, monkeypatch):
        _isolate_cache(monkeypatch, tmp_path / "cache")
        _job(tmp_path, SteppedApproach(stop_after_asks=3), keep_checkpoint=True).run()
        expected = _result(tmp_path / "r.json")
        EventLog.events.clear()
        _job(tmp_path, SteppedApproach(stop_after_asks=3), keep_checkpoint=True, callbacks=EventLog()).run()
        assert (EventLog.events, _result(tmp_path / "r.json")) == ([], expected)  # finished: nothing evaluated


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
        saved = SR_results.load(str(tmp_path / "r.json"))[0]
        full = SR_results.load(str(tmp_path / "full.json"))[0]
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

    # Reason: clause (ExperimentGrid passes keep_all_models to its jobs and saves it in to_dict; a
    # recipe without it loads True).
    @pytest.mark.parametrize("flag, default", [("keep_all_models", True), ("keep_checkpoint", False)])
    def test_grid_job_flag_round_trip(self, tmp_path, flag, default):
        run = str(tmp_path / "run")

        def job_flag(grid):
            return getattr(grid.build_job("ds1", "ProGED", 7).info, flag)

        assert job_flag(_make_grid(run)) is default
        grid = _make_grid(run, top_k=1, **{flag: not default})  # top_k=1: the smallest allowed when compact
        assert job_flag(grid) is (not default)
        d = grid.to_dict()
        assert d[flag] is (not default)
        assert job_flag(ExperimentGrid.from_dict(d, results_dir=run)) is (not default)
        del d[flag]  # a recipe written before the flag existed
        assert job_flag(ExperimentGrid.from_dict(d, results_dir=run)) is default

    # Reason: clause (ExperimentGrid and from_dict Raise: keep_all_models False and top_k <= 0).
    @pytest.mark.parametrize("top_k", [0, -1])
    def test_grid_compact_top_k_not_positive_raises(self, tmp_path, top_k):
        run = str(tmp_path / "run")
        with pytest.raises(ValueError, match="top_k"):
            _make_grid(run, keep_all_models=False, top_k=top_k)
        d = _make_grid(run, keep_all_models=False, top_k=1).to_dict() | {"top_k": top_k}
        with pytest.raises(ValueError, match="top_k"):
            ExperimentGrid.from_dict(d, results_dir=run)
