"""Tests for SRToolkit.approaches.EDHiE (tiny configs: training is expensive)."""

import logging
import random
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pymoo")

from SRToolkit.approaches.EDHiE import EDHiE  # noqa: E402
from SRToolkit.dataset.background_knowledge import BackgroundKnowledge  # noqa: E402
from SRToolkit.evaluation import SR_evaluator  # noqa: E402
from SRToolkit.utils.expression_constraints import MaxTokens, RequiredVariables  # noqa: E402
from SRToolkit.utils.grammar.grammar import Grammar, Rule  # noqa: E402
from SRToolkit.utils.symbol_library import SymbolLibrary  # noqa: E402

_LOGGER = "SRToolkit.dataset.background_knowledge"
_X = np.random.default_rng(0).random((10, 2)) + 0.5


def _lib():
    return SymbolLibrary.from_symbol_list(["+", "*", "C"], num_variables=2)


def _products_of_x0():
    return Grammar(Rule.from_line("E -> E '*' 'X_0' [0.5] | 'X_0' [0.5]"), start="E")


def _set_up(constraint, lib):
    constraint.setup(lib)
    return constraint


def _tiny(**kwargs):
    return EDHiE(latent_size=8, max_expression_length=10, epochs=1, population_size=10, verbose=False, **kwargs)


def test_training_expressions_from_reference_grammar_follow_constraints(caplog):
    lib = _lib()
    bk = BackgroundKnowledge(lib, reference_grammar=_products_of_x0())
    approach = _tiny(num_expressions=20, max_consecutive_skips=50)
    with caplog.at_level(logging.INFO, logger=_LOGGER), pytest.warns(UserWarning):
        expressions = approach.training_expressions(bk, (_set_up(MaxTokens(5), lib),))
    assert len(expressions) == 20
    assert {"".join(e) for e in expressions} == {"X_0", "X_0*X_0", "X_0*X_0*X_0"}
    assert [r.levelno for r in caplog.records if r.name == _LOGGER] == [logging.INFO]


def test_training_expressions_from_symbol_library_are_unique():
    lib = _lib()
    approach = _tiny(num_expressions=30)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        expressions = approach.training_expressions(BackgroundKnowledge(lib), ())
    assert len({"".join(e) for e in expressions}) == 30 == len(expressions)
    assert all(len(e) <= 10 for e in expressions)
    allowed = set(lib.symbols2index()) | {"(", ")"}
    assert {t for e in expressions for t in e} <= allowed


def test_training_expressions_raise_when_nothing_passes():
    lib = _lib()
    bk = BackgroundKnowledge(lib, reference_grammar=_products_of_x0())
    approach = _tiny(num_expressions=20, max_consecutive_skips=50)
    with pytest.raises(ValueError):
        approach.training_expressions(bk, (_set_up(RequiredVariables(["X_1"]), lib),))


@pytest.mark.slow
def test_adapt_trains_and_keeps_scope_once():
    # Regression: a reused instance must train again for the next task, not keep this model.
    lib = _lib()
    approach = _tiny(num_expressions=50)
    approach.adapt(np.empty((0, 9)), BackgroundKnowledge(lib), (_set_up(MaxTokens(10), lib),))
    assert approach.model is not None
    assert approach.adaptation_scope == "once"


@pytest.mark.slow
def test_adapt_restores_global_random_states():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    before = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state().clone())
    _tiny(num_expressions=50, training_seed=3).adapt(_X, BackgroundKnowledge(_lib()), ())
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1])
    assert torch.equal(torch.get_rng_state(), before[2])


@pytest.mark.slow
def test_adapt_is_reproducible_with_training_seed():
    def weights():
        approach = _tiny(num_expressions=50, training_seed=3)
        approach.adapt(_X, BackgroundKnowledge(_lib()), ())
        return [p.detach().clone() for p in approach.model.parameters()]

    first, second = weights(), weights()
    assert all(torch.equal(a, b) for a, b in zip(first, second))


@pytest.fixture(scope="module")
def trained_model():
    approach = _tiny(num_expressions=200, training_seed=0)
    approach.adapt(_X, BackgroundKnowledge(_lib()), ())
    return approach.model


def test_weights_path_makes_scope_never(tmp_path, trained_model):
    approach = _tiny()
    approach.model = trained_model
    path = str(tmp_path / "hvae.pt")
    approach.save_adapted_state(path)
    assert _tiny(weights_path=path).adaptation_scope == "never"


def _search(model, seed=0, **kwargs):
    approach = _tiny(**kwargs)
    approach.model = model
    se = SR_evaluator(
        _X,
        _X[:, 0],
        symbol_library=_lib(),
        max_evaluations=30,
        success_threshold=-1,
        required_constraints=[MaxTokens(7)],
    )
    approach.search(se, seed=seed)
    return se


@pytest.mark.slow
def test_search_skips_evaluated_and_rejected(trained_model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        se = _search(trained_model, max_consecutive_skips=500)
    assert len(se.models) == se.total_evaluations
    assert all(m.status != "rejected" for m in se.models)


@pytest.mark.slow
def test_search_without_skipping_bills_every_individual(trained_model):
    se = _search(trained_model, skip_evaluated=False, skip_rejected=False)
    assert se.total_evaluations == 30
    assert len(se.models) < 30


@pytest.mark.slow
def test_search_stops_after_max_consecutive_skips(trained_model):
    with pytest.warns(UserWarning):
        se = _search(trained_model, max_consecutive_skips=1)
    assert se.total_evaluations < 30


def test_config_round_trip_keeps_skip_settings():
    config = _tiny(max_consecutive_skips=7, skip_evaluated=False, skip_rejected=False, training_seed=3).config.to_dict()
    rebuilt = EDHiE.from_config(config)
    settings = (rebuilt.max_consecutive_skips, rebuilt.skip_evaluated, rebuilt.skip_rejected, rebuilt.training_seed)
    assert settings == (7, False, False, 3)


def test_from_config_defaults_for_older_configs():
    approach = EDHiE.from_config({"latent_size": 8})
    settings = (approach.max_consecutive_skips, approach.skip_evaluated, approach.skip_rejected, approach.training_seed)
    assert settings == (10000, True, True, None)
