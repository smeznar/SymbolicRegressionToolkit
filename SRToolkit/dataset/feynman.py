"""
Feynman symbolic regression benchmark.
"""

import warnings
from typing import Optional

from SRToolkit.utils.symbol_library import SymbolLibrary

from .data_source import SampleSource, UrlSource
from .sampling import UniformSampling
from .sr_benchmark import SR_benchmark


class Feynman(SR_benchmark):
    """
    The Feynman symbolic regression benchmark.

    Contains 100 physics equations with up to 9 variables. Data is downloaded on first use from
    the SymbolicRegressionToolkit repository (10,000 samples per dataset instead of the original
    1,000,000 from the paper). If the download fails, data is generated from the stored per-variable
    samplers using ``n_samples`` points and the given ``seed``.

    References:
        [Udrescu & Tegmark (2020)][cite-feynman]

    Examples:
        >>> benchmark = Feynman()
        >>> len(benchmark.list_datasets(verbose=False))
        100

    Args:
        n_samples: Number of samples to generate per dataset when ``force_generate=True``
            (sampler-based data generation). Defaults to ``10000``.
        seed: Random seed used for sampler-based data generation. Defaults to ``42``.
        force_generate: If ``True``, generate fresh data from the stored samplers instead of
            downloading the pre-generated data. Defaults to ``False``.
    """

    __data_version__ = "1.0.0"

    def __init__(
        self,
        n_samples: int = 10000,
        seed: Optional[int] = 42,
        force_generate: bool = False,
    ):
        super().__init__("feynman", version="1.0.0")
        self._n_samples = n_samples
        self._seed = seed
        self._force_generate = force_generate
        self._populate()

    def _populate(self):
        # fmt: off
        seed = self._seed
        url = "https://raw.githubusercontent.com/smeznar/SymbolicRegressionToolkit/master/data/feynman.zip"

        self.metadata = {
            "description": "Feynman benchmark containing 100 equations from the domain of physics. "
            "Expressions can contain up to 9 variables.",
            "citation": """@article{Tegmark2020Feynman,
  title={{AI Feynman: A physics-inspired method for symbolic regression}},
  author={Udrescu, Silviu-Marian and Tegmark, Max},
  journal={Science Advances},
  volume={6},
  number={16},
  pages={eaay2631},
  year={2020},
  publisher={American Association for the Advancement of Science}
}
""",
        }

        # The canonical data is downloaded once from the archive (see _ensure_data) so every
        # machine benchmarks on identical inputs. Each dataset's own data_source is a
        # SampleSource: a transparent, per-dataset fallback that regenerates the data from
        # that dataset's samplers if the download is unavailable (or force_generate is set).
        self._archive_source = UrlSource(url)
        data_source = SampleSource(n_samples=self._n_samples, seed=seed)

        sl_1v = SymbolLibrary.from_symbol_list(
            ["+", "-", "*", "/", "u-", "sqrt", "sin", "cos", "exp", "arcsin", "tanh", "ln", "^2", "^3", "pi", "C"], 1
        )
        sl_2v = SymbolLibrary.from_symbol_list(
            ["+", "-", "*", "/", "u-", "sqrt", "sin", "cos", "exp", "arcsin", "tanh", "ln", "^2", "^3", "pi", "C"], 2
        )
        sl_3v = SymbolLibrary.from_symbol_list(
            ["+", "-", "*", "/", "u-", "sqrt", "sin", "cos", "exp", "arcsin", "tanh", "ln", "^2", "^3", "pi", "C"], 3
        )
        sl_4v = SymbolLibrary.from_symbol_list(
            ["+", "-", "*", "/", "u-", "sqrt", "sin", "cos", "exp", "arcsin", "tanh", "ln", "^2", "^3", "pi", "C"], 4
        )
        sl_5v = SymbolLibrary.from_symbol_list(
            ["+", "-", "*", "/", "u-", "sqrt", "sin", "cos", "exp", "arcsin", "tanh", "ln", "^2", "^3", "pi", "C"], 5
        )
        sl_6v = SymbolLibrary.from_symbol_list(
            ["+", "-", "*", "/", "u-", "sqrt", "sin", "cos", "exp", "arcsin", "tanh", "ln", "^2", "^3", "pi", "C"], 6
        )
        sl_8v = SymbolLibrary.from_symbol_list(
            ["+", "-", "*", "/", "u-", "sqrt", "sin", "cos", "exp", "arcsin", "tanh", "ln", "^2", "^3", "pi", "C"], 8
        )
        sl_9v = SymbolLibrary.from_symbol_list(
            ["+", "-", "*", "/", "u-", "sqrt", "sin", "cos", "exp", "arcsin", "tanh", "ln", "^2", "^3", "pi", "C"], 9
        )

        self.add_dataset(sl_3v, None, dataset_name="I.16.6", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_2", "+", "X_1", ")", "/", "(", "1", "+", "(", "X_2", "*", "X_1", ")",
                                       "/", "(", "X_0", "^2", ")", ")"], original_equation="v1 = (u+v)/(1+u*v/c^2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.15.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["u-", "X_0", "*", "X_1", "*", "cos", "(", "X_2", ")"],
                         original_equation="E_n = -mom*B*cos(theta)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.27.16", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "X_2", "^2"], original_equation="flux = epsilon*c*Ef^2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_6v, None, dataset_name="I.11.19", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_3", "+", "X_1", "*", "X_4", "+", "X_2", "*", "X_5"],
                         original_equation="A = x1*y1+x2*y2+x3*y3", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.15.3x", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_0", "-", "X_1", "*", "X_3", ")", "/", "sqrt", "(", "1", "-", "X_1", "^2",
                                       "/", "X_2", "^2", ")"], original_equation="x1 = (x-u*t)/sqrt(1-u^2/c^2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(5, 10, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(3, 20, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.10.7", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_2", "^2", ")"],
                         original_equation="m = m_0/sqrt(1-v^2/c^2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 2, uses_negative=False),
                                                                   UniformSampling(3, 10, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_9v, None, dataset_name="I.9.18", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_2", "*", "X_0", "*", "X_1", "/", "(", "(", "X_4", "-", "X_3", ")", "^2", "+",
                                       "(", "X_6", "-", "X_5", ")", "^2", "+", "(", "X_8", "-", "X_7", ")", "^2", ")"],
                         original_equation="F = G*m1*m2/((x2-x1)^2+(y2-y1)^2+(z2-z1)^2)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(3, 4, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(3, 4, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(3, 4, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.15.3t", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_3", "-", "X_2", "*", "X_0", "/", "X_1", "^2", ")", "/", "sqrt", "(", "1",
                                       "-", "X_2", "^2", "/", "X_1", "^2", ")"],
                         original_equation="t1 = (t-u*x/c^2)/sqrt(1-u^2/c^2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(3, 10, uses_negative=False),
                                                                   UniformSampling(1, 2, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_8v, None, dataset_name="II.36.38", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_0", "*", "X_1", ")", "/", "(", "X_2", "*", "X_3", ")", "+", "(", "(",
                                       "X_0", "*", "X_4", ")", "/", "(", "X_5", "*", "X_6", "^2", "*", "X_2", "*",
                                       "X_3", ")", ")", "*", "X_7"],
                         original_equation="f = mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.43.43", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "1", "/", "(", "X_0", "-", "1", ")", ")", "*", "X_1", "*", "X_3", "/",
                                       "X_2"], original_equation="kappa = 1/(gamma-1)*kb*v/A", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(2, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.15.5", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["u-", "X_0", "*", "X_1", "*", "cos", "(", "X_2", ")"],
                         original_equation="E_n = -p_d*Ef*cos(theta)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.37.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "+", "X_1", "+", "2", "*", "sqrt", "(", "X_0", "*", "X_1", ")", "*",
                                       "cos", "(", "X_2", ")"],
                         original_equation="Int = I1+I2+2*sqrt(I1*I2)*cos(delta)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="II.6.11", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "1", "/", "(", "4", "*", "pi", "*", "X_0", ")", ")", "*", "X_1", "*", "cos",
                                       "(", "X_2", ")", "/", "X_3", "^2"],
                         original_equation="Volt = 1/(4*pi*epsilon)*p_d*cos(theta)/r^2", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="III.7.38", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["2", "*", "X_0", "*", "X_1", "/", "(", "X_2", "/", "(", "2", "*", "pi", ")",
                                       ")"], original_equation="omega = 2*mom*B/(h/(2*pi))", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.34.2a", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "/", "(", "2", "*", "pi", "*", "X_2", ")"],
                         original_equation="l = q*v/(2*pi*r)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.13.23", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_2", "^2", ")"],
                         original_equation="rho_c = rho_c_0/sqrt(1-v^2/c^2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 2, uses_negative=False),
                                                                   UniformSampling(3, 10, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="I.29.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "/", "X_1"], original_equation="k = omega/c", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 10, uses_negative=False),
                                   UniformSampling(1, 10, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.38.12", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["4", "*", "pi", "*", "X_3", "*", "(", "X_2", "/", "(", "2", "*", "pi", ")", ")",
                                       "^2", "/", "(", "X_0", "*", "X_1", "^2", ")"],
                         original_equation="r = 4*pi*epsilon*(h/(2*pi))^2/(m*q^2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="III.15.27", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["2", "*", "pi", "*", "X_0", "/", "(", "X_1", "*", "X_2", ")"],
                         original_equation="k = 2*pi*alpha/(n*d)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="I.41.16", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_2", "/", "(", "2", "*", "pi", ")", ")", "*", "X_0", "^3", "/", "(", "pi",
                                       "^2", "*", "X_4", "^2", "*", "(", "exp", "(", "(", "X_2", "/", "(", "2", "*",
                                       "pi", ")", ")", "*", "X_0", "/", "(", "X_3", "*", "X_1", ")", ")", "-", "1", ")",
                                       ")"],
                         original_equation="L_rad = h/(2*pi)*omega^3/(pi^2*c^2*(exp((h/(2*pi))*omega/(kb*T))-1))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.48.20", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_2", "^2", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_2",
                                       "^2", ")"], original_equation="E_n = m*c^2/sqrt(1-v^2/c^2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(3, 10, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="II.11.20", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "^2", "*", "X_2", "/", "(", "3", "*", "X_3", "*", "X_4", ")"],
                         original_equation="Pol = n_rho*p_d^2*Ef/(3*kb*T)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="I.25.13", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "/", "X_1"], original_equation="Volt = q/C", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="III.15.12", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["2", "*", "X_0", "*", "(", "1", "-", "cos", "(", "X_1", "*", "X_2", ")", ")"],
                         original_equation="E_n = 2*U*(1-cos(k*d))", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.24.6", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["0.25", "*", "X_0", "*", "(", "X_1", "^2", "+", "X_2", "^2", ")", "*", "X_3",
                                       "^2"], original_equation="E_n = 1/2*m*(omega^2+omega_0^2)*1/2*x^2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="I.34.27", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_1", "/", "(", "2", "*", "pi", ")", ")", "*", "X_0"],
                         original_equation="E_n =(h/(2*pi))*omega", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.43.31", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_2", "*", "X_1"], original_equation="D = mob*kb*T",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.29.16", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["sqrt", "(", "X_0", "^2", "+", "X_1", "^2", "-", "2", "*", "X_0", "*", "X_1",
                                       "*", "cos", "(", "X_2", "-", "X_3", ")", ")"],
                         original_equation="x = sqrt(x1^2+x2^2-2*x1*x2*cos(theta1-theta2))", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.18.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_0", "*", "X_2", "+", "X_1", "*", "X_3", ")", "/", "(", "X_0", "+", "X_1",
                                       ")"], original_equation="r = (m1*r1+m2*r2)/(m1+m2)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_6v, None, dataset_name="II.6.15a", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_1", "/", "(", "4", "*", "pi", "*", "X_0", ")", ")", "*", "(", "3", "*",
                                       "X_5", "/", "(", "X_2", "^2", "*", "X_2", "^3", ")", ")", "*", "sqrt", "(",
                                       "X_3", "^2", "+", "X_4", "^2", ")"],
                         original_equation="Ef = p_d/(4*pi*epsilon)*3*z/r^5*sqrt(x^2+y^2)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.30.3", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "sin", "(", "X_2", "*", "X_1", "/", "2", ")", "^2", "/", "sin", "(",
                                       "X_1", "/", "2", ")", "^2"],
                         original_equation="Int = Int_0*sin(n*theta/2)^2/sin(theta/2)^2", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_6v, None, dataset_name="III.9.52", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_0", "*", "X_1", "*", "X_2", "/", "(", "X_3", "/", "(", "2", "*", "pi",
                                       ")", ")", ")", "*", "sin", "(", "(", "X_4", "-", "X_5", ")", "*", "X_2", "/",
                                       "2", ")", "^2", "/", "(", "(", "X_4", "-", "X_5", ")", "*", "X_2", "/", "2", ")",
                                       "^2"],
                         original_equation="prob = (p_d*Ef*t/(h/(2*pi)))*sin((omega-omega_0)*t/2)^2/((omega-omega_0)*t/2)^2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.34.2", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "X_2", "/", "2"], original_equation="mom = q*v*r/2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.39.11", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "1", "/", "(", "X_0", "-", "1", ")", ")", "*", "X_1", "*", "X_2"],
                         original_equation="E_n = (1/(gamma-1))*pr*V", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(2, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="II.11.28", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["1", "+", "X_0", "*", "X_1", "/", "(", "1", "-", "(", "X_0", "*", "X_1", "/",
                                       "3", ")", ")"], original_equation="theta = 1+n*alpha/(1-(n*alpha/3))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(0, 1, uses_negative=False),
                                   UniformSampling(0, 1, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="II.3.24", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "/", "(", "4", "*", "pi", "*", "X_1", "^2", ")"],
                         original_equation="flux = Pwr/(4*pi*r^2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.24.17", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["sqrt", "(", "X_0", "^2", "/", "X_1", "^2", "-", "pi", "^2", "/", "X_2", "^2",
                                       ")"], original_equation="k = sqrt(omega^2/c^2-pi^2/d^2)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(4, 6, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(2, 4, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="II.13.17", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "1", "/", "(", "4", "*", "pi", "*", "X_0", "*", "X_1", "^2", ")", ")", "*",
                                       "2", "*", "X_2", "/", "X_3"], original_equation="B = 1/(4*pi*epsilon*c^2)*2*I/r",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="I.12.5", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1"], original_equation="F = q2*Ef", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="II.35.18", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "/", "(", "exp", "(", "X_3", "*", "X_4", "/", "(", "X_1", "*", "X_2", ")",
                                       ")", "+", "exp", "(", "u-", "X_3", "*", "X_4", "/", "(", "X_1", "*", "X_2", ")",
                                       ")", ")"], original_equation="n = n_0/(exp(mom*B/(kb*T))+exp(-mom*B/(kb*T)))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="II.34.11", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "X_2", "/", "(", "2", "*", "X_3", ")"],
                         original_equation="omega = g_*q*B/(2*m)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.34.29a", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "/", "(", "4", "*", "pi", "*", "X_2", ")"],
                         original_equation="E_n = q*h/(4*pi*m)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_6v, None, dataset_name="I.32.17", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "0.5", "*", "X_0", "*", "X_1", "*", "X_2", "^2", ")", "*", "(", "8", "*",
                                       "pi", "*", "X_3", "^2", "/", "3", ")", "*", "(", "(", "X_4", "^2", "*", "X_4",
                                       "^2", ")", "/", "(", "X_4", "^2", "-", "X_5", "^2", ")", "^2", ")"],
                         original_equation="Pwr = (1/2*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(3, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="II.35.21", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "tanh", "(", "X_1", "*", "X_2", "/", "(", "X_3", "*",
                                       "X_4", ")", ")"], original_equation="M = n_rho*mom*tanh(mom*B/(kb*T))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="I.44.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "X_2", "*", "ln", "(", "X_4", "/", "X_3", ")"],
                         original_equation="E_n = n*kb*T*ln(V2/V1)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="III.4.32", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["1", "/", "(", "exp", "(", "(", "X_0", "/", "(", "2", "*", "pi", ")", ")", "*",
                                       "X_1", "/", "(", "X_2", "*", "X_3", ")", ")", "-", "1", ")"],
                         original_equation="n = 1/(exp((h/(2*pi))*omega/(kb*T))-1)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.10.9", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_0", "/", "X_1", ")", "*", "1", "/", "(", "1", "+", "X_2", ")"],
                         original_equation="Ef = sigma_den/epsilon*1/(1+chi)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="II.38.3", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "X_3", "/", "X_2"], original_equation="F = Y*A*x/d",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.6.2b", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["exp", "(", "u-", "(", "(", "(", "X_1", "-", "X_2", ")", "/", "X_0", ")", "^2",
                                       ")", "/", "2", ")", "/", "(", "sqrt", "(", "2", "*", "pi", ")", "*", "X_0", ")"],
                         original_equation="f = exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="II.8.31", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "^2", "/", "2"], original_equation="E_den = epsilon*Ef**2/2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_1v, None, dataset_name="I.6.2a", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["exp", "(", "u-", "X_0", "^2", "/", "2", ")", "/", "sqrt", "(", "2", "*", "pi",
                                       ")"], original_equation="f = exp(-theta**2/2)/sqrt(2*pi)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="III.12.43", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "(", "X_1", "/", "(", "2", "*", "pi", ")", ")"],
                         original_equation="L = n*(h/(2*pi))", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="III.17.37", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "(", "1", "+", "X_1", "*", "cos", "(", "X_2", ")", ")"],
                         original_equation="f = beta*(1+alpha*cos(theta))", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="III.10.19", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "sqrt", "(", "X_1", "^2", "+", "X_2", "^2", "+", "X_3", "^2", ")"],
                         original_equation="E_n = mom*sqrt(Bx**2+By**2+Bz**2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_6v, None, dataset_name="II.11.7", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "(", "1", "+", "X_4", "*", "X_5", "*", "cos", "(", "X_3", ")", "/",
                                       "(", "X_1", "*", "X_2", ")", ")"],
                         original_equation="n = n_0*(1+p_d*Ef*cos(theta)/(kb*T))", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 3, uses_negative=False),
                                                                   UniformSampling(1, 3, uses_negative=False),
                                                                   UniformSampling(1, 3, uses_negative=False),
                                                                   UniformSampling(1, 3, uses_negative=False),
                                                                   UniformSampling(1, 3, uses_negative=False),
                                                                   UniformSampling(1, 3, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="I.39.1", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["1.5", "*", "X_0", "*", "X_1"], original_equation="E_n = 3/2*pr*V",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.37.1", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "(", "1", "+", "X_2", ")", "*", "X_1"],
                         original_equation="E_n = mom*(1+chi)*B", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.12.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_2", "/", "(", "4", "*", "pi", "*", "X_1", "*", "X_2", "^3", ")"],
                         original_equation="Ef = q1*r/(4*pi*epsilon*r**3)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="II.27.18", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "^2"], original_equation="E_den = epsilon*Ef**2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.12.2", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "X_3", "/", "(", "4", "*", "pi", "*", "X_2", "*", "X_3",
                                       "^3", ")"], original_equation="F = q1*q2*r/(4*pi*epsilon*r**3)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="III.13.18", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["2", "*", "X_0", "*", "X_1", "^2", "*", "X_2", "/", "(", "X_3", "/", "(", "2",
                                       "*", "pi", ")", ")"], original_equation="v = 2*E_n*d**2*k/(h/(2*pi))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="II.11.3", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "/", "(", "X_2", "*", "(", "X_3", "^2", "-", "X_4", "^2", ")",
                                       ")"], original_equation="x = q*Ef/(m*(omega_0**2-omega**2))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(3, 5, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_6v, None, dataset_name="I.40.1", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "exp", "(", "u-", "X_1", "*", "X_4", "*", "X_2", "/", "(", "X_5",
                                       "*", "X_3", ")", ")"], original_equation="n = n_0*exp(-m*g*x/(kb*T))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="III.21.20", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["u-", "X_0", "*", "X_1", "*", "X_2", "/", "X_3"],
                         original_equation="j = -rho_c_0*q*A_vec/m", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.43.16", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "X_2", "/", "X_3"],
                         original_equation="v = mu_drift*q*Volt/d", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.15.10", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_2", "^2",
                                       ")"], original_equation="p = m_0*v/sqrt(1-v**2/c**2)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(3, 10, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.30.5", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["arcsin", "(", "X_0", "/", "(", "X_2", "*", "X_1", ")", ")"],
                         original_equation="theta = arcsin(lambd/(n*d))", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 2, uses_negative=False),
                                                                   UniformSampling(2, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.50.26", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "(", "cos", "(", "X_1", "*", "X_2", ")", "+", "X_3", "*", "cos", "(",
                                       "X_1", "*", "X_2", ")", "^2", ")"],
                         original_equation="x = x1*(cos(omega*t)+alpha*cos(omega*t)**2)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="I.12.11", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "(", "X_1", "+", "X_2", "*", "X_3", "*", "sin", "(", "X_4", ")",
                                       ")"], original_equation="F = q*(Ef+B*v*sin(theta))", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="I.6.2", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["exp", "(", "u-", "(", "(", "X_1", "/", "X_0", ")", "^2", ")", "/", "2", ")",
                                       "/", "(", "sqrt", "(", "2", "*", "pi", ")", "*", "X_0", ")"],
                         original_equation="f = exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="I.14.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["0.5", "*", "X_0", "*", "X_1", "^2"], original_equation="U = 1/2*k_spring*x**2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.47.23", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["sqrt", "(", "X_0", "*", "X_1", "/", "X_2", ")"],
                         original_equation="c = sqrt(gamma*pr/rho)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.8.7", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["0.6", "*", "X_0", "^2", "/", "(", "4", "*", "pi", "*", "X_1", "*", "X_2", ")"],
                         original_equation="E_n = 3/5*q**2/(4*pi*epsilon*d)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="III.15.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_0", "/", "(", "2", "*", "pi", ")", ")", "^2", "/", "(", "2", "*", "X_1",
                                       "*", "X_2", "^2", ")"], original_equation="m = (h/(2*pi))**2/(2*E_n*d**2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.34.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "(", "1", "+", "(", "X_1", "/", "X_0", ")", ")", "/", "sqrt", "(", "1", "-",
                                       "X_1", "^2", "/", "X_0", "^2", ")", ")", "*", "X_2"],
                         original_equation="omega = ((1+v/c)/sqrt(1-v**2/c**2))*omega_0", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(3, 10, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="III.8.54", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["sin", "(", "X_0", "*", "X_1", "/", "(", "X_2", "/", "(", "2", "*", "pi", ")",
                                       ")", ")", "^2"], original_equation="prob = sin(E_n*t/(h/(2*pi)))**2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(1, 4, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="I.26.2", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["arcsin", "(", "X_0", "*", "sin", "(", "X_1", ")", ")"],
                         original_equation="theta1 = arcsin(n*sin(theta2))", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(0, 1, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="III.19.51", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "u-", "X_0", "*", "(", "X_1", "^2", "*", "X_1", "^2", ")", "/", "(", "(",
                                       "2", "*", "(", "4", "*", "pi", "*", "X_4", ")", "^2", ")", "*", "(", "X_2", "/",
                                       "(", "2", "*", "pi", ")", ")", "^2", ")", "*", "(", "1", "/", "X_3", "^2", ")",
                                       ")"],
                         original_equation="E_n = -m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="III.4.33", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_0", "/", "(", "2", "*", "pi", ")", ")", "*", "X_1", "/", "(", "exp", "(",
                                       "(", "X_0", "/", "(", "2", "*", "pi", ")", ")", "*", "X_1", "/", "(", "X_2", "*",
                                       "X_3", ")", ")", "-", "1", ")"],
                         original_equation="E_n = (h/(2*pi))*omega/(exp((h/(2*pi))*omega/(kb*T))-1)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.34.1", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_2", "/", "(", "1", "-", "X_1", "/", "X_0", ")"],
                         original_equation="omega = omega_0/(1-v/c)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(3, 10, uses_negative=False),
                                                                   UniformSampling(1, 2, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="II.11.27", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_0", "*", "X_1", "/", "(", "1", "-", "(", "X_0", "*", "X_1", "/", "3",
                                       ")", ")", ")", "*", "X_2", "*", "X_3"],
                         original_equation="Pol = n*alpha/(1-(n*alpha/3))*epsilon*Ef", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(0, 1, uses_negative=False),
                                   UniformSampling(0, 1, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.13.34", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_2", "^2",
                                       ")"], original_equation="j = rho_c_0*v/sqrt(1-v**2/c**2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(3, 10, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.4.23", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "/", "(", "4", "*", "pi", "*", "X_1", "*", "X_2", ")"],
                         original_equation="Volt = q/(4*pi*epsilon*r)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.32.5", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "^2", "*", "X_1", "^2", "/", "(", "6", "*", "pi", "*", "X_2", "*", "X_3",
                                       "^3", ")"], original_equation="Pwr = q**2*a**2/(6*pi*epsilon*c**3)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="I.13.12", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_4", "*", "X_0", "*", "X_1", "*", "(", "1", "/", "X_3", "-", "1", "/", "X_2",
                                       ")"], original_equation="U = G*m1*m2*(1/r2-1/r1)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="II.2.42", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "(", "X_2", "-", "X_1", ")", "*", "X_3", "/", "X_4"],
                         original_equation="Pwr = kappa*(T2-T1)*A/d", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.27.6", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["1", "/", "(", "1", "/", "X_0", "+", "X_2", "/", "X_1", ")"],
                         original_equation="foc = 1/(1/d1+n/d2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="III.14.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "(", "exp", "(", "X_1", "*", "X_2", "/", "(", "X_3", "*", "X_4", ")",
                                       ")", "-", "1", ")"], original_equation="I = I_0*(exp(q*Volt/(kb*T))-1)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.18.12", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "sin", "(", "X_2", ")"],
                         original_equation="tau = r*F*sin(theta)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(0, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.18.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "X_2", "*", "sin", "(", "X_3", ")"],
                         original_equation="L = m*r*v*sin(theta)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="II.21.32", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "/", "(", "4", "*", "pi", "*", "X_1", "*", "X_2", "*", "(", "1", "-",
                                       "X_3", "/", "X_4", ")", ")"],
                         original_equation="Volt = q/(4*pi*epsilon*r*(1-v/c))", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 2, uses_negative=False),
                                                                   UniformSampling(3, 10, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="II.38.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "/", "(", "2", "*", "(", "1", "+", "X_1", ")", ")"],
                         original_equation="mu_S = Y/(2*(1+sigma))", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.34.8", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "X_2", "/", "X_3"], original_equation="omega = q*v*B/p",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.8.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["sqrt", "(", "(", "X_1", "-", "X_0", ")", "^2", "+", "(", "X_3", "-", "X_2", ")",
                                       "^2", ")"], original_equation="d = sqrt((x2-x1)**2+(y2-y1)**2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="II.6.15b", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_1", "/", "(", "4", "*", "pi", "*", "X_0", ")", ")", "*", "3", "*", "cos",
                                       "(", "X_2", ")", "*", "sin", "(", "X_2", ")", "/", "X_3", "^3"],
                         original_equation="E_f = p_d/(4*pi*epsilon)*3*cos(theta)*sin(theta)/r**3",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="I.12.1", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1"], original_equation="F = mu*Nn", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="II.34.29b", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_3", "*", "X_4", "*", "X_2", "/", "(", "X_1", "/", "(", "2", "*",
                                       "pi", ")", ")"], original_equation="E_n = g_*mom*B*Jz/(h/(2*pi))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.13.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["0.5", "*", "X_0", "*", "(", "X_1", "^2", "+", "X_2", "^2", "+", "X_3", "^2",
                                       ")"], original_equation="K = 1/2*m*(v**2+u**2+w**2)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.39.22", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_3", "*", "X_1", "/", "X_2"], original_equation="pr = n*kb*T/V",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.14.3", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "X_2"], original_equation="U = m*g*z",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)], data_source=data_source)

    # fmt: on

    def _ensure_data(self, dataset_name: str) -> None:
        """Download the canonical Feynman archive into the cache once, unless ``force_generate``.

        On a cache miss the whole archive is fetched and every dataset's ``.npz`` extracted, so
        the next ``create_dataset`` reads the same data on any machine. If the download fails we
        warn and let each dataset's ``SampleSource`` regenerate the data locally.
        """
        if self._force_generate:
            return
        from SRToolkit.dataset import data_cache

        cache_path = data_cache.dataset_path(self.benchmark_name, self.version, dataset_name)
        if cache_path.exists():
            return
        try:
            self._archive_source.materialize(cache_path, self.datasets[dataset_name])
        except Exception as e:
            warnings.warn(
                f"[Feynman] Could not download the canonical data ({e}); falling back to local "
                f"sampling. Generated data may differ across machines.",
                stacklevel=2,
            )
