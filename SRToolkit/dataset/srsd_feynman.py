"""
SRSD Feynman symbolic regression benchmark.
"""

import warnings
from typing import Optional

from SRToolkit.utils.symbol_library import SymbolLibrary

from .data_source import SampleSource, UrlSource
from .sampling import IntegerUniformSampling, LogUniformSampling, UniformSampling
from .sr_benchmark import SR_benchmark

_SYMBOL_LIST = [
    "+",
    "-",
    "*",
    "/",
    "u-",
    "sqrt",
    "sin",
    "cos",
    "exp",
    "arcsin",
    "tanh",
    "ln",
    "^2",
    "^3",
    "^4",
    "^5",
    "pi",
    "C",
]


class SRSD_Feynman(SR_benchmark):
    """
    The SRSD Feynman symbolic regression benchmark.

    Contains 120 physics equations from the Feynman Symbolic Regression Dataset with
    per-variable sampling strategies (log-uniform, linear, or integer with sign constraints).
    Data is generated on first instantiation and cached as .npz files for subsequent use.

    References:
        [Matsubara et al. (2024)][cite-srsd-feynman]

    Examples:
        >>> benchmark = SRSD_Feynman()
        >>> len(benchmark.list_datasets(verbose=False))
        120

    Args:
        n_samples: Number of samples to generate per dataset when ``force_generate=True``.
        seed: Random seed used for data generation.
        force_generate: If ``True``, generate fresh data from the stored samplers instead of
            downloading the pre-generated data. Defaults to ``False``.
    """

    GRAVITATIONAL_CONSTANT = 6.67430e-11
    GRAVITATIONAL_ACCELERATION = 9.80665
    SPEED_OF_LIGHT = 2.99792458e8
    ELECTRIC_CONSTANT = 8.854e-12
    PLANCK_CONSTANT = 6.626e-34
    BOLTZMANN_CONSTANT = 1.380649e-23
    BOHR_MAGNETON = 9.2740100783e-24
    DIRAC_CONSTANT = 1.054571817e-34
    ELECTRON_MASS = 9.10938356e-31
    FINE_STRUCTURE_CONSTANT = 7.2973525693e-3

    __data_version__ = "1.0.0"

    def __init__(
        self,
        n_samples: int = 10000,
        seed: Optional[int] = 42,
        force_generate: bool = False,
    ):
        super().__init__("SRSD_Feynman", version="1.0.0")
        self._n_samples = n_samples
        self._seed = seed
        self._force_generate = force_generate
        self._populate()

    def _populate(self):
        # fmt: off
        seed = None
        url = "https://raw.githubusercontent.com/smeznar/SymbolicRegressionToolkit/master/data/srsd.zip"

        # The canonical data is downloaded once from the archive (see _ensure_data) so every
        # machine benchmarks on identical inputs. Each dataset's own data_source is a
        # SampleSource: a transparent, per-dataset fallback that regenerates the data from
        # that dataset's samplers if the download is unavailable (or force_generate is set).
        self._archive_source = UrlSource(url)
        data_source = SampleSource(n_samples=self._n_samples, seed=seed)

        self.metadata = {
            "description": "SRSD Feynman benchmark containing 120 physics equations with per-variable "
            "sampling strategies (log-uniform, linear, or integer with optional sign constraints).",
            "citation": """@article{matsubara2024rethinking,
      title={Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery},
      author={Matsubara, Yoshitomo and Chiba, Naoya and Igarashi, Ryo and Ushiku, Yoshitaka},
      journal={Journal of Data-centric Machine Learning Research},
      year={2024},
      url={https://openreview.net/forum?id=qrUdrXsiXX}
    }
""",
        }



        sl_1v = SymbolLibrary.from_symbol_list(_SYMBOL_LIST, 1)
        sl_2v = SymbolLibrary.from_symbol_list(_SYMBOL_LIST, 2)
        sl_3v = SymbolLibrary.from_symbol_list(_SYMBOL_LIST, 3)
        sl_4v = SymbolLibrary.from_symbol_list(_SYMBOL_LIST, 4)
        sl_5v = SymbolLibrary.from_symbol_list(_SYMBOL_LIST, 5)
        sl_6v = SymbolLibrary.from_symbol_list(_SYMBOL_LIST, 6)
        sl_7v = SymbolLibrary.from_symbol_list(_SYMBOL_LIST, 7)
        sl_8v = SymbolLibrary.from_symbol_list(_SYMBOL_LIST, 8)

        self.add_dataset(sl_2v, None, dataset_name="SRSD I.6.20", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['exp', '(', 'u-', '(', '(', 'X_0', '/', 'X_1', ')', '^2', ')', '/', '2', ')',
                                       '/', '(', 'sqrt', '(', '2', '*', 'pi', ')', '*', 'X_1', ')'],
                         original_equation="exp(-(theta / sigma) ** 2 / 2) / (sqrt(2 * pi) * sigma)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(0.1, 10.0), LogUniformSampling(0.1, 10.0, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_1v, None, dataset_name="SRSD I.6.20a", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['exp', '(', '(', 'u-', '(', 'X_0', '^2', ')', ')', '/', '2', ')', '/', 'sqrt',
                                       '(', '2', '*', 'pi', ')'],
                         original_equation="exp(-theta ** 2 / 2) / sqrt(2 * pi)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[LogUniformSampling(0.1, 10.0)],
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.6.20b", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['exp', '(', 'u-', '(', '(', '(', 'X_0', '-', 'X_1', ')', '/', 'X_2', ')', '^2',
                                       ')', '/', '2', ')', '/', '(', 'sqrt', '(', '2', '*', 'pi', ')', '*', 'X_2', ')'],
                         original_equation="exp(-((theta - theta1) / sigma) ** 2 / 2) / (sqrt(2 * pi) * sigma)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.8.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sqrt', '(', '(', '(', '(', 'X_0', '-', 'X_1', ')', ')', '^2', '+', '(', '(',
                                       'X_2', '-', 'X_3', ')', ')', '^2', ')', ')'],
                         original_equation="sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
            ], data_source=data_source)
        self.add_dataset(sl_8v, None, dataset_name="SRSD I.9.18", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.GRAVITATIONAL_CONSTANT}', '*', 'X_0', '*', 'X_1', '/', '(', '(', 'X_2',
                                       '-', 'X_3', ')', '^2', '+', '(', 'X_4', '-', 'X_5', ')', '^2', '+', '(', 'X_6',
                                       '-', 'X_7', ')', '^2', ')'],
                         original_equation="6.6743e-11 * m1 * m2 / ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1.0, 1000.0, uses_negative=False),
                LogUniformSampling(1.0, 1000.0, uses_negative=False),
                LogUniformSampling(1.0, 10.0),
                LogUniformSampling(1.0, 10.0),
                LogUniformSampling(1.0, 10.0),
                LogUniformSampling(1.0, 10.0),
                LogUniformSampling(1.0, 10.0),
                LogUniformSampling(1.0, 10.0),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.10.7", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', 'sqrt', '(', '1', '-', 'X_1', '^2', '/', f'{self.SPEED_OF_LIGHT}',
                                       '^2', ')'], original_equation="m_0 / sqrt(1 - v ** 2 / 2.99792458e8 ** 2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(100000.0, 100000000.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_6v, None, dataset_name="SRSD I.11.19", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '+', 'X_2', '*', 'X_3', '+', 'X_4', '*', 'X_5'],
                         original_equation="x1 * y1 + x2 * y2 + x3 * y3", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.12.1", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1'], original_equation="mu * Nn", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(0.01, 1.0, uses_negative=False),
                                   LogUniformSampling(0.01, 1.0, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.12.2", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '/', '(', '4', '*', 'pi', '*', f'{self.ELECTRIC_CONSTANT}',
                                       '*', 'X_2', '^2', ')'],
                         original_equation="q1 * q2 * r / (4 * pi * 8.854e-12 * r ** 3)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.12.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', '4', '*', 'pi', '*', f'{self.ELECTRIC_CONSTANT}', '*', 'X_1',
                                       '^2', ')'], original_equation="q1 * r / (4 * pi * 8.854e-12 * r ** 3)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(0.1, 10.0), LogUniformSampling(0.1, 10.0, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.12.5", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1'], original_equation="q2 * Ef", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(0.1, 10.0), LogUniformSampling(0.1, 10.0)],
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="SRSD I.12.11", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', '(', 'X_1', '+', 'X_2', '*', 'X_3', '*', 'sin', '(', 'X_4', ')',
                                       ')'], original_equation="q * (Ef + B * v * sin(theta))", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                UniformSampling(0.0, 1.5707963267948966, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.13.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['0.5', '*', 'X_0', '*', '(', 'X_1', '^2', '+', 'X_2', '^2', '+', 'X_3', '^2',
                                       ')'], original_equation="1 / 2 * m * (v ** 2 + u ** 2 + w ** 2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.13.12", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.GRAVITATIONAL_CONSTANT}', '*', 'X_0', '*', 'X_1', '*', '(', '1', '/',
                                       'X_2', '-', '1', '/', 'X_3', ')'],
                         original_equation="6.67430e-11 * m1 * m2 * (1 / r2 - 1 / r1)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.01, 1.0, uses_negative=False),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.14.3", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.GRAVITATIONAL_ACCELERATION}', '*', 'X_0', '*', 'X_1'],
                         original_equation="9.8066 * m * z", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(0.01, 1.0, uses_negative=False), LogUniformSampling(0.01, 1.0)],
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.14.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['0.5', '*', 'X_0', '*', 'X_1', '^2'],
                         original_equation="1 / 2 * k_spring * x ** 2", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(100.0, 10000.0, uses_negative=False),
                                   LogUniformSampling(0.01, 1.0)], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.15.10", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '/', 'sqrt', '(', '1', '-', 'X_1', '^2', '/',
                                       f'{self.SPEED_OF_LIGHT}', '^2', ')'],
                         original_equation="m_0 * v / sqrt(1 - v ** 2 / 2.99792458e8 ** 2)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(0.01, 1.0, uses_negative=False),
                                   LogUniformSampling(100000.0, 10000000.0)], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.15.3t", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['(', 'X_0', '-', 'X_1', '*', 'X_2', '/', f'{self.SPEED_OF_LIGHT}', '^2', ')',
                                       '/', 'sqrt', '(', '1', '-', 'X_1', '^2', '/', f'{self.SPEED_OF_LIGHT}', '^2',
                                       ')'],
                         original_equation="(t - u * x / c ** 2) / sqrt(1 - u ** 2 / 2.99792458e8 ** 2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-06, 0.0001, uses_negative=False),
                LogUniformSampling(100000.0, 10000000.0),
                LogUniformSampling(1.0, 100.0),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.15.3x", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['(', 'X_0', '-', 'X_1', '*', 'X_2', ')', '/', 'sqrt', '(', '1', '-', 'X_1', '^2',
                                       '/', f'{self.SPEED_OF_LIGHT}', '^2', ')'],
                         original_equation="(x - u * t) / sqrt(1 - u ** 2 / 2.99792458e8 ** 2)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1.0, 100.0),
                LogUniformSampling(1000000.0, 100000000.0),
                LogUniformSampling(1e-06, 0.0001, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.16.6", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['(', 'X_0', '+', 'X_1', ')', '/', '(', '1', '+', 'X_0', '*', 'X_1', '/',
                                       f'{self.SPEED_OF_LIGHT}', '^2', ')'],
                         original_equation="(u + v) / (1 + u * v / 2.99792458e8 ** 2)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(1000000.0, 100000000.0),
                                   LogUniformSampling(1000000.0, 100000000.0)], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.18.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['(', 'X_0', '*', 'X_1', '+', 'X_2', '*', 'X_3', ')', '/', '(', 'X_0', '+', 'X_2',
                                       ')'], original_equation="(m1 * r1 + m2 * r2) / (m1 + m2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.18.12", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '*', 'sin', '(', 'X_2', ')'],
                         original_equation="r * F * sin(theta)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.18.16", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '*', 'X_2', '*', 'sin', '(', 'X_3', ')'],
                         original_equation="m * r * v * sin(theta)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.24.6", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['0.25', '*', 'X_0', '*', '(', 'X_1', '^2', '+', 'X_2', '^2', ')', '*', 'X_3',
                                       '^2'],
                         original_equation="1 / 2 * m * (omega ** 2 + omega_0 ** 2) * 1 / 2 * x ** 2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.25.13", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', 'X_1'], original_equation="q / C", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[LogUniformSampling(1e-05, 0.001),
                                                                   LogUniformSampling(1e-05, 0.001,
                                                                                      uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.26.2", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sin', '(', 'X_0', ')', '/', 'sin', '(', 'X_1', ')'],
                         original_equation="sin(theta1) / sin(theta2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                UniformSampling(0, 1.5707963267948966, uses_negative=False),
                UniformSampling(0, 1.5707963267948966, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.27.6", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['1', '/', '(', '1', '/', 'X_0', '+', 'X_1', '/', 'X_2', ')'],
                         original_equation="1 / (1 / d1 + n / d2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.001, 0.1, uses_negative=False),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.001, 0.1, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_1v, None, dataset_name="SRSD I.29.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', f'{self.SPEED_OF_LIGHT}'], original_equation="omega / 2.99792458e8",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(1000000000.0, 100000000000.0, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.29.16", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sqrt', '(', 'X_0', '^2', '+', 'X_1', '^2', '+', '2', '*', 'X_0', '*', 'X_1',
                                       '*', 'cos', '(', 'X_2', '-', 'X_3', ')', ')'],
                         original_equation="sqrt(x1 ** 2 + x2 ** 2 + 2 * x1 * x2 * cos(theta1 - theta2))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.30.3", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'sin', '(', 'X_1', '*', 'X_2', '/', '2', ')', '^2', '/', 'sin', '(',
                                       'X_2', '/', '2', ')', '^2'],
                         original_equation="Int_0 * sin(n * theta / 2) ** 2 / sin(theta / 2) ** 2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
                IntegerUniformSampling(10, 1000, uses_negative=False),
                UniformSampling(-6.283185307179586, 6.283185307179586, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.30.5", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', 'X_1', '*', 'sin', '(', 'X_2', ')', ')'],
                         original_equation="lambda / (n * sin(theta))", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-11, 1e-09, uses_negative=False),
                IntegerUniformSampling(1, 100, uses_negative=False),
                UniformSampling(0, 1.5707963267948966, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.32.5", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '^2', '*', 'X_1', '^2', '/', '(', '6', '*', 'pi', '*',
                                       f'{self.ELECTRIC_CONSTANT}', '*', f'{self.SPEED_OF_LIGHT}', '^3', ')'],
                         original_equation="q ** 2 * a ** 2 / (6 * pi * 8.854e-12 * 2.99792458e8 ** 3)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(0.001, 0.1),
                                   LogUniformSampling(100000.0, 10000000.0, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.32.17", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.ELECTRIC_CONSTANT}', '*', f'{self.SPEED_OF_LIGHT}', '*', 'X_0', '^2',
                                       '*', '8', '*', 'pi', '*', 'X_1', '^2', '/', '3', '*', 'X_2', '^4', '/', '(', '(',
                                       'X_2', '^2', '-', 'X_3', '^2', ')', '^2', ')', '/', '2'],
                         original_equation="(1 / 2 * 8.854e-12 * 2.99792458e8 * Ef ** 2) * (8 * pi * r ** 2 / 3) * (omega ** 4 / (omega ** 2 - omega_0 ** 2) ** 2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(10.0, 1000.0),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
                LogUniformSampling(1000000000.0, 100000000000.0),
                LogUniformSampling(1000000000.0, 100000000000.0),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.34.10", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', '1', '-', 'X_1', '/', f'{self.SPEED_OF_LIGHT}', ')'],
                         original_equation="omega_0 / (1 - v / 2.99792458e8)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1000000000.0, 100000000000.0, uses_negative=False),
                LogUniformSampling(100000.0, 10000000.0),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.34.8", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '*', 'X_2', '/', 'X_3'], original_equation="q * v * B / p",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-11, 1e-09),
                LogUniformSampling(100000.0, 10000000.0),
                LogUniformSampling(10.0, 1000.0),
                LogUniformSampling(1000000000.0, 100000000000.0),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.34.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['(', '1', '+', 'X_0', '/', f'{self.SPEED_OF_LIGHT}', ')', '*', 'X_1', '/',
                                       'sqrt', '(', '1', '-', 'X_0', '^2', '/', f'{self.SPEED_OF_LIGHT}', '^2', ')'],
                         original_equation="(1 + v / 2.99792458e8) / sqrt(1 - v ** 2 / 2.99792458e8 ** 2) * omega_0",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1000000.0, 100000000.0),
                LogUniformSampling(1000000000.0, 100000000000.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_1v, None, dataset_name="SRSD I.34.27", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.DIRAC_CONSTANT}', '*', 'X_0'],
                         original_equation="(6.626e-34 / (2 * pi)) * omega", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(1000000000.0, 100000000000.0, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.37.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '+', 'X_1', '+', '2', '*', 'sqrt', '(', 'X_0', '*', 'X_1', ')', '*',
                                       'cos', '(', 'X_2', ')'],
                         original_equation="I1 + I2 + 2 * sqrt(I1 * I2) * cos(delta)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.001, 0.1, uses_negative=False),
                LogUniformSampling(0.001, 0.1, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.38.12", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['4', '*', 'pi', '*', f'{self.ELECTRIC_CONSTANT}', '*', f'{self.DIRAC_CONSTANT}',
                                       '^2', '/', '(', 'X_0', '*', 'X_1', '^2', ')'],
                         original_equation="4 * pi * 8.854e-12 * (6.626e-34 / (2 * pi)) ** 2 / (m * q ** 2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-28, 1e-26, uses_negative=False),
                LogUniformSampling(1e-11, 1e-09, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.39.10", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['1.5', '*', 'X_0', '*', 'X_1'], original_equation="3 / 2 * pr * V",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(10000.0, 1000000.0, uses_negative=False),
                LogUniformSampling(1e-05, 0.001, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.39.11", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_1', '*', 'X_2', '/', '(', 'X_0', '-', '1', ')'],
                         original_equation="1 / (gamma - 1) * pr * V", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                UniformSampling(1, 2, uses_negative=False),
                LogUniformSampling(10000.0, 1000000.0, uses_negative=False),
                LogUniformSampling(1e-05, 0.001, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.39.22", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.BOLTZMANN_CONSTANT}', '*', 'X_0', '*', 'X_1', '/', 'X_2'],
                         original_equation="n * 1.380649e-23 * T / V", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
                LogUniformSampling(1e-05, 0.001, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.40.1", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'exp', '(', 'u-', f'{self.GRAVITATIONAL_ACCELERATION}', '*', 'X_1',
                                       '*', 'X_2', '/', '(', f'{self.BOLTZMANN_CONSTANT}', '*', 'X_3', ')', ')'],
                         original_equation="n_0 * exp(-m * 9.80665 * x / (1.380649e-23 * T))", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e25, 1e27, uses_negative=False),
                LogUniformSampling(1e-24, 1e-22, uses_negative=False),
                LogUniformSampling(0.01, 1.0),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.41.16", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.DIRAC_CONSTANT}', '*', 'X_0', '^3', '/', '(', 'pi', '^2', '*',
                                       f'{self.SPEED_OF_LIGHT}', '^2', '*', '(', 'exp', '(', f'{self.DIRAC_CONSTANT}',
                                       '*', 'X_0', '/', '(', f'{self.BOLTZMANN_CONSTANT}', '*', 'X_1', ')', ')', '-',
                                       '1', ')', ')'],
                         original_equation="6.626e-34 / (2 * pi) * omega ** 3 / (pi ** 2 * 2.99792458e8 ** 2 * (exp((6.626e-34 / (2 * pi)) * omega / (1.380649e-23 * T)) - 1))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.43.16", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '*', 'X_2', '/', 'X_3'],
                         original_equation="mu_drift * q * Volt / d", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-06, 0.0001),
                LogUniformSampling(1e-11, 1e-09),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.001, 0.1, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.43.31", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.BOLTZMANN_CONSTANT}', '*', 'X_0', '*', 'X_1'],
                         original_equation="mob * 1.380649e-23 * T", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(10000000000000.0, 1000000000000000.0, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.43.43", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.BOLTZMANN_CONSTANT}', '*', 'X_1', '/', '(', '(', 'X_0', '-', '1', ')',
                                       '*', 'X_2', ')'], original_equation="1 / (gamma - 1) * 1.380649e-23 * v / A",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                UniformSampling(1, 2, uses_negative=False),
                LogUniformSampling(100.0, 10000.0, uses_negative=False),
                LogUniformSampling(1e-21, 1e-19, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.44.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.BOLTZMANN_CONSTANT}', '*', 'X_0', '*', 'X_1', '*', 'ln', '(', 'X_2', '/',
                                       'X_3', ')'], original_equation="n * 1.380649e-23 * T * ln(V2 / V1)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e25, 1e27, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
                LogUniformSampling(1e-05, 0.001, uses_negative=False),
                LogUniformSampling(1e-05, 0.001, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.47.23", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sqrt', '(', 'X_0', '*', 'X_1', '/', 'X_2', ')'],
                         original_equation="sqrt(gamma * pr / rho)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                UniformSampling(1, 2, uses_negative=False),
                UniformSampling(5e-06, 1.5e-05, uses_negative=False),
                UniformSampling(1, 2, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.48.2", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', f'{self.SPEED_OF_LIGHT}', '^2', '/', 'sqrt', '(', '1', '-', 'X_1',
                                       '^2', '/', f'{self.SPEED_OF_LIGHT}', '^2', ')'],
                         original_equation="m * 2.99792458e8 ** 2 / sqrt(1 - v ** 2 / 2.99792458e8 ** 2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-29, 1e-27, uses_negative=False),
                LogUniformSampling(1000000.0, 100000000.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.50.26", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', '(', 'cos', '(', 'X_1', '*', 'X_2', ')', '+', 'X_3', '*', 'cos', '(',
                                       'X_1', '*', 'X_2', ')', '^2', ')'],
                         original_equation="x1 * (cos(omega * t) + alpha * cos(omega * t) ** 2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(10.0, 1000.0),
                LogUniformSampling(0.001, 0.1, uses_negative=False),
                LogUniformSampling(0.001, 0.1),
            ], data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="SRSD II.2.42", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', '(', 'X_1', '-', 'X_2', ')', '*', 'X_3', '/', 'X_4'],
                         original_equation="kappa * (T2 - T1) * A / d", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
                LogUniformSampling(0.0001, 0.01, uses_negative=False),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD II.3.24", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', '4', '*', 'pi', '*', 'X_1', '^2', ')'],
                         original_equation="Pwr / (4 * pi * r ** 2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(1.0, 100.0), LogUniformSampling(0.01, 1.0, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD II.4.23", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', '4', '*', 'pi', '*', f'{self.ELECTRIC_CONSTANT}', '*', 'X_1',
                                       ')'], original_equation="q / (4 * pi * 8.854e-12 * r)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(0.001, 0.1), LogUniformSampling(0.01, 1.0, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD II.6.11", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'cos', '(', 'X_1', ')', '/', '(', '4', '*', 'pi', '*',
                                       f'{self.ELECTRIC_CONSTANT}', '*', 'X_2', '^2', ')'],
                         original_equation="1 / (4 * pi * 8.854e-12) * p_d * cos(theta) / r ** 2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-22, 1e-20),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
                LogUniformSampling(1e-10, 1e-08, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="SRSD II.6.15a", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', '4', '*', 'pi', '*', f'{self.ELECTRIC_CONSTANT}', ')', '*', '3',
                                       '*', 'X_1', '/', '(', 'X_2', '^5', ')', '*', 'sqrt', '(', 'X_3', '^2', '+',
                                       'X_4', '^2', ')'],
                         original_equation="p_d / (4 * pi * 8.854e-12) * 3 * z / r ** 5 * sqrt(x ** 2 + y ** 2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-22, 1e-20),
                LogUniformSampling(1e-10, 1e-08),
                LogUniformSampling(1e-10, 1e-08, uses_negative=False),
                LogUniformSampling(1e-10, 1e-08),
                LogUniformSampling(1e-10, 1e-08),
            ], data_source=data_source)

        self.add_dataset(sl_3v, None, dataset_name="SRSD II.6.15b", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', '4', '*', 'pi', '*', f'{self.ELECTRIC_CONSTANT}', ')', '*', '3',
                                       '*', 'cos', '(', 'X_1', ')', '*', 'sin', '(', 'X_1', ')', '/', '(', 'X_2', '^3',
                                       ')'],
                         original_equation="p_d / (4 * pi * 8.854e-12) * 3 * cos(theta) * sin(theta) / r ** 3",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-22, 1e-20),
                UniformSampling(0, 3.141592653589793, uses_negative=False),
                LogUniformSampling(1e-10, 1e-08, uses_negative=False),
            ], data_source=data_source)

        self.add_dataset(sl_2v, None, dataset_name="SRSD II.8.7", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['3', '*', 'X_0', '^2', '/', '(', '5', '*', '4', '*', 'pi', '*',
                                       f'{self.ELECTRIC_CONSTANT}', '*', 'X_1', ')'],
                         original_equation="3 / 5 * q ** 2 / (4 * pi * 8.854e-12 * d)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[LogUniformSampling(1e-11, 1e-09),
                                                                              LogUniformSampling(1e-12, 1e-10,
                                                                                                 uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_1v, None, dataset_name="SRSD II.8.31", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.ELECTRIC_CONSTANT}', '*', 'X_0', '^2', '/', '2'],
                         original_equation="8.854e-12 * Ef ** 2 / 2", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(10.0, 1000.0, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD II.10.9", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', f'{self.ELECTRIC_CONSTANT}', '*', '(', '1', '+', 'X_1', ')',
                                       ')'], original_equation="sigma_den / 8.854e-12 * 1 / (1 + chi)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(0.001, 0.1), LogUniformSampling(1.0, 100.0, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="SRSD II.11.3", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '/', '(', 'X_2', '*', '(', 'X_3', '^2', '-', 'X_4', '^2', ')',
                                       ')'], original_equation="q * Ef / (m * (omega_0 ** 2 - omega ** 2))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-11, 1e-09),
                LogUniformSampling(1e-09, 1e-07, uses_negative=False),
                LogUniformSampling(1e-28, 1e-26, uses_negative=False),
                LogUniformSampling(1000000000.0, 100000000000.0),
                LogUniformSampling(1000000000.0, 100000000000.0),
            ], data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="SRSD II.11.17", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', '(', '1', '+', 'X_1', '*', 'X_2', '*', 'cos', '(', 'X_3', ')', '/',
                                       '(', f'{self.BOLTZMANN_CONSTANT}', '*', 'X_4', ')', ')'],
                         original_equation="n_0 * (1 + p_d * Ef * cos(theta) / (1.380649e-23 * T))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e27, 1e29, uses_negative=False),
                LogUniformSampling(1e-22, 1e-20),
                LogUniformSampling(10.0, 1000.0),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD II.11.20", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '^2', '*', 'X_2', '/', '(', '3', '*',
                                       f'{self.BOLTZMANN_CONSTANT}', '*', 'X_3', ')'],
                         original_equation="n_rho * p_d ** 2 * Ef / (3 * 1.380649e-23 * T)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(1e-22, 1e-20),
                LogUniformSampling(10.0, 1000.0),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD II.11.27", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.ELECTRIC_CONSTANT}', '*', 'X_0', '*', 'X_1', '*', 'X_2', '/', '(', '1',
                                       '-', 'X_0', '*', 'X_1', '/', '3', ')'],
                         original_equation="n * alpha / (1 - (n * alpha / 3)) * 8.854e-12 * Ef", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(1e-33, 1e-31, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD II.11.28", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['1', '+', 'X_0', '*', 'X_1', '/', '(', '1', '-', 'X_0', '*', 'X_1', '/', '3',
                                       ')'], original_equation="1 + n * alpha / (1 - (n * alpha / 3))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(1e-33, 1e-31, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD II.13.17", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['2', '*', 'X_0', '/', '(', '4', '*', 'pi', '*', f'{self.ELECTRIC_CONSTANT}', '*',
                                       f'{self.SPEED_OF_LIGHT}', '^2', '*', 'X_1', ')'],
                         original_equation="1 / (4 * pi * 8.854e-12 * 2.99792458e8 ** 2) * 2 * I / r",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(0.001, 0.1), LogUniformSampling(0.001, 0.1, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD II.13.23", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', 'sqrt', '(', '1', '-', 'X_1', '^2', '/', f'{self.SPEED_OF_LIGHT}',
                                       '^2', ')'], original_equation="rho_c_0 / sqrt(1 - v ** 2 / 2.99792458e8 ** 2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e27, 1e29, uses_negative=False),
                LogUniformSampling(1000000.0, 100000000.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD II.13.34", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '/', 'sqrt', '(', '1', '-', 'X_1', '^2', '/',
                                       f'{self.SPEED_OF_LIGHT}', '^2', ')'],
                         original_equation="rho_c_0 * v / sqrt(1 - v ** 2 / 2.99792458e8 ** 2)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e27, 1e29, uses_negative=False),
                LogUniformSampling(1000000.0, 100000000.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD II.15.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['u-', 'X_0', '*', 'X_1', '*', 'cos', '(', 'X_2', ')'],
                         original_equation="-mom * B * cos(theta)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-25, 1e-23),
                LogUniformSampling(0.001, 0.1),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD II.15.5", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['u-', 'X_0', '*', 'X_1', '*', 'cos', '(', 'X_2', ')'],
                         original_equation="-p_d * Ef * cos(theta)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-22, 1e-20),
                LogUniformSampling(10.0, 1000.0),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD II.21.32", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', '4', '*', 'pi', '*', f'{self.ELECTRIC_CONSTANT}', '*', 'X_1',
                                       '*', '(', '1', '-', 'X_2', '/', f'{self.SPEED_OF_LIGHT}', ')', ')'],
                         original_equation="q / (4 * pi * 8.854e-12 * r * (1 - v / 2.99792458e8))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.001, 0.1),
                LogUniformSampling(1.0, 100.0, uses_negative=False),
                LogUniformSampling(1000000.0, 100000000.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD II.24.17", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sqrt', '(', 'X_0', '^2', '/', f'{self.SPEED_OF_LIGHT}', '^2', '-', 'pi', '^2',
                                       '/', 'X_1', '^2', ')'],
                         original_equation="sqrt(omega ** 2 / 2.99792458e8 ** 2 - pi ** 2 / d ** 2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(1000000000.0, 100000000000.0),
                                   LogUniformSampling(0.001, 0.1, uses_negative=False)], data_source=data_source)
        self.add_dataset(sl_1v, None, dataset_name="SRSD II.27.16", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.ELECTRIC_CONSTANT}', '*', f'{self.SPEED_OF_LIGHT}', '*', 'X_0', '^2'],
                         original_equation="8.854e-12 * 2.99792458e8 * Ef ** 2", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[LogUniformSampling(0.1, 10.0, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_1v, None, dataset_name="SRSD II.27.18", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.ELECTRIC_CONSTANT}', '*', 'X_0', '^2'],
                         original_equation="8.854e-12 * Ef ** 2", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[LogUniformSampling(0.1, 10.0, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD II.34.2a", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '/', '(', '2', '*', 'pi', '*', 'X_2', ')'],
                         original_equation="q * v / (2 * pi * r)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-11, 1e-09),
                LogUniformSampling(100000.0, 10000000.0),
                LogUniformSampling(1e-11, 1e-09, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD II.34.2", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '*', 'X_2', '/', '2'], original_equation="q * v * r / 2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-11, 1e-09),
                LogUniformSampling(100000.0, 10000000.0),
                LogUniformSampling(1e-11, 1e-09, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD II.34.11", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '*', 'X_2', '/', '(', '2', '*', 'X_3', ')'],
                         original_equation="g_ * q * B / (2 * m)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                UniformSampling(-1.0, 1.0, uses_negative=False),
                LogUniformSampling(1e-11, 1e-09),
                LogUniformSampling(1e-09, 1e-07),
                LogUniformSampling(1e-30, 1e-28, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD II.34.29a", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', f'{self.PLANCK_CONSTANT}', '/', '(', '4', '*', 'pi', '*', 'X_1',
                                       ')'], original_equation="q * 6.626e-34 / (4 * pi * m)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[LogUniformSampling(1e-11, 1e-09),
                                                                              LogUniformSampling(1e-30, 1e-28,
                                                                                                 uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD II.34.29b", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', f'{self.BOHR_MAGNETON}', '*', 'X_1', '*', 'X_2', '*', '2', '*', 'pi',
                                       '/', f'{self.PLANCK_CONSTANT}'],
                         original_equation="g_ * 9.2740100783e-24 * B * Jz / (6.626e-34 / (2 * pi))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                UniformSampling(-1.0, 1.0, uses_negative=False),
                LogUniformSampling(0.001, 0.1),
                LogUniformSampling(1e-26, 1e-22),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD II.35.18", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', 'exp', '(', 'X_1', '*', 'X_2', '/', '(',
                                       f'{self.BOLTZMANN_CONSTANT}', '*', 'X_3', ')', ')', '+', 'exp', '(', 'u-', 'X_1',
                                       '*', 'X_2', '/', '(', f'{self.BOLTZMANN_CONSTANT}', '*', 'X_3', ')', ')', ')'],
                         original_equation="n_0 / (exp(mom * B / (1.380649e-23 * T)) + exp(-mom * B / (1.380649e-23 * T)))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(1e-25, 1e-23, uses_negative=False),
                LogUniformSampling(0.001, 0.1, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD II.35.21", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '*', 'tanh', '(', 'X_1', '*', 'X_2', '/', '(',
                                       f'{self.BOLTZMANN_CONSTANT}', '*', 'X_3', ')', ')'],
                         original_equation="n_rho * mom * tanh(mom * B / (1.380649e-23 * T))", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(1e-25, 1e-23, uses_negative=False),
                LogUniformSampling(0.001, 0.1, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="SRSD II.36.38", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '/', '(', f'{self.BOLTZMANN_CONSTANT}', '*', 'X_2', ')', '+',
                                       'X_0', '*', 'X_3', '/', '(', f'{self.ELECTRIC_CONSTANT}', '*',
                                       f'{self.SPEED_OF_LIGHT}', '^2', '*', f'{self.BOLTZMANN_CONSTANT}', '*', 'X_2',
                                       ')', '*', 'X_4'],
                         original_equation="mom * H / (1.380649e-23 * T) + (mom * alpha) / (8.854e-12 * 2.99792458e8 ** 2 * 1.380649e-23 * T) * M",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-25, 1e-23),
                LogUniformSampling(0.001, 0.1),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
                UniformSampling(0, 1, uses_negative=False),
                LogUniformSampling(1e23, 1e25, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD II.37.1", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', '(', '1', '+', 'X_1', ')', '*', 'X_2'],
                         original_equation="mom * (1 + chi) * B", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(1e-25, 1e-23), LogUniformSampling(10000.0, 1000000.0),
                                   LogUniformSampling(0.001, 0.1)], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD II.38.3", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '*', 'X_2', '/', 'X_3'], original_equation="Y * A * x / d",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.0001, 0.01, uses_negative=False),
                LogUniformSampling(0.001, 0.1),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD II.38.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', '2', '*', '(', '1', '+', 'X_1', ')', ')'],
                         original_equation="Y / (2 * (1 + sigma))", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[LogUniformSampling(0.1, 10.0, uses_negative=False),
                                                                   LogUniformSampling(0.01, 1.0, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD III.4.32", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['1', '/', '(', 'exp', '(', '(', f'{self.PLANCK_CONSTANT}', '/', '(', '2', '*',
                                       'pi', ')', ')', '*', 'X_0', '/', '(', f'{self.BOLTZMANN_CONSTANT}', '*', 'X_1',
                                       ')', ')', '-', '1', ')'],
                         original_equation="1 / (exp((6.626e-34 / (2 * pi)) * omega / (1.380649e-23 * T)) - 1)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD III.4.33", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.DIRAC_CONSTANT}', '*', 'X_0', '/', '(', 'exp', '(',
                                       f'{self.DIRAC_CONSTANT}', '*', 'X_0', '/', '(', f'{self.BOLTZMANN_CONSTANT}',
                                       '*', 'X_1', ')', ')', '-', '1', ')'],
                         original_equation="(6.626e-34 / (2 * pi)) * omega / (exp((6.626e-34 / (2 * pi)) * omega / (1.380649e-23 * T)) - 1)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD III.7.38", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['2', '*', 'X_0', '*', 'X_1', '/', '(', f'{self.PLANCK_CONSTANT}', '/', '(', '2',
                                       '*', 'pi', ')', ')'], original_equation="2 * mom * B / (6.626e-34 / (2 * pi))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(1e-11, 1e-09), LogUniformSampling(0.001, 0.1)],
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD III.8.54", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sin', '(', 'X_0', '*', 'X_1', '/', '(', f'{self.PLANCK_CONSTANT}', '/', '(',
                                       '2', '*', 'pi', ')', ')', ')', '^2'],
                         original_equation="sin(E_n * t / (6.626e-34 / (2 * pi))) ** 2", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[LogUniformSampling(1e-18, 1e-16),
                                                                              LogUniformSampling(1e-18, 1e-16,
                                                                                                 uses_negative=False)],
                         data_source=data_source)

        self.add_dataset(sl_5v, None, dataset_name="SRSD III.9.52", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '^2', '*', 'X_1', '^2', '*', 'X_2', '^2', '/', '(',
                                       f'{self.PLANCK_CONSTANT}', '/', '(', '2', '*', 'pi', ')', ')', '^2', '*', 'sin',
                                       '(', '(', 'X_3', '-', 'X_4', ')', '*', 'X_2', '/', '2', ')', '^2', '/', '(', '(',
                                       'X_3', '-', 'X_4', ')', '*', 'X_2', '/', '2', ')', '^2'],
                         original_equation="(p_d * Ef * t / (6.626e-34 / (2 * pi))) ** 2 * sin((omega - omega_0) * t / 2) ** 2 / ((omega - omega_0) * t / 2) ** 2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-22, 1e-20),
                LogUniformSampling(10.0, 1000.0),
                LogUniformSampling(1e-18, 1e-16, uses_negative=False),
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD III.10.19", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'sqrt', '(', 'X_1', '^2', '+', 'X_2', '^2', '+', 'X_3', '^2', ')'],
                         original_equation="mom * sqrt(Bx ** 2 + By ** 2 + Bz ** 2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-25, 1e-23),
                LogUniformSampling(0.001, 0.1),
                LogUniformSampling(0.001, 0.1),
                LogUniformSampling(0.001, 0.1),
            ], data_source=data_source)
        self.add_dataset(sl_1v, None, dataset_name="SRSD III.12.43", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', f'{self.DIRAC_CONSTANT}'],
                         original_equation="n * (6.626e-34 / (2 * pi))", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[IntegerUniformSampling(1, 100, uses_negative=False)],
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD III.13.18", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['2', '*', 'X_0', '*', 'X_1', '^2', '*', 'X_2', '/', '(',
                                       f'{self.PLANCK_CONSTANT}', '/', '(', '2', '*', 'pi', ')', ')'],
                         original_equation="2 * E_n * d ** 2 * k / (6.626e-34 / (2 * pi))", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-18, 1e-16),
                LogUniformSampling(1e-10, 1e-08, uses_negative=False),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD III.14.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', '(', 'exp', '(', 'X_1', '*', 'X_2', '/', '(',
                                       f'{self.BOLTZMANN_CONSTANT}', '*', 'X_3', ')', ')', '-', '1', ')'],
                         original_equation="I_0 * (exp(q * Volt / (1.380649e-23 * T)) - 1)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.001, 0.1),
                LogUniformSampling(1e-22, 1e-20, uses_negative=False),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD III.15.12", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['2', '*', 'X_0', '*', '(', '1', '-', 'cos', '(', 'X_1', '*', 'X_2', ')', ')'],
                         original_equation="2 * U * (1 - cos(k * d))", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-18, 1e-16, uses_negative=False),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(1e-10, 1e-08, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD III.15.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.DIRAC_CONSTANT}', '^2', '/', '(', '2', '*', 'X_0', '*', 'X_1', '^2',
                                       ')'], original_equation="(6.626e-34 / (2 * pi)) ** 2 / (2 * E_n * d ** 2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-18, 1e-16, uses_negative=False),
                LogUniformSampling(1e-10, 1e-08, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD III.15.27", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['2', '*', 'pi', '*', 'X_0', '/', '(', 'X_1', '*', 'X_2', ')'],
                         original_equation="2 * pi * alpha / (n * d)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                IntegerUniformSampling(1, 100),
                IntegerUniformSampling(1, 100, uses_negative=False),
                LogUniformSampling(1e-10, 1e-08, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD III.17.37", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', '(', '1', '+', 'X_1', '*', 'cos', '(', 'X_2', ')', ')'],
                         original_equation="beta * (1 + alpha * cos(theta))", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-18, 1e-16, uses_negative=False),
                LogUniformSampling(1e-18, 1e-16),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD III.19.51", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['u-', 'X_0', '*', 'X_1', '^4', '/', '(', '2', '*', '(', '4', '*', 'pi', '*',
                                       f'{self.ELECTRIC_CONSTANT}', ')', '^2', '*', '(', f'{self.PLANCK_CONSTANT}', '/',
                                       '(', '2', '*', 'pi', ')', ')', '^2', ')', '*', '(', '1', '/', 'X_2', '^2', ')'],
                         original_equation="-m * q ** 4 / (2 * (4 * pi * 8.854e-12) ** 2 * (6.626e-34 / (2 * pi)) ** 2) * (1 / n ** 2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-30, 1e-28, uses_negative=False),
                LogUniformSampling(1e-11, 1e-09),
                IntegerUniformSampling(1, 100, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD III.21.20", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['u-', 'X_0', '*', 'X_1', '*', 'X_2', '/', 'X_3'],
                         original_equation="-rho_c_0 * q * A_vec / m", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-29, 1e-27, uses_positive=False),
                LogUniformSampling(1e-11, 1e-09, uses_positive=False),
                LogUniformSampling(0.001, 0.1),
                LogUniformSampling(1e-30, 1e-28, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD Bonus 1", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['(', 'X_0', '*', 'X_1', '*', f'{self.FINE_STRUCTURE_CONSTANT}', '*',
                                       f'{self.DIRAC_CONSTANT}', '*', f'{self.SPEED_OF_LIGHT}', '/', '(', '4', '*',
                                       'X_2', '*', 'sin', '(', 'X_3', '/', '2', ')', '^2', ')', ')', '^2'],
                         original_equation="(Z_1 * Z_2 * alpha * 1.054571817e-34 * 2.99792458e8 / (4 * E_n * sin(theta / 2) ** 2)) ** 2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                IntegerUniformSampling(1, 10, uses_negative=False),
                IntegerUniformSampling(1, 10, uses_negative=False),
                LogUniformSampling(1e-18, 1e-16, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_6v, None, dataset_name="SRSD Bonus 2", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '/', 'X_2', '^2', '*', '(', '1', '+', 'sqrt', '(', '1', '+',
                                       '2', '*', 'X_3', '*', 'X_2', '^2', '/', '(', 'X_0', '*', 'X_1', '^2', ')', ')',
                                       '*', 'cos', '(', 'X_4', '-', 'X_5', ')', ')'],
                         original_equation="m * k_G / L ** 2 * (1 + sqrt(1 + 2 * E_n * L ** 2 / (m * k_G ** 2)) * cos(theta1 - theta2))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(1000000000.0, 100000000000.0, uses_negative=False),
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
                LogUniformSampling(1e25, 1e27, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD Bonus 3", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', '(', '1', '-', 'X_1', '^2', ')', '/', '(', '1', '+', 'X_1', '*',
                                       'cos', '(', 'X_2', '-', 'X_3', ')', ')'],
                         original_equation="d * (1 - alpha ** 2) / (1 + alpha * cos(theta1 - theta2))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
                UniformSampling(0.0, 1.0, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="SRSD Bonus 4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sqrt', '(', '2', '/', 'X_0', '*', '(', 'X_1', '-', 'X_2', '-', 'X_3', '^2', '/',
                                       '(', '2', '*', 'X_0', '*', 'X_4', '^2', ')', ')', ')'],
                         original_equation="sqrt(2 / m * (E_n - U - L ** 2 / (2 * m * r ** 2)))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(1e25, 1e27, uses_negative=False),
                LogUniformSampling(1e25, 1e27, uses_negative=False),
                LogUniformSampling(100000000.0, 10000000000.0),
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD Bonus 5", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['2', '*', 'pi', '*', 'sqrt', '(', 'X_0', '^3', ')', '/', 'sqrt', '(',
                                       f'{self.GRAVITATIONAL_CONSTANT}', '*', '(', 'X_1', '+', 'X_2', ')', ')'],
                         original_equation="2 * pi * d ** (3 / 2) / sqrt(6.67430e-11 * (m1 + m2))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(1e23, 1e25, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_7v, None, dataset_name="SRSD Bonus 6", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sqrt', '(', '1', '+', '2', '*', 'X_0', '*', 'X_1', '*', 'X_2', '^2', '/', '(',
                                       'X_3', '*', '(', 'X_4', '*', 'X_5', '*', 'X_6', '^2', ')', '^2', ')', ')'],
                         original_equation="sqrt(1 + 2 * epsilon ** 2 * E_n * L ** 2 / (m * (Z_1 * Z_2 * q ** 2) ** 2))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-18, 1e-16),
                LogUniformSampling(1e-18, 1e-16, uses_negative=False),
                LogUniformSampling(1e-10, 1e-08, uses_negative=False),
                LogUniformSampling(1e-30, 1e-28, uses_negative=False),
                IntegerUniformSampling(1, 10, uses_negative=False),
                IntegerUniformSampling(1, 10, uses_negative=False),
                LogUniformSampling(1e-11, 1e-09),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD Bonus 7", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sqrt', '(', '8', '*', 'pi', '*', f'{self.GRAVITATIONAL_CONSTANT}', '*', 'X_0',
                                       '/', '3', '-', 'X_1', '*', f'{self.SPEED_OF_LIGHT}', '^2', '/', 'X_2', '^2',
                                       ')'],
                         original_equation="sqrt(8 * pi * 6.67430e-11 * rho / 3 - alpha * 2.99792458e8 ** 2 / d ** 2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-28, 1e-26, uses_negative=False),
                IntegerUniformSampling(-1, 2),
                LogUniformSampling(1e25, 1e27, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD Bonus 8", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', '1', '+', 'X_0', '/', '(', f'{self.ELECTRON_MASS}', '*',
                                       f'{self.SPEED_OF_LIGHT}', '^2', ')', '*', '(', '1', '-', 'cos', '(', 'X_1', ')',
                                       ')', ')'],
                         original_equation="E_n / (1 + E_n / (9.10938356e-31 * 2.99792458e8 ** 2) * (1 - cos(theta)))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-24, 1e-22, uses_negative=False),
                UniformSampling(-3.141592653589793, 3.141592653589793),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD Bonus 9", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['u-', '32', '/', '5', '*', f'{self.GRAVITATIONAL_CONSTANT}', '^4', '/',
                                       f'{self.SPEED_OF_LIGHT}', '^5', '*', 'X_0', '^2', '*', 'X_1', '^2', '*', '(',
                                       'X_0', '+', 'X_1', ')', '/', 'X_2', '^5'],
                         original_equation="-32/5 * 6.67430e-11 ** 4 / 2.99792458e8 ** 5 * (m1 * m2) ** 2 * (m1 + m2) / r ** 5",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD Bonus 10", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['(', 'cos', '(', 'X_0', ')', '-', 'X_1', '/', f'{self.SPEED_OF_LIGHT}', ')', '/',
                                       '(', '1', '-', 'X_1', '/', f'{self.SPEED_OF_LIGHT}', '*', 'cos', '(', 'X_0', ')',
                                       ')'],
                         original_equation="(cos(theta2) - v / 2.99792458e8) / (1 - v / 2.99792458e8 * cos(theta2))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                UniformSampling(0, 3.141592653589793, uses_negative=False),
                LogUniformSampling(1000000.0, 100000000.0),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD Bonus 11", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'sin', '(', 'X_1', '/', '2', ')', '^2', '*', 'sin', '(', 'X_2', '*',
                                       'X_3', '/', '2', ')', '^2', '/', '(', 'X_1', '^2', '/', '4', '*', 'sin', '(',
                                       'X_3', '/', '2', ')', '^2', ')'],
                         original_equation="I_0 * (sin(alpha / 2) * sin(n * delta / 2) / (alpha / 2 * sin(delta / 2))) ** 2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.001, 0.1, uses_negative=False),
                LogUniformSampling(1e-11, 1e-09, uses_negative=False),
                IntegerUniformSampling(1, 100, uses_negative=False),
                LogUniformSampling(1e-11, 1e-09, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="SRSD Bonus 12", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', '4', '*', 'pi', '*', 'X_1', '*', 'X_2', '^2', ')', '*', '(',
                                       '4', '*', 'pi', '*', 'X_1', '*', 'X_3', '*', 'X_4', '-', 'X_0', '*', 'X_4', '*',
                                       'X_2', '^3', '/', '(', 'X_2', '^2', '-', 'X_4', '^2', ')', '^2', ')'],
                         original_equation="q / (4 * pi * epsilon * y ** 2) * (4 * pi * epsilon * Volt * d - q * d * y ** 3 / (y ** 2 - d ** 2) ** 2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.001, 0.1),
                LogUniformSampling(1e-12, 1e-10, uses_negative=False),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="SRSD Bonus 13", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_1', '/', '(', '4', '*', 'pi', '*', 'X_0', '*', 'sqrt', '(', 'X_2', '^2', '+',
                                       'X_3', '^2', '-', '2', '*', 'X_2', '*', 'X_3', '*', 'cos', '(', 'X_4', ')', ')',
                                       ')'],
                         original_equation="1 / (4 * pi * epsilon) * q / sqrt(r ** 2 + d ** 2 - 2 * r * d * cos(alpha))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-12, 1e-10, uses_negative=False),
                LogUniformSampling(0.001, 0.1),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
                UniformSampling(0, 3.141592653589793, uses_negative=False),
            ], data_source=data_source)

        self.add_dataset(sl_5v, None, dataset_name="SRSD Bonus 14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'cos', '(', 'X_1', ')', '*', '(', 'u-', 'X_2', '+', 'X_3', '^3', '/',
                                       'X_2', '^2', '*', '(', 'X_4', '-', '1', ')', '/', '(', 'X_4', '+', '2', ')',
                                       ')'],
                         original_equation="Ef * cos(theta) * (-r + d ** 3 / r ** 2 * (alpha - 1) / (alpha + 2))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(10.0, 1000.0),
                UniformSampling(0, 3.141592653589793, uses_negative=False),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD Bonus 15", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sqrt', '(', '1', '-', 'X_0', '^2', '/', f'{self.SPEED_OF_LIGHT}', '^2', ')',
                                       '*', 'X_1', '/', '(', '1', '+', 'X_0', '/', f'{self.SPEED_OF_LIGHT}', '*', 'cos',
                                       '(', 'X_2', ')', ')'],
                         original_equation="sqrt(1 - v ** 2 / 2.99792458e8 ** 2) * omega / (1 + v / 2.99792458e8 * cos(theta))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(100000.0, 10000000.0, uses_negative=False),
                LogUniformSampling(1000000000.0, 100000000000.0, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="SRSD Bonus 16", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sqrt', '(', '(', 'X_0', '-', 'X_1', '*', 'X_2', ')', '^2', '*',
                                       f'{self.SPEED_OF_LIGHT}', '^2', '+', 'X_3', '^2', '*', f'{self.SPEED_OF_LIGHT}',
                                       '^4', ')', '+', 'X_1', '*', 'X_4'],
                         original_equation="sqrt((p - q * A_vec) ** 2 * 2.99792458e8 ** 2 + m ** 2 * 2.99792458e8 ** 4) + q * Volt",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-09, 1e-07),
                LogUniformSampling(1e-11, 1e-09),
                LogUniformSampling(10.0, 1000.0),
                LogUniformSampling(1e-30, 1e-28, uses_negative=False),
                LogUniformSampling(0.1, 10.0),
            ], data_source=data_source)
        self.add_dataset(sl_6v, None, dataset_name="SRSD Bonus 17", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['(', 'X_1', '^2', '+', 'X_0', '^2', '*', 'X_2', '^2', '*', 'X_3', '^2', '*', '(',
                                       '1', '+', 'X_4', '*', 'X_3', '/', 'X_5', ')', ')', '/', '(', '2', '*', 'X_0',
                                       ')'],
                         original_equation="1 / (2 * m) * (p ** 2 + m ** 2 * omega ** 2 * x ** 2 * (1 + alpha * x / y))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-30, 1e-28, uses_negative=False),
                LogUniformSampling(1e-09, 1e-07),
                LogUniformSampling(1000000000.0, 100000000000.0),
                LogUniformSampling(1e-11, 1e-09),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(1e-11, 1e-09, uses_negative=False),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD Bonus 18", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['3', '/', '(', '8', '*', 'pi', '*', f'{self.GRAVITATIONAL_CONSTANT}', ')', '*',
                                       '(', f'{self.SPEED_OF_LIGHT}', '^2', '*', 'X_0', '/', 'X_1', '^2', '+', 'X_2',
                                       '^2', ')'],
                         original_equation="3 / (8 * pi * 6.67430e-11) * (2.99792458e8 ** 2 * k_f / r ** 2 + H_G ** 2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(10.0, 1000.0),
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
                LogUniformSampling(1.0, 100.0),
            ], data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD Bonus 19", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['u-', '1', '/', '(', '8', '*', 'pi', '*', f'{self.GRAVITATIONAL_CONSTANT}', ')',
                                       '*', '(', f'{self.SPEED_OF_LIGHT}', '^4', '*', 'X_0', '/', 'X_1', '^2', '+',
                                       'X_2', '^2', '*', f'{self.SPEED_OF_LIGHT}', '^2', '*', '(', '1', '-', '2', '*',
                                       'X_3', ')', ')'],
                         original_equation="-1 / (8 * pi * 6.67430e-11) * (2.99792458e8 ** 4 * k_f / r ** 2 + H_G ** 2 * 2.99792458e8 ** 2 * (1 - 2 * alpha))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(10.0, 1000.0),
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
                LogUniformSampling(1.0, 100.0, uses_negative=False),
                UniformSampling(-10, 10),
            ], data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD Bonus 20", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.FINE_STRUCTURE_CONSTANT}', '^2', '*', f'{self.PLANCK_CONSTANT}', '^2',
                                       '/', '(', '4', '*', 'pi', '*', f'{self.ELECTRON_MASS}', '^2', '*',
                                       f'{self.SPEED_OF_LIGHT}', '^2', ')', '*', 'X_0', '^2', '/', 'X_1', '^2', '*',
                                       '(', 'X_0', '/', 'X_1', '+', 'X_1', '/', 'X_0', '-', 'sin', '(', 'X_2', ')',
                                       '^2', ')'],
                         original_equation="1 / (4 * pi) * 7.2973525693e-3 ** 2 * 6.626e-34 ** 2 / (9.10938356e-31 ** 2 * 2.99792458e8 ** 2) * (omega_0 / omega) ** 2 * (omega_0 / omega + omega / omega_0 - sin(beta) ** 2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1000000000.0, 100000000000.0, uses_negative=False),
                LogUniformSampling(1000000000.0, 100000000000.0, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ], data_source=data_source)

    # fmt: on

    def _ensure_data(self, dataset_name: str) -> None:
        """Download the canonical SRSD archive into the cache once, unless ``force_generate``.

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
                f"[SRSD_Feynman] Could not download the canonical data ({e}); falling back to "
                f"local sampling. Generated data may differ across machines.",
                stacklevel=2,
            )
