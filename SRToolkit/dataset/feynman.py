"""Feynman symbolic regression benchmark."""

from typing import Optional

from SRToolkit.utils.symbol_library import SymbolLibrary

from .background_knowledge import VariableInfo
from .data_source import FallbackSource, SampleSource, UrlSource
from .sampling import UniformSampling
from .sr_benchmark import SR_benchmark


class Feynman(SR_benchmark):
    """The Feynman symbolic regression benchmark.

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

        # Each dataset's data_source prefers the canonical archive (so every machine
        # benchmarks on identical inputs) and falls back to regenerating from that dataset's
        # samplers if the download is unavailable. The chain is serialised in the config, so
        # the preference travels to grid recipes, exports, and cold workers. With
        # force_generate it is sampling-only — the canonical archive is never consulted.
        if self._force_generate:
            data_source = SampleSource(n_samples=self._n_samples, seed=seed)
        else:
            data_source = FallbackSource(
                [UrlSource(url), SampleSource(n_samples=self._n_samples, seed=seed)]
            )

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
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("c", "speed of light in vacuum"),
                                    "X_1": VariableInfo("v", "velocity of the body relative to the moving frame"),
                                    "X_2": VariableInfo("u", "velocity of the moving frame relative to the lab frame")},
                         target=VariableInfo("v1", "velocity of the body relative to the lab frame"),
                         description=(
                             "A body moves with velocity v relative to a frame that itself moves with velocity u "
                             "(parallel) relative to the lab; the target is the body's velocity in the lab. It is "
                             "symmetric in u and v, close to their plain sum when both are small compared with the "
                             "speed of light, never exceeds the speed of light when both inputs are below it, and "
                             "equals the speed of light when either input does."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.15.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["u-", "X_0", "*", "X_1", "*", "cos", "(", "X_2", ")"],
                         original_equation="E_n = -mom*B*cos(theta)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("mom", "magnetic dipole moment"),
                                    "X_1": VariableInfo("B", "magnetic field strength"),
                                    "X_2": VariableInfo("theta", "angle between the magnetic moment and the field")},
                         target=VariableInfo("E_n", "potential energy of the magnetic dipole"),
                         description=(
                             "Potential energy of a magnetic dipole in a uniform magnetic field. It is proportional "
                             "to the moment and to the field, lowest (negative) when the moment is aligned with the "
                             "field, zero when perpendicular, and highest when anti-aligned."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.27.16", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "X_2", "^2"], original_equation="flux = epsilon*c*Ef^2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("epsilon", "vacuum permittivity (electric constant)"),
                                    "X_1": VariableInfo("c", "speed of light in vacuum"),
                                    "X_2": VariableInfo("Ef", "electric field strength of the wave")},
                         target=VariableInfo("flux", "energy flux of the wave"),
                         description=(
                             "Energy flux (power per unit area) carried by a plane electromagnetic wave in vacuum. It"
                             " is non-negative, even in the field strength and zero without field."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_6v, None, dataset_name="I.11.19", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_3", "+", "X_1", "*", "X_4", "+", "X_2", "*", "X_5"],
                         original_equation="A = x1*y1+x2*y2+x3*y3", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("x1", "x-component of the force"),
                                    "X_1": VariableInfo("x2", "y-component of the force"),
                                    "X_2": VariableInfo("x3", "z-component of the force"),
                                    "X_3": VariableInfo("y1", "x-component of the displacement"),
                                    "X_4": VariableInfo("y2", "y-component of the displacement"),
                                    "X_5": VariableInfo("y3", "z-component of the displacement")},
                         target=VariableInfo("A", "work done by the force over the displacement"),
                         description=(
                             "A constant force with Cartesian components (x1, x2, x3) acts on a body while it is "
                             "displaced by (y1, y2, y3); the target is the work done. It is linear in each component "
                             "of the force and of the displacement, unchanged when force and displacement are "
                             "exchanged or rotated together, and zero when they are perpendicular."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.15.3x", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_0", "-", "X_1", "*", "X_3", ")", "/", "sqrt", "(", "1", "-", "X_1", "^2",
                                       "/", "X_2", "^2", ")"], original_equation="x1 = (x-u*t)/sqrt(1-u^2/c^2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(5, 10, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(3, 20, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False)],
                         variables={"X_0": VariableInfo("x", "position coordinate of the event in the rest frame"),
                                    "X_1": VariableInfo("u", "velocity of the moving frame along x"),
                                    "X_2": VariableInfo("c", "speed of light in vacuum"),
                                    "X_3": VariableInfo("t", "time of the event in the rest frame")},
                         target=VariableInfo("x1", "position coordinate of the event in the moving frame"),
                         description=(
                             "An event has position x and time t in one inertial frame; the target is its position in"
                             " a second frame moving with velocity u along the x-axis (origins coinciding at t = 0). "
                             "It reduces to x when u = 0, approaches the classical shift x - u*t when u is much "
                             "smaller than the speed of light, and is defined only for |u| below the speed of light."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.10.7", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_2", "^2", ")"],
                         original_equation="m = m_0/sqrt(1-v^2/c^2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 2, uses_negative=False),
                                                                   UniformSampling(3, 10, uses_negative=False)],
                         variables={"X_0": VariableInfo("m_0", "rest mass of the body"),
                                    "X_1": VariableInfo("v", "speed of the body"),
                                    "X_2": VariableInfo("c", "speed of light in vacuum")},
                         target=VariableInfo("m", "inertial mass of the moving body"),
                         description=(
                             "Effective inertial mass of a body moving at speed v, given its rest mass, in special "
                             "relativity. It equals the rest mass when the body is at rest, is even in v, increases "
                             "monotonically with speed and diverges as the speed approaches the speed of light; it is"
                             " defined only for speeds below the speed of light."
                         ),
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
                                   UniformSampling(1, 2, uses_negative=False)],
                         variables={"X_0": VariableInfo("m1", "mass of the first body"),
                                    "X_1": VariableInfo("m2", "mass of the second body"),
                                    "X_2": VariableInfo("G", "gravitational constant"),
                                    "X_3": VariableInfo("x1", "x-coordinate of the first body"),
                                    "X_4": VariableInfo("x2", "x-coordinate of the second body"),
                                    "X_5": VariableInfo("y1", "y-coordinate of the first body"),
                                    "X_6": VariableInfo("y2", "y-coordinate of the second body"),
                                    "X_7": VariableInfo("z1", "z-coordinate of the first body"),
                                    "X_8": VariableInfo("z2", "z-coordinate of the second body")},
                         target=VariableInfo("F", "magnitude of the attractive force between the two masses"),
                         description=(
                             "Gravitational attraction between two point masses at given positions in three-"
                             "dimensional space. The force is positive, grows with each mass, is symmetric under "
                             "exchanging the two bodies, depends on the positions only through their separation, "
                             "diverges as the bodies approach each other and vanishes at large separation."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.15.3t", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_3", "-", "X_2", "*", "X_0", "/", "X_1", "^2", ")", "/", "sqrt", "(", "1",
                                       "-", "X_2", "^2", "/", "X_1", "^2", ")"],
                         original_equation="t1 = (t-u*x/c^2)/sqrt(1-u^2/c^2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(3, 10, uses_negative=False),
                                                                   UniformSampling(1, 2, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("x", "position coordinate of the event in the rest frame"),
                                    "X_1": VariableInfo("c", "speed of light in vacuum"),
                                    "X_2": VariableInfo("u", "velocity of the moving frame along x"),
                                    "X_3": VariableInfo("t", "time of the event in the rest frame")},
                         target=VariableInfo("t1", "time coordinate of the event in the moving frame"),
                         description=(
                             "An event has position x and time t in one inertial frame; the target is its time in a "
                             "second frame moving with velocity u along the x-axis (origins coinciding at t = 0). It "
                             "equals t when u = 0, approaches t when u is much smaller than the speed of light, and "
                             "is defined only for |u| below the speed of light."
                         ),
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
                                   UniformSampling(1, 3, uses_negative=False)],
                         variables={"X_0": VariableInfo("mom", "magnetic moment of an atom"),
                                    "X_1": VariableInfo("H", "applied magnetic field"),
                                    "X_2": VariableInfo("kb", "Boltzmann constant"),
                                    "X_3": VariableInfo("T", "absolute temperature"),
                                    "X_4": VariableInfo("alpha", "coupling constant of the internal field to the magnetization"),
                                    "X_5": VariableInfo("epsilon", "vacuum permittivity (electric constant)"),
                                    "X_6": VariableInfo("c", "speed of light in vacuum"),
                                    "X_7": VariableInfo("M", "magnetization of the material")},
                         target=VariableInfo("f", "ratio of the magnetic energy of an atom to the thermal energy"),
                         description=(
                             "In a magnetized material the field acting on an atom is the applied field plus an "
                             "internal contribution proportional to the magnetization; the target compares the "
                             "resulting magnetic energy of an atomic moment with the thermal energy at temperature T."
                             " It is linear in the applied field and in the magnetization, proportional to the moment"
                             " and decreases with temperature."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.43.43", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "1", "/", "(", "X_0", "-", "1", ")", ")", "*", "X_1", "*", "X_3", "/",
                                       "X_2"], original_equation="kappa = 1/(gamma-1)*kb*v/A", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(2, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("gamma", "adiabatic index (ratio of specific heats)"),
                                    "X_1": VariableInfo("kb", "Boltzmann constant"),
                                    "X_2": VariableInfo("A", "collision cross-section of the molecules"),
                                    "X_3": VariableInfo("v", "mean speed of the molecules")},
                         target=VariableInfo("kappa", "thermal conductivity of the gas"),
                         description=(
                             "Thermal conductivity of a dilute gas estimated from the motion and collisions of its "
                             "molecules. It increases with the mean molecular speed, decreases with the collision "
                             "cross-section, does not depend on the gas density, decreases with gamma and diverges as"
                             " gamma approaches one."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.15.5", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["u-", "X_0", "*", "X_1", "*", "cos", "(", "X_2", ")"],
                         original_equation="E_n = -p_d*Ef*cos(theta)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("p_d", "electric dipole moment"),
                                    "X_1": VariableInfo("Ef", "electric field strength"),
                                    "X_2": VariableInfo("theta", "angle between the dipole moment and the field")},
                         target=VariableInfo("E_n", "potential energy of the electric dipole"),
                         description=(
                             "Potential energy of an electric dipole in a uniform electric field. It is proportional "
                             "to the moment and to the field, lowest (negative) when the dipole is aligned with the "
                             "field, zero when perpendicular, and highest when anti-aligned."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.37.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "+", "X_1", "+", "2", "*", "sqrt", "(", "X_0", "*", "X_1", ")", "*",
                                       "cos", "(", "X_2", ")"],
                         original_equation="Int = I1+I2+2*sqrt(I1*I2)*cos(delta)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("I1", "intensity of the first beam"),
                                    "X_1": VariableInfo("I2", "intensity of the second beam"),
                                    "X_2": VariableInfo("delta", "phase difference between the beams")},
                         target=VariableInfo("Int", "resulting intensity"),
                         description=(
                             "Intensity where two coherent beams overlap with a given phase difference. It is "
                             "symmetric in the two intensities, even and 2*pi-periodic in delta, largest for beams in"
                             " phase, smallest (zero for equal beams) in antiphase, and its average over delta is the"
                             " sum of the two intensities."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="II.6.11", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "1", "/", "(", "4", "*", "pi", "*", "X_0", ")", ")", "*", "X_1", "*", "cos",
                                       "(", "X_2", ")", "/", "X_3", "^2"],
                         original_equation="Volt = 1/(4*pi*epsilon)*p_d*cos(theta)/r^2", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False)],
                         variables={"X_0": VariableInfo("epsilon", "vacuum permittivity (electric constant)"),
                                    "X_1": VariableInfo("p_d", "electric dipole moment"),
                                    "X_2": VariableInfo("theta", "angle between the dipole axis and the direction to the observation point"),
                                    "X_3": VariableInfo("r", "distance from the dipole")},
                         target=VariableInfo("Volt", "electric potential"),
                         description=(
                             "Electric potential of a point electric dipole in vacuum, at distance r and angle theta "
                             "from the dipole axis. It is proportional to the dipole moment, positive on the side the"
                             " dipole points to, zero in the plane perpendicular to the axis (theta = pi/2), changes "
                             "sign under theta -> pi - theta, and falls off with distance faster than the potential "
                             "of a single charge."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="III.7.38", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["2", "*", "X_0", "*", "X_1", "/", "(", "X_2", "/", "(", "2", "*", "pi", ")",
                                       ")"], original_equation="omega = 2*mom*B/(h/(2*pi))", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("mom", "magnetic moment of the particle"),
                                    "X_1": VariableInfo("B", "magnetic field strength"),
                                    "X_2": VariableInfo("h", "Planck constant")},
                         target=VariableInfo("omega", "angular frequency of the spin flip"),
                         description=(
                             "Angular frequency at which a spin-1/2 particle with magnetic moment mom in a uniform "
                             "magnetic field oscillates between (or precesses across) its two spin states. It is "
                             "proportional to the moment and to the field."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.34.2a", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "/", "(", "2", "*", "pi", "*", "X_2", ")"],
                         original_equation="I = q*v/(2*pi*r)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("q", "charge of the particle"),
                                    "X_1": VariableInfo("v", "orbital speed"),
                                    "X_2": VariableInfo("r", "radius of the orbit")},
                         target=VariableInfo("I", "electric current"),
                         description=(
                             "Average electric current produced by a point charge circulating on a circular orbit. It"
                             " is proportional to the charge and to the speed and decreases with the radius."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.13.23", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_2", "^2", ")"],
                         original_equation="rho_c = rho_c_0/sqrt(1-v^2/c^2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 2, uses_negative=False),
                                                                   UniformSampling(3, 10, uses_negative=False)],
                         variables={"X_0": VariableInfo("rho_c_0", "charge density in the rest frame of the charges"),
                                    "X_1": VariableInfo("v", "speed of the charges"),
                                    "X_2": VariableInfo("c", "speed of light in vacuum")},
                         target=VariableInfo("rho_c", "charge density in the frame where the charges move"),
                         description=(
                             "Charge density observed in a frame in which a charge distribution moves at speed v, "
                             "given its density in its own rest frame (special relativity). It equals the rest-frame "
                             "density at v = 0, is proportional to it, even in v, increases with speed and diverges "
                             "as v approaches the speed of light."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="I.29.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "/", "X_1"], original_equation="k = omega/c", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 10, uses_negative=False),
                                   UniformSampling(1, 10, uses_negative=False)],
                         variables={"X_0": VariableInfo("omega", "angular frequency of the wave"),
                                    "X_1": VariableInfo("c", "speed of light in vacuum")},
                         target=VariableInfo("k", "wavenumber"),
                         description=(
                             "Wavenumber of an electromagnetic wave travelling in vacuum, given its angular "
                             "frequency. It is proportional to the frequency."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.38.12", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["4", "*", "pi", "*", "X_3", "*", "(", "X_2", "/", "(", "2", "*", "pi", ")", ")",
                                       "^2", "/", "(", "X_0", "*", "X_1", "^2", ")"],
                         original_equation="r = 4*pi*epsilon*(h/(2*pi))^2/(m*q^2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("m", "mass of the orbiting particle"),
                                    "X_1": VariableInfo("q", "magnitude of the charge of the particle and of the nucleus"),
                                    "X_2": VariableInfo("h", "Planck constant"),
                                    "X_3": VariableInfo("epsilon", "vacuum permittivity (electric constant)")},
                         target=VariableInfo("r", "radius of the ground-state orbit"),
                         description=(
                             "Characteristic size of the lowest-energy state of a light charged particle bound to an "
                             "opposite, much heavier charge of equal magnitude (a hydrogen-like atom) in quantum "
                             "theory. It decreases with the particle mass and with the charge magnitude."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="III.15.27", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["2", "*", "pi", "*", "X_0", "/", "(", "X_1", "*", "X_2", ")"],
                         original_equation="k = 2*pi*alpha/(n*d)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("alpha", "integer labelling the state"),
                                    "X_1": VariableInfo("n", "number of atoms in the ring"),
                                    "X_2": VariableInfo("d", "lattice spacing")},
                         target=VariableInfo("k", "allowed wavenumber"),
                         description=(
                             "Allowed wavenumbers of an electron on a closed ring (periodic chain) of n atoms with "
                             "spacing d. They are proportional to the integer label alpha, evenly spaced, and "
                             "decrease with the number of atoms and with the spacing."
                         ),
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
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("omega", "angular frequency"),
                                    "X_1": VariableInfo("T", "absolute temperature"),
                                    "X_2": VariableInfo("h", "Planck constant"),
                                    "X_3": VariableInfo("kb", "Boltzmann constant"),
                                    "X_4": VariableInfo("c", "speed of light in vacuum")},
                         target=VariableInfo("L_rad", "spectral intensity of the radiation per unit angular frequency"),
                         description=(
                             "Spectral distribution of the electromagnetic radiation in thermal equilibrium inside a "
                             "cavity whose walls are at temperature T. It is positive, increases with temperature at "
                             "every frequency, vanishes at zero and at very high frequency with a single maximum "
                             "whose position moves up in proportion to the temperature, and at low frequency becomes "
                             "linear in temperature."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.48.20", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_2", "^2", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_2",
                                       "^2", ")"], original_equation="E_n = m*c^2/sqrt(1-v^2/c^2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(3, 10, uses_negative=False)],
                         variables={"X_0": VariableInfo("m", "rest mass of the body"),
                                    "X_1": VariableInfo("v", "speed of the body"),
                                    "X_2": VariableInfo("c", "speed of light in vacuum")},
                         target=VariableInfo("E_n", "total energy of the body"),
                         description=(
                             "Total energy (rest energy included) of a free body moving at speed v, in special "
                             "relativity. It is proportional to the mass, even in v, smallest for a body at rest, and"
                             " increases monotonically without bound as the speed approaches the speed of light."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="II.11.20", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "^2", "*", "X_2", "/", "(", "3", "*", "X_3", "*", "X_4", ")"],
                         original_equation="Pol = n_rho*p_d^2*Ef/(3*kb*T)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("n_rho", "number density of molecules"),
                                    "X_1": VariableInfo("p_d", "permanent electric dipole moment of a molecule"),
                                    "X_2": VariableInfo("Ef", "applied electric field strength"),
                                    "X_3": VariableInfo("kb", "Boltzmann constant"),
                                    "X_4": VariableInfo("T", "absolute temperature")},
                         target=VariableInfo("Pol", "electric polarization"),
                         description=(
                             "Orientational polarization of a dilute gas of polar molecules in a weak electric field "
                             "at temperature T. It is proportional to the density and to the field, even in the "
                             "dipole moment (growing with its magnitude) and decreases with temperature."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="I.25.13", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "/", "X_1"], original_equation="Volt = q/C", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("q", "charge stored on the capacitor"),
                                    "X_1": VariableInfo("C", "capacitance")},
                         target=VariableInfo("Volt", "voltage across the capacitor"),
                         description=(
                             "Voltage across a capacitor holding charge q. It is proportional to the charge and "
                             "decreases with increasing capacitance."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="III.15.12", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["2", "*", "X_0", "*", "(", "1", "-", "cos", "(", "X_1", "*", "X_2", ")", ")"],
                         original_equation="E_n = 2*U*(1-cos(k*d))", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("U", "coupling energy between neighbouring atoms"),
                                    "X_1": VariableInfo("k", "wavenumber of the electron"),
                                    "X_2": VariableInfo("d", "lattice spacing")},
                         target=VariableInfo("E_n", "energy of the electron"),
                         description=(
                             "Energy, measured from the bottom of the band, of an electron with wavenumber k in a "
                             "one-dimensional crystal lattice. It is non-negative, proportional to the coupling "
                             "energy, even in k and periodic with period 2*pi/d, zero at k = 0 and maximal at k = "
                             "pi/d."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.24.6", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["0.25", "*", "X_0", "*", "(", "X_1", "^2", "+", "X_2", "^2", ")", "*", "X_3",
                                       "^2"], original_equation="E_n = 1/2*m*(omega^2+omega_0^2)*1/2*x^2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False)],
                         variables={"X_0": VariableInfo("m", "mass of the oscillator"),
                                    "X_1": VariableInfo("omega", "angular frequency of the driving force"),
                                    "X_2": VariableInfo("omega_0", "natural angular frequency of the oscillator"),
                                    "X_3": VariableInfo("x", "amplitude of the oscillation")},
                         target=VariableInfo("E_n", "time-averaged total energy of the oscillator"),
                         description=(
                             "Time-averaged total (kinetic plus potential) energy of a mass on a spring with natural "
                             "frequency omega_0 that oscillates steadily at the driving frequency omega with "
                             "amplitude x. It is non-negative, proportional to the mass, grows with the amplitude, "
                             "and is symmetric under exchanging the two frequencies."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="I.34.27", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_1", "/", "(", "2", "*", "pi", ")", ")", "*", "X_0"],
                         original_equation="E_n = (h/(2*pi))*omega", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("omega", "angular frequency of the radiation"),
                                    "X_1": VariableInfo("h", "Planck constant")},
                         target=VariableInfo("E_n", "energy of one quantum of radiation"),
                         description=(
                             "Energy carried by a single quantum (photon) of electromagnetic radiation of a given "
                             "angular frequency. It is positive and proportional to the frequency."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.43.31", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_2", "*", "X_1"], original_equation="D = mob*kb*T",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("mob", "mobility (drift velocity per unit applied force)"),
                                    "X_1": VariableInfo("T", "absolute temperature"),
                                    "X_2": VariableInfo("kb", "Boltzmann constant")},
                         target=VariableInfo("D", "diffusion coefficient"),
                         description=(
                             "Diffusion coefficient of small particles suspended in a fluid at temperature T, given "
                             "their mobility. It is proportional to both the mobility and the temperature."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.29.16", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["sqrt", "(", "X_0", "^2", "+", "X_1", "^2", "-", "2", "*", "X_0", "*", "X_1",
                                       "*", "cos", "(", "X_2", "-", "X_3", ")", ")"],
                         original_equation="x = sqrt(x1^2+x2^2-2*x1*x2*cos(theta1-theta2))", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("x1", "distance of the first point from the origin"),
                                    "X_1": VariableInfo("x2", "distance of the second point from the origin"),
                                    "X_2": VariableInfo("theta1", "polar angle of the first point"),
                                    "X_3": VariableInfo("theta2", "polar angle of the second point")},
                         target=VariableInfo("x", "distance between the two points"),
                         description=(
                             "Distance between two points of a plane given in polar coordinates. It is non-negative, "
                             "depends on the angles only through their difference (even and 2*pi-periodic in it), "
                             "equals |x1 - x2| when the angles coincide and x1 + x2 when the points lie in opposite "
                             "directions, and is symmetric under exchanging the points."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.18.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_0", "*", "X_2", "+", "X_1", "*", "X_3", ")", "/", "(", "X_0", "+", "X_1",
                                       ")"], original_equation="r = (m1*r1+m2*r2)/(m1+m2)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("m1", "mass of the first body"),
                                    "X_1": VariableInfo("m2", "mass of the second body"),
                                    "X_2": VariableInfo("r1", "position of the first body"),
                                    "X_3": VariableInfo("r2", "position of the second body")},
                         target=VariableInfo("r", "position of the centre of mass"),
                         description=(
                             "Position of the mass-weighted centre of two point masses on a line. It lies between r1 "
                             "and r2, closer to the heavier body, is unchanged when the two bodies (mass and position"
                             " together) are exchanged or when both masses are scaled by the same factor, and equals "
                             "the midpoint for equal masses."
                         ),
                         data_source=data_source)
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
                                   UniformSampling(1, 3, uses_negative=False)],
                         variables={"X_0": VariableInfo("epsilon", "vacuum permittivity (electric constant)"),
                                    "X_1": VariableInfo("p_d", "electric dipole moment (along the z-axis)"),
                                    "X_2": VariableInfo("r", "distance from the dipole"),
                                    "X_3": VariableInfo("x", "x-coordinate of the observation point"),
                                    "X_4": VariableInfo("y", "y-coordinate of the observation point"),
                                    "X_5": VariableInfo("z", "z-coordinate of the observation point (along the dipole axis)")},
                         target=VariableInfo("Ef", "electric field component perpendicular to the dipole axis"),
                         description=(
                             "Electric field component perpendicular to the axis of a point dipole at the origin, at "
                             "the point (x, y, z) whose distance from the dipole is r. It is proportional to the "
                             "dipole moment, vanishes on the axis and in the plane z = 0, is odd in z, depends on x "
                             "and y only through the distance from the axis, and decays rapidly with r."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.30.3", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "sin", "(", "X_2", "*", "X_1", "/", "2", ")", "^2", "/", "sin", "(",
                                       "X_1", "/", "2", ")", "^2"],
                         original_equation="Int = Int_0*sin(n*theta/2)^2/sin(theta/2)^2", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("Int_0", "intensity produced by a single source"),
                                    "X_1": VariableInfo("theta", "phase difference between neighbouring sources"),
                                    "X_2": VariableInfo("n", "number of sources")},
                         target=VariableInfo("Int", "radiated intensity"),
                         description=(
                             "Intensity far from a row of n identical, equally spaced coherent sources, as a function"
                             " of the phase difference theta between neighbouring sources. It is non-negative, "
                             "proportional to Int_0, even and 2*pi-periodic in theta, reaches its maximum (n^2 times "
                             "the single-source intensity) at theta = 0 and has n - 1 zeros per period between the "
                             "maxima."
                         ),
                         data_source=data_source)
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
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("p_d", "electric dipole moment of the transition"),
                                    "X_1": VariableInfo("Ef", "amplitude of the oscillating electric field"),
                                    "X_2": VariableInfo("t", "duration of the interaction"),
                                    "X_3": VariableInfo("h", "Planck constant"),
                                    "X_4": VariableInfo("omega", "angular frequency of the driving field"),
                                    "X_5": VariableInfo("omega_0", "resonance angular frequency of the two-state system")},
                         target=VariableInfo("prob", "transition probability"),
                         description=(
                             "Weakly driven two-state system (resonance omega_0) exposed to an oscillating electric "
                             "field for a time t. The response peaks sharply at resonance, is even in the detuning "
                             "omega - omega_0, vanishes when the detuning times t is a nonzero multiple of 2*pi, and "
                             "its peak grows with the dipole moment, the field and the interaction time."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.34.2", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "X_2", "/", "2"], original_equation="mom = q*v*r/2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("q", "charge of the particle"),
                                    "X_1": VariableInfo("v", "orbital speed"),
                                    "X_2": VariableInfo("r", "radius of the orbit")},
                         target=VariableInfo("mom", "magnetic moment of the orbit"),
                         description=(
                             "Magnetic moment produced by a point charge moving on a circular orbit. It is "
                             "proportional to the charge, the speed and the radius, and its sign follows the charge."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.39.11", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "1", "/", "(", "X_0", "-", "1", ")", ")", "*", "X_1", "*", "X_2"],
                         original_equation="E_n = (1/(gamma-1))*pr*V", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(2, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("gamma", "adiabatic index (ratio of specific heats)"),
                                    "X_1": VariableInfo("pr", "pressure of the gas"),
                                    "X_2": VariableInfo("V", "volume of the gas")},
                         target=VariableInfo("E_n", "internal energy of the gas"),
                         description=(
                             "Internal energy of a dilute gas with adiabatic index gamma > 1, at pressure pr in "
                             "volume V. It is proportional to the pressure and to the volume, decreases with "
                             "increasing gamma and diverges as gamma approaches one."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="II.11.28", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["1", "+", "X_0", "*", "X_1", "/", "(", "1", "-", "(", "X_0", "*", "X_1", "/",
                                       "3", ")", ")"], original_equation="theta = 1+n*alpha/(1-(n*alpha/3))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(0, 1, uses_negative=False),
                                   UniformSampling(0, 1, uses_negative=False)],
                         variables={"X_0": VariableInfo("n", "number density of molecules"),
                                    "X_1": VariableInfo("alpha", "molecular polarizability")},
                         target=VariableInfo("theta", "dielectric constant (relative permittivity)"),
                         description=(
                             "Dielectric constant of a dense isotropic dielectric, accounting for the field of the "
                             "neighbouring polarized molecules. It equals one for empty space, depends on density and"
                             " polarizability only through their product, increases with that product and diverges at"
                             " a finite value of it."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="II.3.24", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "/", "(", "4", "*", "pi", "*", "X_1", "^2", ")"],
                         original_equation="flux = Pwr/(4*pi*r^2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("Pwr", "total power emitted by the source"),
                                    "X_1": VariableInfo("r", "distance from the source")},
                         target=VariableInfo("flux", "energy flux (intensity)"),
                         description=(
                             "Energy flux at distance r from a small source that radiates equally in all directions "
                             "into a non-absorbing medium. It is proportional to the emitted power and decreases "
                             "monotonically with distance."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.24.17", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["sqrt", "(", "X_0", "^2", "/", "X_1", "^2", "-", "pi", "^2", "/", "X_2", "^2",
                                       ")"], original_equation="k = sqrt(omega^2/c^2-pi^2/d^2)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(4, 6, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(2, 4, uses_negative=False)],
                         variables={"X_0": VariableInfo("omega", "angular frequency of the wave"),
                                    "X_1": VariableInfo("c", "speed of light in vacuum"),
                                    "X_2": VariableInfo("d", "transverse width of the guide")},
                         target=VariableInfo("k", "propagation wavenumber along the guide"),
                         description=(
                             "Wavenumber along a rectangular metal waveguide for its lowest mode at angular frequency"
                             " omega. It is real only above a cutoff frequency set by the width (below it the wave "
                             "does not propagate), zero at cutoff, increases with frequency and with the width, and "
                             "approaches the free-space wavenumber at high frequency."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="II.13.17", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "1", "/", "(", "4", "*", "pi", "*", "X_0", "*", "X_1", "^2", ")", ")", "*",
                                       "2", "*", "X_2", "/", "X_3"], original_equation="B = 1/(4*pi*epsilon*c^2)*2*I/r",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("epsilon", "vacuum permittivity (electric constant)"),
                                    "X_1": VariableInfo("c", "speed of light in vacuum"),
                                    "X_2": VariableInfo("I", "electric current in the wire"),
                                    "X_3": VariableInfo("r", "distance from the wire")},
                         target=VariableInfo("B", "magnetic field strength"),
                         description=(
                             "Magnetic field at distance r from a long straight wire carrying a steady current in "
                             "vacuum. It is proportional to the current, decreases monotonically with distance and is"
                             " the same at all points of a circle around the wire."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="I.12.5", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1"], original_equation="F = q2*Ef", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("q2", "test charge"),
                                    "X_1": VariableInfo("Ef", "electric field strength at the position of the charge")},
                         target=VariableInfo("F", "electric force on the charge"),
                         description=(
                             "Force exerted on a small test charge by a given electric field. It is proportional to "
                             "both the charge and the field, reverses sign with either, and vanishes when either is "
                             "zero."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="II.35.18", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "/", "(", "exp", "(", "X_3", "*", "X_4", "/", "(", "X_1", "*", "X_2", ")",
                                       ")", "+", "exp", "(", "u-", "X_3", "*", "X_4", "/", "(", "X_1", "*", "X_2", ")",
                                       ")", ")"], original_equation="n = n_0/(exp(mom*B/(kb*T))+exp(-mom*B/(kb*T)))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False)],
                         variables={"X_0": VariableInfo("n_0", "total number density of atoms"),
                                    "X_1": VariableInfo("kb", "Boltzmann constant"),
                                    "X_2": VariableInfo("T", "absolute temperature"),
                                    "X_3": VariableInfo("mom", "magnetic moment of an atom"),
                                    "X_4": VariableInfo("B", "magnetic field strength")},
                         target=VariableInfo("n", "density per thermal weight (total density divided by the summed weights of the two states)"),
                         description=(
                             "Atoms with magnetic moment mom that can only point along or against a magnetic field, "
                             "in thermal equilibrium at temperature T. The target is the total density divided by the"
                             " sum of the thermal weights of the two orientations; it equals n_0/2 without field, is "
                             "even in the field, and decreases monotonically to zero as the magnetic energy grows "
                             "relative to the thermal energy."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="II.34.11", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "X_2", "/", "(", "2", "*", "X_3", ")"],
                         original_equation="omega = g_*q*B/(2*m)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("g_", "g-factor of the particle"),
                                    "X_1": VariableInfo("q", "charge of the particle"),
                                    "X_2": VariableInfo("B", "magnetic field strength"),
                                    "X_3": VariableInfo("m", "mass of the particle")},
                         target=VariableInfo("omega", "angular frequency of precession of the magnetic moment"),
                         description=(
                             "Angular frequency with which the magnetic moment of a spinning charged particle "
                             "precesses about a uniform magnetic field. It is proportional to the g-factor, the "
                             "charge and the field, and decreases with the mass."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.34.29a", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "/", "(", "4", "*", "pi", "*", "X_2", ")"],
                         original_equation="mom = q*h/(4*pi*m)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("q", "charge of the particle"),
                                    "X_1": VariableInfo("h", "Planck constant"),
                                    "X_2": VariableInfo("m", "mass of the particle")},
                         target=VariableInfo("mom", "unit of magnetic moment"),
                         description=(
                             "Natural unit of magnetic moment of a charged particle in quantum theory: the orbital "
                             "magnetic moment that goes with the smallest nonzero unit of angular momentum. It is "
                             "proportional to the charge and decreases with the mass."
                         ),
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
                                   UniformSampling(3, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("epsilon", "vacuum permittivity (electric constant)"),
                                    "X_1": VariableInfo("c", "speed of light in vacuum"),
                                    "X_2": VariableInfo("Ef", "electric field amplitude of the incident wave"),
                                    "X_3": VariableInfo("r", "characteristic radius of the scatterer (sets its cross-section)"),
                                    "X_4": VariableInfo("omega", "angular frequency of the incident wave"),
                                    "X_5": VariableInfo("omega_0", "natural (resonance) angular frequency of the bound charge")},
                         target=VariableInfo("Pwr", "time-averaged scattered power"),
                         description=(
                             "Time-averaged power scattered by an elastically bound, undamped charge illuminated by a"
                             " plane electromagnetic wave. It is proportional to the incident intensity, diverges at "
                             "resonance (omega = omega_0), rises steeply with frequency far below resonance and "
                             "approaches a constant far above it."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="II.35.21", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "tanh", "(", "X_1", "*", "X_2", "/", "(", "X_3", "*",
                                       "X_4", ")", ")"], original_equation="M = n_rho*mom*tanh(mom*B/(kb*T))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("n_rho", "number density of atoms"),
                                    "X_1": VariableInfo("mom", "magnetic moment of an atom"),
                                    "X_2": VariableInfo("B", "magnetic field strength"),
                                    "X_3": VariableInfo("kb", "Boltzmann constant"),
                                    "X_4": VariableInfo("T", "absolute temperature")},
                         target=VariableInfo("M", "magnetization"),
                         description=(
                             "Magnetization of a dilute system of atoms whose moments can only point along or against"
                             " an applied field, at temperature T. It is odd in the field, linear for weak fields or "
                             "high temperature, and saturates at full alignment for strong fields or low temperature;"
                             " it increases with the field and decreases with temperature."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="I.44.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "X_2", "*", "ln", "(", "X_4", "/", "X_3", ")"],
                         original_equation="E_n = n*kb*T*ln(V2/V1)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("n", "number of molecules"),
                                    "X_1": VariableInfo("kb", "Boltzmann constant"),
                                    "X_2": VariableInfo("T", "absolute temperature"),
                                    "X_3": VariableInfo("V1", "initial volume"),
                                    "X_4": VariableInfo("V2", "final volume")},
                         target=VariableInfo("E_n", "work done by the expanding gas"),
                         description=(
                             "Work done by a dilute gas of n molecules expanding at constant temperature T from "
                             "volume V1 to V2. It is positive for expansion, zero for V1 = V2, changes sign when the "
                             "volumes are exchanged, depends on the volumes only through their ratio, and is "
                             "proportional to n and T."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="III.4.32", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["1", "/", "(", "exp", "(", "(", "X_0", "/", "(", "2", "*", "pi", ")", ")", "*",
                                       "X_1", "/", "(", "X_2", "*", "X_3", ")", ")", "-", "1", ")"],
                         original_equation="n = 1/(exp((h/(2*pi))*omega/(kb*T))-1)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("h", "Planck constant"),
                                    "X_1": VariableInfo("omega", "angular frequency of the mode"),
                                    "X_2": VariableInfo("kb", "Boltzmann constant"),
                                    "X_3": VariableInfo("T", "absolute temperature")},
                         target=VariableInfo("n", "mean number of quanta in the mode"),
                         description=(
                             "Mean number of quanta in one radiation (oscillator) mode in thermal equilibrium at "
                             "temperature T. It is positive, decreases with frequency, increases with temperature, "
                             "depends on them only through their ratio, is large at low frequency (approximately "
                             "thermal energy over quantum energy) and exponentially small at high frequency."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.10.9", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_0", "/", "X_1", ")", "*", "1", "/", "(", "1", "+", "X_2", ")"],
                         original_equation="Ef = sigma_den/epsilon*1/(1+chi)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("sigma_den", "free surface charge density on the plates"),
                                    "X_1": VariableInfo("epsilon", "vacuum permittivity (electric constant)"),
                                    "X_2": VariableInfo("chi", "electric susceptibility of the dielectric")},
                         target=VariableInfo("Ef", "electric field inside the dielectric"),
                         description=(
                             "Electric field inside a linear dielectric that fills the gap of a parallel-plate "
                             "capacitor whose plates carry a given free surface charge density. It is proportional to"
                             " the charge density, equals the empty-capacitor value at chi = 0 and decreases as the "
                             "susceptibility grows."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="II.38.3", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "X_3", "/", "X_2"], original_equation="F = Y*A*x/d",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("Y", "Young's modulus of the material"),
                                    "X_1": VariableInfo("A", "cross-sectional area of the rod"),
                                    "X_2": VariableInfo("d", "length of the rod"),
                                    "X_3": VariableInfo("x", "extension of the rod")},
                         target=VariableInfo("F", "stretching force"),
                         description=(
                             "Force needed to stretch an elastic rod by a small extension x. It is proportional to "
                             "the modulus, the cross-sectional area and the extension, and decreases with the length "
                             "of the rod."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.6.2b", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["exp", "(", "u-", "(", "(", "(", "X_1", "-", "X_2", ")", "/", "X_0", ")", "^2",
                                       ")", "/", "2", ")", "/", "(", "sqrt", "(", "2", "*", "pi", ")", "*", "X_0", ")"],
                         original_equation="f = exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False)],
                         variables={"X_0": VariableInfo("sigma", "spread (standard deviation) of the random quantity"),
                                    "X_1": VariableInfo("theta", "value of the random quantity"),
                                    "X_2": VariableInfo("theta1", "mean (centre) of the distribution")},
                         target=VariableInfo("f", "probability density"),
                         description=(
                             "Probability density of a continuous random quantity theta that scatters symmetrically "
                             "around the mean theta1 with spread sigma. It is positive, maximal at theta = theta1, "
                             "symmetric under reflection about theta1, depends on theta and theta1 only through their"
                             " difference, and integrates to one over theta; a larger spread gives a lower and "
                             "broader peak."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="II.8.31", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "^2", "/", "2"], original_equation="E_den = epsilon*Ef**2/2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("epsilon", "vacuum permittivity (electric constant)"),
                                    "X_1": VariableInfo("Ef", "electric field strength")},
                         target=VariableInfo("E_den", "energy density of the electric field"),
                         description=(
                             "Energy stored per unit volume in an electric field in vacuum. It is non-negative, even "
                             "in the field and zero without field."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_1v, None, dataset_name="I.6.2a", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["exp", "(", "u-", "X_0", "^2", "/", "2", ")", "/", "sqrt", "(", "2", "*", "pi",
                                       ")"], original_equation="f = exp(-theta**2/2)/sqrt(2*pi)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False)],
                         variables={"X_0": VariableInfo("theta", "value of the random quantity, measured from its mean in units of its spread")},
                         target=VariableInfo("f", "probability density"),
                         description=(
                             "Probability density of a continuous random quantity that has been shifted to zero mean "
                             "and scaled to unit spread. The density is positive, symmetric about zero, has a single "
                             "maximum at zero, falls off faster than any power for large |theta|, and integrates to "
                             "one."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="III.12.43", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "(", "X_1", "/", "(", "2", "*", "pi", ")", ")"],
                         original_equation="L = n*(h/(2*pi))", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("n", "angular momentum quantum number"),
                                    "X_1": VariableInfo("h", "Planck constant")},
                         target=VariableInfo("L", "angular momentum"),
                         description=(
                             "Allowed values of a component of the angular momentum of a quantum system, labelled by "
                             "a quantum number n. The values are proportional to n and evenly spaced."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="III.17.37", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "(", "1", "+", "X_1", "*", "cos", "(", "X_2", ")", ")"],
                         original_equation="f = beta*(1+alpha*cos(theta))", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("beta", "overall rate (isotropic part)"),
                                    "X_1": VariableInfo("alpha", "asymmetry coefficient"),
                                    "X_2": VariableInfo("theta", "emission angle relative to the reference axis")},
                         target=VariableInfo("f", "angular distribution of the emitted particles"),
                         description=(
                             "Angular distribution of particles emitted, e.g. in a decay, relative to a reference "
                             "axis. It is proportional to beta, even and 2*pi-periodic in theta, equals beta at theta"
                             " = pi/2, and alpha sets the forward-backward asymmetry."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="III.10.19", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "sqrt", "(", "X_1", "^2", "+", "X_2", "^2", "+", "X_3", "^2", ")"],
                         original_equation="E_n = mom*sqrt(Bx**2+By**2+Bz**2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("mom", "magnetic moment"),
                                    "X_1": VariableInfo("Bx", "x-component of the magnetic field"),
                                    "X_2": VariableInfo("By", "y-component of the magnetic field"),
                                    "X_3": VariableInfo("Bz", "z-component of the magnetic field")},
                         target=VariableInfo("E_n", "magnetic interaction energy"),
                         description=(
                             "Magnitude of the energy of a magnetic moment aligned with a magnetic field given by its"
                             " Cartesian components. It is proportional to the moment, depends on the field only "
                             "through its magnitude (unchanged under rotations of the field) and is even in each "
                             "component."
                         ),
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
                         variables={"X_0": VariableInfo("n_0", "number density without applied field"),
                                    "X_1": VariableInfo("kb", "Boltzmann constant"),
                                    "X_2": VariableInfo("T", "absolute temperature"),
                                    "X_3": VariableInfo("theta", "angle between the dipole moment and the field"),
                                    "X_4": VariableInfo("p_d", "permanent electric dipole moment of a molecule"),
                                    "X_5": VariableInfo("Ef", "applied electric field strength")},
                         target=VariableInfo("n", "number density of molecules with the given orientation"),
                         description=(
                             "Polar molecules in thermal equilibrium at temperature T in a weak electric field: the "
                             "target is the number per unit volume (and solid angle) whose dipoles point at angle "
                             "theta to the field, to first order in the field. It exceeds n_0 for dipoles along the "
                             "field and falls below it for the opposite orientation; the anisotropy grows with the "
                             "field and the dipole moment and decreases with temperature."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="I.39.1", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["1.5", "*", "X_0", "*", "X_1"], original_equation="E_n = 3/2*pr*V",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("pr", "pressure of the gas"),
                                    "X_1": VariableInfo("V", "volume of the gas")},
                         target=VariableInfo("E_n", "internal energy of the gas"),
                         description=(
                             "Internal (thermal) energy of a dilute monatomic gas at pressure pr filling volume V. It"
                             " is proportional to both the pressure and the volume."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.37.1", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "(", "1", "+", "X_2", ")", "*", "X_1"],
                         original_equation="E_n = mom*(1+chi)*B", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("mom", "magnetic moment"),
                                    "X_1": VariableInfo("B", "applied magnetic field strength"),
                                    "X_2": VariableInfo("chi", "magnetic susceptibility of the medium")},
                         target=VariableInfo("E_n", "magnetic energy of the moment"),
                         description=(
                             "Magnetic energy (magnitude) of a moment aligned with the field inside a linear magnetic"
                             " medium. The medium enhances the applied field: the energy is proportional to the "
                             "moment and the field, grows linearly with the susceptibility, and equals the vacuum "
                             "value at chi = 0."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.12.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_2", "/", "(", "4", "*", "pi", "*", "X_1", "*", "X_2", "^3", ")"],
                         original_equation="Ef = q1*r/(4*pi*epsilon*r**3)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("q1", "point charge creating the field"),
                                    "X_1": VariableInfo("epsilon", "vacuum permittivity (electric constant)"),
                                    "X_2": VariableInfo("r", "distance from the charge")},
                         target=VariableInfo("Ef", "electric field strength of the point charge"),
                         description=(
                             "Electric field at distance r from a point charge at rest in vacuum. It has the sign of "
                             "the charge, is proportional to it, and decreases monotonically with distance, diverging"
                             " at the charge and vanishing far away."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="II.27.18", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "^2"], original_equation="E_den = epsilon*Ef**2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("epsilon", "vacuum permittivity (electric constant)"),
                                    "X_1": VariableInfo("Ef", "electric field strength of the wave")},
                         target=VariableInfo("E_den", "electromagnetic energy density of the wave"),
                         description=(
                             "Total electromagnetic energy per unit volume (electric and magnetic parts together) of "
                             "a plane wave in vacuum. It is non-negative, even in the field strength and zero without"
                             " field."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.12.2", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "X_3", "/", "(", "4", "*", "pi", "*", "X_2", "*", "X_3",
                                       "^3", ")"], original_equation="F = q1*q2*r/(4*pi*epsilon*r**3)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("q1", "first point charge"),
                                    "X_1": VariableInfo("q2", "second point charge"),
                                    "X_2": VariableInfo("epsilon", "vacuum permittivity (electric constant)"),
                                    "X_3": VariableInfo("r", "distance between the charges")},
                         target=VariableInfo("F", "electrostatic force between the two charges (positive means repulsion)"),
                         description=(
                             "Force between two point charges at rest in vacuum, separated by distance r. Like "
                             "charges repel (positive) and unlike charges attract (negative); the force is symmetric "
                             "in the two charges, grows with each charge magnitude, and decreases monotonically with "
                             "distance, diverging as r -> 0 and vanishing as r -> infinity."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="III.13.18", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["2", "*", "X_0", "*", "X_1", "^2", "*", "X_2", "/", "(", "X_3", "/", "(", "2",
                                       "*", "pi", ")", ")"], original_equation="v = 2*E_n*d**2*k/(h/(2*pi))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("E_n", "coupling energy between neighbouring atoms"),
                                    "X_1": VariableInfo("d", "lattice spacing"),
                                    "X_2": VariableInfo("k", "wavenumber of the electron"),
                                    "X_3": VariableInfo("h", "Planck constant")},
                         target=VariableInfo("v", "velocity of the electron wave packet"),
                         description=(
                             "Velocity of an electron wave packet with small wavenumber moving along a one-"
                             "dimensional crystal lattice. It is zero at k = 0, proportional to k and to the coupling"
                             " energy, and increases with the lattice spacing."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="II.11.3", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "/", "(", "X_2", "*", "(", "X_3", "^2", "-", "X_4", "^2", ")",
                                       ")"], original_equation="x = q*Ef/(m*(omega_0**2-omega**2))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(3, 5, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False)],
                         variables={"X_0": VariableInfo("q", "charge of the bound particle"),
                                    "X_1": VariableInfo("Ef", "amplitude of the driving electric field"),
                                    "X_2": VariableInfo("m", "mass of the bound particle"),
                                    "X_3": VariableInfo("omega_0", "natural angular frequency of the binding"),
                                    "X_4": VariableInfo("omega", "angular frequency of the driving field")},
                         target=VariableInfo("x", "amplitude of the displacement of the bound charge"),
                         description=(
                             "Forced oscillation of an elastically bound, undamped charged particle driven by an "
                             "oscillating electric field. The amplitude is proportional to the charge and the field, "
                             "decreases with the mass, is even in the frequencies, positive (in phase) below "
                             "resonance, negative (in antiphase) above it, and diverges at resonance."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_6v, None, dataset_name="I.40.1", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "exp", "(", "u-", "X_1", "*", "X_4", "*", "X_2", "/", "(", "X_5",
                                       "*", "X_3", ")", ")"], original_equation="n = n_0*exp(-m*g*x/(kb*T))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("n_0", "number density at height zero"),
                                    "X_1": VariableInfo("m", "mass of one molecule"),
                                    "X_2": VariableInfo("x", "height"),
                                    "X_3": VariableInfo("T", "absolute temperature"),
                                    "X_4": VariableInfo("g", "gravitational acceleration"),
                                    "X_5": VariableInfo("kb", "Boltzmann constant")},
                         target=VariableInfo("n", "number density of molecules at height x"),
                         description=(
                             "Number density of gas molecules at height x in an isothermal atmosphere in uniform "
                             "gravity, given the density at the ground. It equals n_0 at x = 0, is proportional to "
                             "n_0, decreases monotonically with height (faster for heavier molecules and lower "
                             "temperature) and tends to zero far up."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="III.21.20", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["u-", "X_0", "*", "X_1", "*", "X_2", "/", "X_3"],
                         original_equation="j = -rho_c_0*q*A_vec/m", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("rho_c_0", "density of the superconducting charge carriers"),
                                    "X_1": VariableInfo("q", "charge of a carrier"),
                                    "X_2": VariableInfo("A_vec", "magnetic vector potential"),
                                    "X_3": VariableInfo("m", "mass of a carrier")},
                         target=VariableInfo("j", "electric current density"),
                         description=(
                             "Electric current density in a superconductor in a region with a given magnetic vector "
                             "potential (in a suitable gauge). It is proportional to the vector potential and points "
                             "opposite to it for positive density and charge, grows with the carrier density and "
                             "charge, and decreases with the carrier mass."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.43.16", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "X_2", "/", "X_3"],
                         original_equation="v = mu_drift*q*Volt/d", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("mu_drift", "mobility (drift velocity per unit applied force)"),
                                    "X_1": VariableInfo("q", "charge of the carrier"),
                                    "X_2": VariableInfo("Volt", "voltage applied across the sample"),
                                    "X_3": VariableInfo("d", "length of the sample along the field")},
                         target=VariableInfo("v", "drift velocity"),
                         description=(
                             "Mean drift velocity of charge carriers in a uniform electric field produced by a "
                             "voltage across a sample of length d. It is proportional to the mobility, the charge and"
                             " the voltage, and decreases with the length."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.15.10", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_2", "^2",
                                       ")"], original_equation="p = m_0*v/sqrt(1-v**2/c**2)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(3, 10, uses_negative=False)],
                         variables={"X_0": VariableInfo("m_0", "rest mass of the body"),
                                    "X_1": VariableInfo("v", "velocity of the body"),
                                    "X_2": VariableInfo("c", "speed of light in vacuum")},
                         target=VariableInfo("p", "momentum of the moving body"),
                         description=(
                             "Momentum of a body with given rest mass moving at velocity v, in special relativity. It"
                             " is proportional to the rest mass, odd in v, close to the classical value for slow "
                             "motion, and increases monotonically without bound as the speed approaches the speed of "
                             "light."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.30.5", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["arcsin", "(", "X_0", "/", "(", "X_2", "*", "X_1", ")", ")"],
                         original_equation="theta = arcsin(lambd/(n*d))", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 2, uses_negative=False),
                                                                   UniformSampling(2, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("lambd", "wavelength"),
                                    "X_1": VariableInfo("d", "spacing between neighbouring sources"),
                                    "X_2": VariableInfo("n", "number of sources")},
                         target=VariableInfo("theta", "angle of the first intensity zero"),
                         description=(
                             "Angle from the forward direction at which the radiation of a row of n equally spaced "
                             "coherent sources first vanishes, for a given wavelength. It increases with the "
                             "wavelength, decreases with the number of sources and their spacing, and exists only "
                             "while the wavelength does not exceed the total length of the row."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.50.26", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "(", "cos", "(", "X_1", "*", "X_2", ")", "+", "X_3", "*", "cos", "(",
                                       "X_1", "*", "X_2", ")", "^2", ")"],
                         original_equation="x = x1*(cos(omega*t)+alpha*cos(omega*t)**2)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False)],
                         variables={"X_0": VariableInfo("x1", "amplitude of the driving oscillation"),
                                    "X_1": VariableInfo("omega", "angular frequency of the driving oscillation"),
                                    "X_2": VariableInfo("t", "time"),
                                    "X_3": VariableInfo("alpha", "strength of the nonlinearity")},
                         target=VariableInfo("x", "displacement (output) of the system"),
                         description=(
                             "Output of a weakly nonlinear system driven by a pure oscillation of amplitude x1 and "
                             "angular frequency omega. It is proportional to x1, periodic in t with the driving "
                             "period, reduces to a pure oscillation when alpha = 0, and otherwise contains a constant"
                             " offset and a component at twice the driving frequency."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="I.12.11", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "(", "X_1", "+", "X_2", "*", "X_3", "*", "sin", "(", "X_4", ")",
                                       ")"], original_equation="F = q*(Ef+B*v*sin(theta))", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("q", "electric charge of the particle"),
                                    "X_1": VariableInfo("Ef", "electric field strength"),
                                    "X_2": VariableInfo("B", "magnetic field strength"),
                                    "X_3": VariableInfo("v", "speed of the particle"),
                                    "X_4": VariableInfo("theta", "angle between the velocity and the magnetic field")},
                         target=VariableInfo("F", "force on the moving charge"),
                         description=(
                             "Force on a point charge moving with speed v through an electric field and a magnetic "
                             "field, taken along a fixed direction in which both contributions add. It is "
                             "proportional to the charge; the magnetic contribution vanishes for a particle at rest "
                             "or moving along the magnetic field (theta = 0 or pi) and is largest when the motion is "
                             "perpendicular to the field."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="I.6.2", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["exp", "(", "u-", "(", "(", "X_1", "/", "X_0", ")", "^2", ")", "/", "2", ")",
                                       "/", "(", "sqrt", "(", "2", "*", "pi", ")", "*", "X_0", ")"],
                         original_equation="f = exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False)],
                         variables={"X_0": VariableInfo("sigma", "spread (standard deviation) of the random quantity"),
                                    "X_1": VariableInfo("theta", "value of the random quantity, measured from its mean")},
                         target=VariableInfo("f", "probability density"),
                         description=(
                             "Probability density of a continuous random quantity theta that scatters symmetrically "
                             "around zero with spread sigma. It is positive, even in theta, maximal at theta = 0 and "
                             "integrates to one over theta; a larger spread gives a lower and broader peak."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="I.14.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["0.5", "*", "X_0", "*", "X_1", "^2"], original_equation="U = 1/2*k_spring*x**2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("k_spring", "spring constant (stiffness)"),
                                    "X_1": VariableInfo("x", "extension or compression from the rest length")},
                         target=VariableInfo("U", "elastic potential energy stored in the spring"),
                         description=(
                             "Energy stored in an ideal spring stretched or compressed by x from its rest length. It "
                             "is non-negative, even in x, zero at the rest length, proportional to the stiffness, and"
                             " grows faster than linearly with the displacement."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.47.23", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["sqrt", "(", "X_0", "*", "X_1", "/", "X_2", ")"],
                         original_equation="c = sqrt(gamma*pr/rho)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("gamma", "adiabatic index (ratio of specific heats)"),
                                    "X_1": VariableInfo("pr", "pressure of the gas"),
                                    "X_2": VariableInfo("rho", "mass density of the gas")},
                         target=VariableInfo("c", "speed of sound"),
                         description=(
                             "Speed of sound waves in a gas. It increases with the adiabatic index and the pressure, "
                             "decreases with the density, and depends on pressure and density only through their "
                             "ratio."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.8.7", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["0.6", "*", "X_0", "^2", "/", "(", "4", "*", "pi", "*", "X_1", "*", "X_2", ")"],
                         original_equation="E_n = 3/5*q**2/(4*pi*epsilon*d)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("q", "total charge"),
                                    "X_1": VariableInfo("epsilon", "vacuum permittivity (electric constant)"),
                                    "X_2": VariableInfo("d", "radius of the sphere")},
                         target=VariableInfo("E_n", "electrostatic energy of the charged sphere"),
                         description=(
                             "Electrostatic energy needed to assemble a charge spread uniformly through a ball in "
                             "vacuum. It is positive, even in the charge, grows with its magnitude and decreases with"
                             " the radius."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="III.15.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_0", "/", "(", "2", "*", "pi", ")", ")", "^2", "/", "(", "2", "*", "X_1",
                                       "*", "X_2", "^2", ")"], original_equation="m = (h/(2*pi))**2/(2*E_n*d**2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("h", "Planck constant"),
                                    "X_1": VariableInfo("E_n", "coupling energy between neighbouring atoms"),
                                    "X_2": VariableInfo("d", "lattice spacing")},
                         target=VariableInfo("m", "effective mass of the electron"),
                         description=(
                             "Effective mass of an electron near the bottom of an energy band of a one-dimensional "
                             "crystal lattice. It is positive and decreases with both the coupling energy and the "
                             "lattice spacing."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.34.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "(", "1", "+", "(", "X_1", "/", "X_0", ")", ")", "/", "sqrt", "(", "1", "-",
                                       "X_1", "^2", "/", "X_0", "^2", ")", ")", "*", "X_2"],
                         original_equation="omega = ((1+v/c)/sqrt(1-v**2/c**2))*omega_0", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(3, 10, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("c", "speed of light in vacuum"),
                                    "X_1": VariableInfo("v", "speed of the source towards the observer"),
                                    "X_2": VariableInfo("omega_0", "angular frequency in the rest frame of the source")},
                         target=VariableInfo("omega", "observed angular frequency"),
                         description=(
                             "A light source emitting at angular frequency omega_0 in its own rest frame approaches "
                             "the observer head-on at speed v; special relativity is taken into account. The observed"
                             " frequency equals omega_0 at rest, increases monotonically with v and diverges as v "
                             "approaches the speed of light; reversing the direction of motion gives the reciprocal "
                             "frequency ratio."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="III.8.54", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["sin", "(", "X_0", "*", "X_1", "/", "(", "X_2", "/", "(", "2", "*", "pi", ")",
                                       ")", ")", "^2"], original_equation="prob = sin(E_n*t/(h/(2*pi)))**2",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(1, 4, uses_negative=False)],
                         variables={"X_0": VariableInfo("E_n", "coupling energy between the two states"),
                                    "X_1": VariableInfo("t", "time"),
                                    "X_2": VariableInfo("h", "Planck constant")},
                         target=VariableInfo("prob", "transition probability"),
                         description=(
                             "Probability that a two-state system with equal unperturbed energies, starting in one "
                             "state, is found in the other after time t when the states are coupled. It oscillates "
                             "periodically between 0 and 1, starts at zero at t = 0, and oscillates faster for "
                             "stronger coupling."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="I.26.2", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["arcsin", "(", "X_0", "*", "sin", "(", "X_1", ")", ")"],
                         original_equation="theta1 = arcsin(n*sin(theta2))", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(0, 1, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("n", "refractive index of the medium"),
                                    "X_1": VariableInfo("theta2", "angle of the ray to the surface normal inside the medium")},
                         target=VariableInfo("theta1", "angle of the ray to the surface normal outside the medium"),
                         description=(
                             "A light ray crosses the flat boundary between vacuum and a transparent medium of "
                             "refractive index n. The outside angle equals the inside angle when n = 1, is zero at "
                             "normal incidence, increases with n and with the inside angle, and exists only while the"
                             " ray can leave the medium (beyond a critical inside angle it is totally reflected)."
                         ),
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
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("m", "mass of the orbiting particle"),
                                    "X_1": VariableInfo("q", "magnitude of the charge of the particle and of the nucleus"),
                                    "X_2": VariableInfo("h", "Planck constant"),
                                    "X_3": VariableInfo("n", "principal quantum number"),
                                    "X_4": VariableInfo("epsilon", "vacuum permittivity (electric constant)")},
                         target=VariableInfo("E_n", "energy of the bound state"),
                         description=(
                             "Energy levels of a light charged particle bound to an opposite, much heavier charge of "
                             "equal magnitude (a hydrogen-like atom), labelled by the principal quantum number n. The"
                             " energies are negative (bound), rise towards zero as n grows, are proportional to the "
                             "mass, and become deeper for larger charge."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="III.4.33", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_0", "/", "(", "2", "*", "pi", ")", ")", "*", "X_1", "/", "(", "exp", "(",
                                       "(", "X_0", "/", "(", "2", "*", "pi", ")", ")", "*", "X_1", "/", "(", "X_2", "*",
                                       "X_3", ")", ")", "-", "1", ")"],
                         original_equation="E_n = (h/(2*pi))*omega/(exp((h/(2*pi))*omega/(kb*T))-1)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("h", "Planck constant"),
                                    "X_1": VariableInfo("omega", "angular frequency of the oscillator"),
                                    "X_2": VariableInfo("kb", "Boltzmann constant"),
                                    "X_3": VariableInfo("T", "absolute temperature")},
                         target=VariableInfo("E_n", "mean thermal energy of the oscillator"),
                         description=(
                             "Mean thermal energy (zero-point energy excluded) of a quantum oscillator in equilibrium"
                             " at temperature T. When the quantum energy is small compared with the thermal energy it"
                             " approaches the classical value, which is proportional to the temperature and "
                             "independent of frequency; it is exponentially suppressed in the opposite limit, "
                             "increases with temperature and decreases with frequency."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.34.1", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_2", "/", "(", "1", "-", "X_1", "/", "X_0", ")"],
                         original_equation="omega = omega_0/(1-v/c)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(3, 10, uses_negative=False),
                                                                   UniformSampling(1, 2, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("c", "speed of the wave (speed of light)"),
                                    "X_1": VariableInfo("v", "speed of the source towards the observer"),
                                    "X_2": VariableInfo("omega_0", "angular frequency emitted by the source")},
                         target=VariableInfo("omega", "observed angular frequency"),
                         description=(
                             "A source emitting at angular frequency omega_0 approaches a stationary observer with "
                             "speed v (source treated non-relativistically). The observed frequency equals omega_0 at"
                             " v = 0, increases with v, diverges as v approaches the wave speed, and is lowered for a"
                             " receding source (v < 0)."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="II.11.27", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_0", "*", "X_1", "/", "(", "1", "-", "(", "X_0", "*", "X_1", "/", "3",
                                       ")", ")", ")", "*", "X_2", "*", "X_3"],
                         original_equation="Pol = n*alpha/(1-(n*alpha/3))*epsilon*Ef", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(0, 1, uses_negative=False),
                                   UniformSampling(0, 1, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False)],
                         variables={"X_0": VariableInfo("n", "number density of molecules"),
                                    "X_1": VariableInfo("alpha", "molecular polarizability"),
                                    "X_2": VariableInfo("epsilon", "vacuum permittivity (electric constant)"),
                                    "X_3": VariableInfo("Ef", "macroscopic electric field strength")},
                         target=VariableInfo("Pol", "electric polarization"),
                         description=(
                             "Polarization of a dense isotropic dielectric, accounting for the field of the "
                             "neighbouring polarized molecules. It is proportional to the field, depends on density "
                             "and polarizability only through their product, grows faster than linearly with that "
                             "product and diverges at a finite value of it."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.13.34", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_2", "^2",
                                       ")"], original_equation="j = rho_c_0*v/sqrt(1-v**2/c**2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(3, 10, uses_negative=False)],
                         variables={"X_0": VariableInfo("rho_c_0", "charge density in the rest frame of the charges"),
                                    "X_1": VariableInfo("v", "velocity of the charges"),
                                    "X_2": VariableInfo("c", "speed of light in vacuum")},
                         target=VariableInfo("j", "electric current density"),
                         description=(
                             "Current density carried by a charge distribution moving at velocity v, given its "
                             "density in its own rest frame (special relativity). It is proportional to the rest-"
                             "frame density, odd in v, close to the classical value for slow motion, and grows "
                             "without bound as the speed approaches the speed of light."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="II.4.23", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "/", "(", "4", "*", "pi", "*", "X_1", "*", "X_2", ")"],
                         original_equation="Volt = q/(4*pi*epsilon*r)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("q", "point charge"),
                                    "X_1": VariableInfo("epsilon", "vacuum permittivity (electric constant)"),
                                    "X_2": VariableInfo("r", "distance from the charge")},
                         target=VariableInfo("Volt", "electric potential"),
                         description=(
                             "Electric potential at distance r from a point charge at rest in vacuum, taken as zero "
                             "at infinity. It has the sign of the charge, is proportional to it, and its magnitude "
                             "decreases monotonically with distance, diverging at the charge."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.32.5", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "^2", "*", "X_1", "^2", "/", "(", "6", "*", "pi", "*", "X_2", "*", "X_3",
                                       "^3", ")"], original_equation="Pwr = q**2*a**2/(6*pi*epsilon*c**3)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("q", "electric charge of the particle"),
                                    "X_1": VariableInfo("a", "acceleration of the particle"),
                                    "X_2": VariableInfo("epsilon", "vacuum permittivity (electric constant)"),
                                    "X_3": VariableInfo("c", "speed of light in vacuum")},
                         target=VariableInfo("Pwr", "total radiated power"),
                         description=(
                             "Total power radiated by a point charge that accelerates while moving slowly compared "
                             "with light, in vacuum. It is non-negative, even in both the charge and the "
                             "acceleration, zero for an unaccelerated charge, and grows with both."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="I.13.12", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_4", "*", "X_0", "*", "X_1", "*", "(", "1", "/", "X_3", "-", "1", "/", "X_2",
                                       ")"], original_equation="U = G*m1*m2*(1/r2-1/r1)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("m1", "mass of the first body"),
                                    "X_1": VariableInfo("m2", "mass of the second body"),
                                    "X_2": VariableInfo("r1", "initial distance between the bodies"),
                                    "X_3": VariableInfo("r2", "final distance between the bodies"),
                                    "X_4": VariableInfo("G", "gravitational constant")},
                         target=VariableInfo("U", "change in gravitational potential energy (energy released)"),
                         description=(
                             "Energy released when the separation of two gravitating masses changes from r1 to r2. It"
                             " is positive when the bodies move closer (r2 < r1), zero for r1 = r2, changes sign when"
                             " r1 and r2 are exchanged, and is proportional to each mass."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="II.2.42", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "(", "X_2", "-", "X_1", ")", "*", "X_3", "/", "X_4"],
                         original_equation="Pwr = kappa*(T2-T1)*A/d", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("kappa", "thermal conductivity of the slab"),
                                    "X_1": VariableInfo("T1", "temperature of the first face"),
                                    "X_2": VariableInfo("T2", "temperature of the second face"),
                                    "X_3": VariableInfo("A", "area of the slab"),
                                    "X_4": VariableInfo("d", "thickness of the slab")},
                         target=VariableInfo("Pwr", "rate of heat flow through the slab"),
                         description=(
                             "Steady rate of heat flow through a flat slab whose two faces are held at temperatures "
                             "T1 and T2. It is proportional to the conductivity, the area and the temperature "
                             "difference, decreases with the thickness, is zero for equal temperatures and changes "
                             "sign when the temperatures are exchanged."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.27.6", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["1", "/", "(", "1", "/", "X_0", "+", "X_2", "/", "X_1", ")"],
                         original_equation="foc = 1/(1/d1+n/d2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("d1", "distance of the object from the surface (outside the medium)"),
                                    "X_1": VariableInfo("d2", "distance of the image from the surface (inside the medium)"),
                                    "X_2": VariableInfo("n", "refractive index of the medium")},
                         target=VariableInfo("foc", "focal length of the refracting surface"),
                         description=(
                             "A single curved boundary between air and a transparent medium of refractive index n "
                             "images an object at distance d1 in front of it to an image at distance d2 inside the "
                             "medium. For positive distances the target is positive, increases with both distances, "
                             "and is smaller than d1 and smaller than d2/n."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="III.14.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "(", "exp", "(", "X_1", "*", "X_2", "/", "(", "X_3", "*", "X_4", ")",
                                       ")", "-", "1", ")"], original_equation="I = I_0*(exp(q*Volt/(kb*T))-1)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False),
                                   UniformSampling(1, 2, uses_negative=False)],
                         variables={"X_0": VariableInfo("I_0", "reverse saturation current"),
                                    "X_1": VariableInfo("q", "charge of the carriers"),
                                    "X_2": VariableInfo("Volt", "applied voltage (forward bias positive)"),
                                    "X_3": VariableInfo("kb", "Boltzmann constant"),
                                    "X_4": VariableInfo("T", "absolute temperature")},
                         target=VariableInfo("I", "electric current through the junction"),
                         description=(
                             "Current through a semiconductor p-n junction at temperature T under an applied voltage."
                             " It is zero at zero voltage, grows exponentially for forward bias and saturates at -I_0"
                             " for reverse bias (strongly rectifying); it is proportional to I_0."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.18.12", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "sin", "(", "X_2", ")"],
                         original_equation="tau = r*F*sin(theta)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(0, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("r", "distance from the pivot to the point where the force acts"),
                                    "X_1": VariableInfo("F", "magnitude of the applied force"),
                                    "X_2": VariableInfo("theta", "angle between the lever arm and the force")},
                         target=VariableInfo("tau", "torque about the pivot"),
                         description=(
                             "Torque produced by a force applied at distance r from a pivot. It is proportional to "
                             "the distance and to the force, zero when the force points along the lever arm (theta = "
                             "0 or pi) and largest when the force is perpendicular to the arm."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.18.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "X_2", "*", "sin", "(", "X_3", ")"],
                         original_equation="L = m*r*v*sin(theta)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("m", "mass of the particle"),
                                    "X_1": VariableInfo("r", "distance of the particle from the origin"),
                                    "X_2": VariableInfo("v", "speed of the particle"),
                                    "X_3": VariableInfo("theta", "angle between the position vector and the velocity")},
                         target=VariableInfo("L", "magnitude of the angular momentum about the origin"),
                         description=(
                             "Angular momentum about the origin of a point mass moving with speed v at distance r. It"
                             " is proportional to the mass, the distance and the speed, zero for purely radial motion"
                             " (theta = 0 or pi) and largest for motion perpendicular to the position vector."
                         ),
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
                         variables={"X_0": VariableInfo("q", "point charge"),
                                    "X_1": VariableInfo("epsilon", "vacuum permittivity (electric constant)"),
                                    "X_2": VariableInfo("r", "distance from the charge (at its retarded position)"),
                                    "X_3": VariableInfo("v", "velocity component of the charge towards the observer"),
                                    "X_4": VariableInfo("c", "speed of light in vacuum")},
                         target=VariableInfo("Volt", "electric potential of the moving charge"),
                         description=(
                             "Electric potential at an observation point produced by a point charge in uniform "
                             "motion, with the distance taken to the position the charge had when the signal left it."
                             " It reduces to the static value at v = 0, is enhanced for an approaching charge and "
                             "reduced for a receding one, and diverges as the approach speed reaches the speed of "
                             "light."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="II.38.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "/", "(", "2", "*", "(", "1", "+", "X_1", ")", ")"],
                         original_equation="mu_S = Y/(2*(1+sigma))", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 5, uses_negative=False),
                                                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("Y", "Young's modulus of the material"),
                                    "X_1": VariableInfo("sigma", "Poisson's ratio of the material")},
                         target=VariableInfo("mu_S", "shear modulus"),
                         description=(
                             "Shear modulus of an isotropic elastic material, given its Young's modulus and Poisson's"
                             " ratio. It is proportional to Young's modulus, decreases with Poisson's ratio, and is "
                             "half of Young's modulus when Poisson's ratio is zero."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.34.8", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "X_2", "/", "X_3"], original_equation="omega = q*v*B/p",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("q", "electric charge of the particle"),
                                    "X_1": VariableInfo("v", "speed of the particle"),
                                    "X_2": VariableInfo("B", "magnetic field strength"),
                                    "X_3": VariableInfo("p", "momentum of the particle")},
                         target=VariableInfo("omega", "angular frequency of the circular motion"),
                         description=(
                             "A charged particle circles in a uniform magnetic field perpendicular to its velocity. "
                             "Its angular frequency is proportional to the charge, the speed and the field, and "
                             "decreases with increasing momentum."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.8.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["sqrt", "(", "(", "X_1", "-", "X_0", ")", "^2", "+", "(", "X_3", "-", "X_2", ")",
                                       "^2", ")"], original_equation="d = sqrt((x2-x1)**2+(y2-y1)**2)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("x1", "x-coordinate of the first point"),
                                    "X_1": VariableInfo("x2", "x-coordinate of the second point"),
                                    "X_2": VariableInfo("y1", "y-coordinate of the first point"),
                                    "X_3": VariableInfo("y2", "y-coordinate of the second point")},
                         target=VariableInfo("d", "distance between the two points"),
                         description=(
                             "Separation of two points in a flat plane, given their Cartesian coordinates. It is non-"
                             "negative, zero only when the points coincide, symmetric under exchanging the two "
                             "points, and unchanged when both points are shifted or rotated together."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="II.6.15b", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["(", "X_1", "/", "(", "4", "*", "pi", "*", "X_0", ")", ")", "*", "3", "*", "cos",
                                       "(", "X_2", ")", "*", "sin", "(", "X_2", ")", "/", "X_3", "^3"],
                         original_equation="Ef = p_d/(4*pi*epsilon)*3*cos(theta)*sin(theta)/r**3",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False),
                                   UniformSampling(1, 3, uses_negative=False)],
                         variables={"X_0": VariableInfo("epsilon", "vacuum permittivity (electric constant)"),
                                    "X_1": VariableInfo("p_d", "electric dipole moment"),
                                    "X_2": VariableInfo("theta", "angle between the dipole axis and the direction to the observation point"),
                                    "X_3": VariableInfo("r", "distance from the dipole")},
                         target=VariableInfo("Ef", "electric field component perpendicular to the dipole axis"),
                         description=(
                             "Electric field component perpendicular to the axis of a point dipole, at distance r and"
                             " angle theta from the axis. It is proportional to the dipole moment, zero on the axis "
                             "and in the plane perpendicular to it, largest in magnitude at 45 degrees, changes sign "
                             "under theta -> pi - theta, and decays rapidly with distance."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="I.12.1", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1"], original_equation="F = mu*Nn", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("mu", "coefficient of friction between the surfaces"),
                                    "X_1": VariableInfo("Nn", "normal force pressing the surfaces together")},
                         target=VariableInfo("F", "friction force"),
                         description=(
                             "Friction force on a block sliding over a flat surface, given the friction coefficient "
                             "and the normal force. It grows with both, vanishes when either is zero, and does not "
                             "depend on the contact area or the sliding speed."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="II.34.29b", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_3", "*", "X_4", "*", "X_2", "/", "(", "X_1", "/", "(", "2", "*",
                                       "pi", ")", ")"], original_equation="E_n = g_*mom*B*Jz/(h/(2*pi))",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("g_", "g-factor (Lande factor) of the state"),
                                    "X_1": VariableInfo("h", "Planck constant"),
                                    "X_2": VariableInfo("Jz", "component of the angular momentum along the field"),
                                    "X_3": VariableInfo("mom", "unit of magnetic moment"),
                                    "X_4": VariableInfo("B", "magnetic field strength")},
                         target=VariableInfo("E_n", "magnetic energy of the state"),
                         description=(
                             "Energy shift of an atomic state in a weak uniform magnetic field, given its angular "
                             "momentum component along the field. It is proportional to the g-factor, the magnetic-"
                             "moment unit, the field and the angular momentum component, changes sign with that "
                             "component and vanishes without field."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.13.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["0.5", "*", "X_0", "*", "(", "X_1", "^2", "+", "X_2", "^2", "+", "X_3", "^2",
                                       ")"], original_equation="K = 1/2*m*(v**2+u**2+w**2)", success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("m", "mass of the particle"),
                                    "X_1": VariableInfo("v", "first Cartesian velocity component"),
                                    "X_2": VariableInfo("u", "second Cartesian velocity component"),
                                    "X_3": VariableInfo("w", "third Cartesian velocity component")},
                         target=VariableInfo("K", "kinetic energy"),
                         description=(
                             "Energy of motion of a point mass, given its three Cartesian velocity components. It is "
                             "non-negative, proportional to the mass, even in each component, symmetric under "
                             "permuting the components and depends on them only through the speed; it is zero only "
                             "for a body at rest."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="I.39.22", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_3", "*", "X_1", "/", "X_2"], original_equation="pr = n*kb*T/V",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("n", "number of molecules"),
                                    "X_1": VariableInfo("T", "absolute temperature"),
                                    "X_2": VariableInfo("V", "volume of the gas"),
                                    "X_3": VariableInfo("kb", "Boltzmann constant")},
                         target=VariableInfo("pr", "pressure of the gas"),
                         description=(
                             "Pressure of a dilute gas of n molecules at temperature T in a container of volume V. It"
                             " is proportional to the number of molecules and to the temperature, decreases with the "
                             "volume, and does not depend on the kind of molecule."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="I.14.3", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "*", "X_1", "*", "X_2"], original_equation="U = m*g*z",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False),
                                   UniformSampling(1, 5, uses_negative=False)],
                         variables={"X_0": VariableInfo("m", "mass of the body"),
                                    "X_1": VariableInfo("g", "gravitational acceleration"),
                                    "X_2": VariableInfo("z", "height above the reference level")},
                         target=VariableInfo("U", "gravitational potential energy"),
                         description=(
                             "Potential energy of a body raised to height z near the ground, in a uniform "
                             "gravitational field. It is proportional to the mass and to the height, and zero at the "
                             "reference level."
                         ),
                         data_source=data_source)

    # fmt: on
