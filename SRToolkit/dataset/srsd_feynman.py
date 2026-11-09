"""SRSD Feynman symbolic regression benchmark."""

from typing import Optional

from SRToolkit.utils.symbol_library import SymbolLibrary

from .background_knowledge import VariableInfo
from .data_source import FallbackSource, SampleSource, UrlSource
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
    """The SRSD Feynman symbolic regression benchmark.

    Contains 120 physics equations from the Feynman Symbolic Regression Dataset with
    per-variable sampling strategies (log-uniform, linear, or integer with sign constraints).
    Data is generated on first instantiation and cached as .npz files for subsequent use.

    Warning:
        Seven datasets currently have a known data problem, both in the downloaded data and
        with ``force_generate=True``:

        - ``SRSD II.24.17``, ``SRSD Bonus 4`` and ``SRSD Bonus 7``: for part
          of the sampled points (about 25–85%) the ground truth takes the square root of a
          negative number, so ``y`` contains NaN. The ground truth itself then scores NaN, so
          no approach can be counted as solving these datasets.
        - ``SRSD II.6.15a``, ``SRSD II.6.15b``, ``SRSD III.19.51`` and ``SRSD Bonus 12``:
          ``y`` reaches 1e16 to 1e24, so rounding alone gives the ground truth an RMSE between
          about 1e-3 and 3e6. Whether it counts as a success depends on ``success_threshold``.

        Leave these datasets out, or read their results with care, until this is fixed.

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

    __data_version__ = "1.0.1"

    def __init__(
        self,
        n_samples: int = 10000,
        seed: Optional[int] = 42,
        force_generate: bool = False,
    ):
        super().__init__("SRSD_Feynman", version="1.0.1")
        self._n_samples = n_samples
        self._seed = seed
        self._force_generate = force_generate
        self._populate()

    def _populate(self):
        # fmt: off
        seed = None
        url = "https://raw.githubusercontent.com/smeznar/SymbolicRegressionToolkit/master/data/srsd.zip"

        # Each dataset's data_source prefers the canonical archive (so every machine
        # benchmarks on identical inputs) and falls back to regenerating from that dataset's
        # samplers if the download is unavailable. The chain is serialised in the config, so
        # the preference travels to grid recipes, exports, and cold workers. SRSD sampling is
        # seedless (seed=None), so the canonical download matters most here. With
        # force_generate it is sampling-only — the canonical archive is never consulted.
        if self._force_generate:
            data_source = SampleSource(n_samples=self._n_samples, seed=seed)
        else:
            data_source = FallbackSource(
                [UrlSource(url), SampleSource(n_samples=self._n_samples, seed=seed)]
            )

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
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(0.1, 10.0), LogUniformSampling(0.1, 10.0, uses_negative=False)],
                         variables={"X_0": VariableInfo("theta", "value of the random quantity, measured from its mean"),
                                    "X_1": VariableInfo("sigma", "spread (standard deviation) of the random quantity")},
                         target=VariableInfo("f", "probability density"),
                         description=(
                             "Probability density of a continuous random quantity theta that scatters symmetrically "
                             "around zero with spread sigma. It is positive, even in theta, maximal at theta = 0 and "
                             "integrates to one over theta; a larger spread gives a lower and broader peak."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_1v, None, dataset_name="SRSD I.6.20a", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['exp', '(', '(', 'u-', '(', 'X_0', '^2', ')', ')', '/', '2', ')', '/', 'sqrt',
                                       '(', '2', '*', 'pi', ')'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[LogUniformSampling(0.1, 10.0)],
                         variables={"X_0": VariableInfo("theta", "value of the random quantity, measured from its mean in units of its spread")},
                         target=VariableInfo("f", "probability density"),
                         description=(
                             "Probability density of a continuous random quantity that has been shifted to zero mean "
                             "and scaled to unit spread. The density is positive, symmetric about zero, has a single "
                             "maximum at zero, falls off faster than any power for large |theta|, and integrates to "
                             "one."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.6.20b", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['exp', '(', 'u-', '(', '(', '(', 'X_0', '-', 'X_1', ')', '/', 'X_2', ')', '^2',
                                       ')', '/', '2', ')', '/', '(', 'sqrt', '(', '2', '*', 'pi', ')', '*', 'X_2', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("theta", "value of the random quantity"),
                                    "X_1": VariableInfo("theta1", "mean (centre) of the distribution"),
                                    "X_2": VariableInfo("sigma", "spread (standard deviation) of the random quantity")},
                         target=VariableInfo("f", "probability density"),
                         description=(
                             "Probability density of a continuous random quantity theta that scatters symmetrically "
                             "around the mean theta1 with spread sigma. It is positive, maximal at theta = theta1, "
                             "symmetric under reflection about theta1, depends on theta and theta1 only through their"
                             " difference, and integrates to one over theta; a larger spread gives a lower and "
                             "broader peak."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.8.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sqrt', '(', '(', '(', '(', 'X_0', '-', 'X_1', ')', ')', '^2', '+', '(', '(',
                                       'X_2', '-', 'X_3', ')', ')', '^2', ')', ')'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
            ],
                         variables={"X_0": VariableInfo("x2", "x-coordinate of the second point"),
                                    "X_1": VariableInfo("x1", "x-coordinate of the first point"),
                                    "X_2": VariableInfo("y2", "y-coordinate of the second point"),
                                    "X_3": VariableInfo("y1", "y-coordinate of the first point")},
                         target=VariableInfo("d", "distance between the two points"),
                         description=(
                             "Separation of two points in a flat plane, given their Cartesian coordinates. It is non-"
                             "negative, zero only when the points coincide, symmetric under exchanging the two "
                             "points, and unchanged when both points are shifted or rotated together."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_8v, None, dataset_name="SRSD I.9.18", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.GRAVITATIONAL_CONSTANT}', '*', 'X_0', '*', 'X_1', '/', '(', '(', 'X_2',
                                       '-', 'X_3', ')', '^2', '+', '(', 'X_4', '-', 'X_5', ')', '^2', '+', '(', 'X_6',
                                       '-', 'X_7', ')', '^2', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1.0, 1000.0, uses_negative=False),
                LogUniformSampling(1.0, 1000.0, uses_negative=False),
                LogUniformSampling(1.0, 10.0),
                LogUniformSampling(1.0, 10.0),
                LogUniformSampling(1.0, 10.0),
                LogUniformSampling(1.0, 10.0),
                LogUniformSampling(1.0, 10.0),
                LogUniformSampling(1.0, 10.0),
            ],
                         variables={"X_0": VariableInfo("m1", "mass of the first body"),
                                    "X_1": VariableInfo("m2", "mass of the second body"),
                                    "X_2": VariableInfo("x2", "x-coordinate of the second body"),
                                    "X_3": VariableInfo("x1", "x-coordinate of the first body"),
                                    "X_4": VariableInfo("y2", "y-coordinate of the second body"),
                                    "X_5": VariableInfo("y1", "y-coordinate of the first body"),
                                    "X_6": VariableInfo("z2", "z-coordinate of the second body"),
                                    "X_7": VariableInfo("z1", "z-coordinate of the first body")},
                         target=VariableInfo("F", "magnitude of the attractive force between the two masses"),
                         description=(
                             "Gravitational attraction between two point masses at given positions in three-"
                             "dimensional space. The force is positive, grows with each mass, is symmetric under "
                             "exchanging the two bodies, depends on the positions only through their separation, "
                             "diverges as the bodies approach each other and vanishes at large separation."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.10.7", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', 'sqrt', '(', '1', '-', 'X_1', '^2', '/', f'{self.SPEED_OF_LIGHT}',
                                       '^2', ')'], success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(100000.0, 100000000.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("m_0", "rest mass of the body"),
                                    "X_1": VariableInfo("v", "speed of the body")},
                         target=VariableInfo("m", "inertial mass of the moving body"),
                         description=(
                             "Effective inertial mass of a body moving at speed v, given its rest mass, in special "
                             "relativity. It equals the rest mass when the body is at rest, is even in v, increases "
                             "monotonically with speed and diverges as the speed approaches the speed of light; it is"
                             " defined only for speeds below the speed of light."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_6v, None, dataset_name="SRSD I.11.19", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '+', 'X_2', '*', 'X_3', '+', 'X_4', '*', 'X_5'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
            ],
                         variables={"X_0": VariableInfo("x1", "x-component of the force"),
                                    "X_1": VariableInfo("y1", "x-component of the displacement"),
                                    "X_2": VariableInfo("x2", "y-component of the force"),
                                    "X_3": VariableInfo("y2", "y-component of the displacement"),
                                    "X_4": VariableInfo("x3", "z-component of the force"),
                                    "X_5": VariableInfo("y3", "z-component of the displacement")},
                         target=VariableInfo("A", "work done by the force over the displacement"),
                         description=(
                             "A constant force with Cartesian components (x1, x2, x3) acts on a body while it is "
                             "displaced by (y1, y2, y3); the target is the work done. It is linear in each component "
                             "of the force and of the displacement, unchanged when force and displacement are "
                             "exchanged or rotated together, and zero when they are perpendicular."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.12.1", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1'], success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(0.01, 1.0, uses_negative=False),
                                   LogUniformSampling(0.01, 1.0, uses_negative=False)],
                         variables={"X_0": VariableInfo("mu", "coefficient of friction between the surfaces"),
                                    "X_1": VariableInfo("Nn", "normal force pressing the surfaces together")},
                         target=VariableInfo("F", "friction force"),
                         description=(
                             "Friction force on a block sliding over a flat surface, given the friction coefficient "
                             "and the normal force. It grows with both, vanishes when either is zero, and does not "
                             "depend on the contact area or the sliding speed."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.12.2", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '/', '(', '4', '*', 'pi', '*', f'{self.ELECTRIC_CONSTANT}',
                                       '*', 'X_2', '^2', ')'],
                         success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("q1", "first point charge"),
                                    "X_1": VariableInfo("q2", "second point charge"),
                                    "X_2": VariableInfo("r", "distance between the charges")},
                         target=VariableInfo("F", "electrostatic force between the two charges (positive means repulsion)"),
                         description=(
                             "Force between two point charges at rest in vacuum, separated by distance r. Like "
                             "charges repel (positive) and unlike charges attract (negative); the force is symmetric "
                             "in the two charges, grows with each charge magnitude, and decreases monotonically with "
                             "distance, diverging as r -> 0 and vanishing as r -> infinity."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.12.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', '4', '*', 'pi', '*', f'{self.ELECTRIC_CONSTANT}', '*', 'X_1',
                                       '^2', ')'], success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(0.1, 10.0), LogUniformSampling(0.1, 10.0, uses_negative=False)],
                         variables={"X_0": VariableInfo("q1", "point charge creating the field"),
                                    "X_1": VariableInfo("r", "distance from the charge")},
                         target=VariableInfo("Ef", "electric field strength of the point charge"),
                         description=(
                             "Electric field at distance r from a point charge at rest in vacuum. It has the sign of "
                             "the charge, is proportional to it, and decreases monotonically with distance, diverging"
                             " at the charge and vanishing far away."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.12.5", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1'], success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(0.1, 10.0), LogUniformSampling(0.1, 10.0)],
                         variables={"X_0": VariableInfo("q2", "test charge"),
                                    "X_1": VariableInfo("Ef", "electric field strength at the position of the charge")},
                         target=VariableInfo("F", "electric force on the charge"),
                         description=(
                             "Force exerted on a small test charge by a given electric field. It is proportional to "
                             "both the charge and the field, reverses sign with either, and vanishes when either is "
                             "zero."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="SRSD I.12.11", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', '(', 'X_1', '+', 'X_2', '*', 'X_3', '*', 'sin', '(', 'X_4', ')',
                                       ')'], success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                UniformSampling(0.0, 1.5707963267948966, uses_negative=False),
            ],
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
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.13.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['0.5', '*', 'X_0', '*', '(', 'X_1', '^2', '+', 'X_2', '^2', '+', 'X_3', '^2',
                                       ')'], success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
            ],
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
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.13.12", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.GRAVITATIONAL_CONSTANT}', '*', 'X_0', '*', 'X_1', '*', '(', '1', '/',
                                       'X_2', '-', '1', '/', 'X_3', ')'],
                         success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.01, 1.0, uses_negative=False),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("m1", "mass of the first body"),
                                    "X_1": VariableInfo("m2", "mass of the second body"),
                                    "X_2": VariableInfo("r2", "final distance between the bodies"),
                                    "X_3": VariableInfo("r1", "initial distance between the bodies")},
                         target=VariableInfo("U", "change in gravitational potential energy (energy released)"),
                         description=(
                             "Energy released when the separation of two gravitating masses changes from r1 to r2. It"
                             " is positive when the bodies move closer (r2 < r1), zero for r1 = r2, changes sign when"
                             " r1 and r2 are exchanged, and is proportional to each mass."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.14.3", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.GRAVITATIONAL_ACCELERATION}', '*', 'X_0', '*', 'X_1'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(0.01, 1.0, uses_negative=False), LogUniformSampling(0.01, 1.0)],
                         variables={"X_0": VariableInfo("m", "mass of the body"),
                                    "X_1": VariableInfo("z", "height above the reference level")},
                         target=VariableInfo("U", "gravitational potential energy"),
                         description=(
                             "Potential energy of a body raised to height z near the ground, in a uniform "
                             "gravitational field. It is proportional to the mass and to the height, and zero at the "
                             "reference level."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.14.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['0.5', '*', 'X_0', '*', 'X_1', '^2'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(100.0, 10000.0, uses_negative=False),
                                   LogUniformSampling(0.01, 1.0)],
                         variables={"X_0": VariableInfo("k_spring", "spring constant (stiffness)"),
                                    "X_1": VariableInfo("x", "extension or compression from the rest length")},
                         target=VariableInfo("U", "elastic potential energy stored in the spring"),
                         description=(
                             "Energy stored in an ideal spring stretched or compressed by x from its rest length. It "
                             "is non-negative, even in x, zero at the rest length, proportional to the stiffness, and"
                             " grows faster than linearly with the displacement."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.15.10", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '/', 'sqrt', '(', '1', '-', 'X_1', '^2', '/',
                                       f'{self.SPEED_OF_LIGHT}', '^2', ')'],
                         success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(0.01, 1.0, uses_negative=False),
                                   LogUniformSampling(100000.0, 10000000.0)],
                         variables={"X_0": VariableInfo("m_0", "rest mass of the body"),
                                    "X_1": VariableInfo("v", "velocity of the body")},
                         target=VariableInfo("p", "momentum of the moving body"),
                         description=(
                             "Momentum of a body with given rest mass moving at velocity v, in special relativity. It"
                             " is proportional to the rest mass, odd in v, close to the classical value for slow "
                             "motion, and increases monotonically without bound as the speed approaches the speed of "
                             "light."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.15.3t", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['(', 'X_0', '-', 'X_1', '*', 'X_2', '/', f'{self.SPEED_OF_LIGHT}', '^2', ')',
                                       '/', 'sqrt', '(', '1', '-', 'X_1', '^2', '/', f'{self.SPEED_OF_LIGHT}', '^2',
                                       ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-06, 0.0001, uses_negative=False),
                LogUniformSampling(100000.0, 10000000.0),
                LogUniformSampling(1.0, 100.0),
            ],
                         variables={"X_0": VariableInfo("t", "time of the event in the rest frame"),
                                    "X_1": VariableInfo("u", "velocity of the moving frame along x"),
                                    "X_2": VariableInfo("x", "position coordinate of the event in the rest frame")},
                         target=VariableInfo("t1", "time coordinate of the event in the moving frame"),
                         description=(
                             "An event has position x and time t in one inertial frame; the target is its time in a "
                             "second frame moving with velocity u along the x-axis (origins coinciding at t = 0). It "
                             "equals t when u = 0, approaches t when u is much smaller than the speed of light, and "
                             "is defined only for |u| below the speed of light."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.15.3x", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['(', 'X_0', '-', 'X_1', '*', 'X_2', ')', '/', 'sqrt', '(', '1', '-', 'X_1', '^2',
                                       '/', f'{self.SPEED_OF_LIGHT}', '^2', ')'],
                         success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1.0, 100.0),
                LogUniformSampling(1000000.0, 100000000.0),
                LogUniformSampling(1e-06, 0.0001, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("x", "position coordinate of the event in the rest frame"),
                                    "X_1": VariableInfo("u", "velocity of the moving frame along x"),
                                    "X_2": VariableInfo("t", "time of the event in the rest frame")},
                         target=VariableInfo("x1", "position coordinate of the event in the moving frame"),
                         description=(
                             "An event has position x and time t in one inertial frame; the target is its position in"
                             " a second frame moving with velocity u along the x-axis (origins coinciding at t = 0). "
                             "It reduces to x when u = 0, approaches the classical shift x - u*t when u is much "
                             "smaller than the speed of light, and is defined only for |u| below the speed of light."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.16.6", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['(', 'X_0', '+', 'X_1', ')', '/', '(', '1', '+', 'X_0', '*', 'X_1', '/',
                                       f'{self.SPEED_OF_LIGHT}', '^2', ')'],
                         success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(1000000.0, 100000000.0),
                                   LogUniformSampling(1000000.0, 100000000.0)],
                         variables={"X_0": VariableInfo("u", "velocity of the moving frame relative to the lab frame"),
                                    "X_1": VariableInfo("v", "velocity of the body relative to the moving frame")},
                         target=VariableInfo("v1", "velocity of the body relative to the lab frame"),
                         description=(
                             "A body moves with velocity v relative to a frame that itself moves with velocity u "
                             "(parallel) relative to the lab; the target is the body's velocity in the lab. It is "
                             "symmetric in u and v, close to their plain sum when both are small compared with the "
                             "speed of light, never exceeds the speed of light when both inputs are below it, and "
                             "equals the speed of light when either input does."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.18.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['(', 'X_0', '*', 'X_1', '+', 'X_2', '*', 'X_3', ')', '/', '(', 'X_0', '+', 'X_2',
                                       ')'], success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0),
            ],
                         variables={"X_0": VariableInfo("m1", "mass of the first body"),
                                    "X_1": VariableInfo("r1", "position of the first body"),
                                    "X_2": VariableInfo("m2", "mass of the second body"),
                                    "X_3": VariableInfo("r2", "position of the second body")},
                         target=VariableInfo("r", "position of the centre of mass"),
                         description=(
                             "Position of the mass-weighted centre of two point masses on a line. It lies between r1 "
                             "and r2, closer to the heavier body, is unchanged when the two bodies (mass and position"
                             " together) are exchanged or when both masses are scaled by the same factor, and equals "
                             "the midpoint for equal masses."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.18.12", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '*', 'sin', '(', 'X_2', ')'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ],
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
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.18.16", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '*', 'X_2', '*', 'sin', '(', 'X_3', ')'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ],
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
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.24.6", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['0.25', '*', 'X_0', '*', '(', 'X_1', '^2', '+', 'X_2', '^2', ')', '*', 'X_3',
                                       '^2'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.1, 10.0),
            ],
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
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.25.13", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', 'X_1'], success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[LogUniformSampling(1e-05, 0.001),
                                                                   LogUniformSampling(1e-05, 0.001,
                                                                                      uses_negative=False)],
                         variables={"X_0": VariableInfo("q", "charge stored on the capacitor"),
                                    "X_1": VariableInfo("C", "capacitance")},
                         target=VariableInfo("Volt", "voltage across the capacitor"),
                         description=(
                             "Voltage across a capacitor holding charge q. It is proportional to the charge and "
                             "decreases with increasing capacitance."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.26.2", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sin', '(', 'X_0', ')', '/', 'sin', '(', 'X_1', ')'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                UniformSampling(0, 1.5707963267948966, uses_negative=False),
                UniformSampling(0, 1.5707963267948966, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("theta1", "angle of the ray to the surface normal outside the medium (in vacuum)"),
                                    "X_1": VariableInfo("theta2", "angle of the ray to the surface normal inside the medium")},
                         target=VariableInfo("n", "refractive index of the medium"),
                         description=(
                             "A light ray enters a transparent medium from vacuum through a flat boundary; theta1 and"
                             " theta2 are the angles of the incident and refracted ray to the surface normal. The "
                             "index equals one when the ray is not deviated and exceeds one when the ray bends "
                             "towards the normal; as a function of the two angles on (0, pi/2) it grows with theta1 "
                             "and decreases with theta2."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.27.6", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['1', '/', '(', '1', '/', 'X_0', '+', 'X_1', '/', 'X_2', ')'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.001, 0.1, uses_negative=False),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.001, 0.1, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("d1", "distance of the object from the surface (outside the medium)"),
                                    "X_1": VariableInfo("n", "refractive index of the medium"),
                                    "X_2": VariableInfo("d2", "distance of the image from the surface (inside the medium)")},
                         target=VariableInfo("foc", "focal length of the refracting surface"),
                         description=(
                             "A single curved boundary between air and a transparent medium of refractive index n "
                             "images an object at distance d1 in front of it to an image at distance d2 inside the "
                             "medium. For positive distances the target is positive, increases with both distances, "
                             "and is smaller than d1 and smaller than d2/n."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_1v, None, dataset_name="SRSD I.29.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', f'{self.SPEED_OF_LIGHT}'], success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(1000000000.0, 100000000000.0, uses_negative=False)],
                         variables={"X_0": VariableInfo("omega", "angular frequency of the wave")},
                         target=VariableInfo("k", "wavenumber"),
                         description=(
                             "Wavenumber of an electromagnetic wave travelling in vacuum, given its angular "
                             "frequency. It is proportional to the frequency."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.29.16", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sqrt', '(', 'X_0', '^2', '+', 'X_1', '^2', '+', '2', '*', 'X_0', '*', 'X_1',
                                       '*', 'cos', '(', 'X_2', '-', 'X_3', ')', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("x1", "amplitude of the first oscillation"),
                                    "X_1": VariableInfo("x2", "amplitude of the second oscillation"),
                                    "X_2": VariableInfo("theta1", "phase of the first oscillation"),
                                    "X_3": VariableInfo("theta2", "phase of the second oscillation")},
                         target=VariableInfo("x", "amplitude of the combined oscillation"),
                         description=(
                             "Two oscillations of the same frequency with amplitudes x1, x2 and phases theta1, theta2"
                             " are superposed (equivalently, two plane vectors are added). The resulting amplitude is"
                             " non-negative, depends on the phases only through their difference (even and 2*pi-"
                             "periodic in it), equals x1 + x2 for oscillations in phase and |x1 - x2| in antiphase, "
                             "and is symmetric in the two oscillations."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.30.3", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'sin', '(', 'X_1', '*', 'X_2', '/', '2', ')', '^2', '/', 'sin', '(',
                                       'X_2', '/', '2', ')', '^2'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
                IntegerUniformSampling(10, 1000, uses_negative=False),
                UniformSampling(-6.283185307179586, 6.283185307179586, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("Int_0", "intensity produced by a single source"),
                                    "X_1": VariableInfo("n", "number of sources"),
                                    "X_2": VariableInfo("theta", "phase difference between neighbouring sources")},
                         target=VariableInfo("Int", "radiated intensity"),
                         description=(
                             "Intensity far from a row of n identical, equally spaced coherent sources, as a function"
                             " of the phase difference theta between neighbouring sources. It is non-negative, "
                             "proportional to Int_0, even and 2*pi-periodic in theta, reaches its maximum (n^2 times "
                             "the single-source intensity) at theta = 0 and has n - 1 zeros per period between the "
                             "maxima."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.30.5", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', 'X_1', '*', 'sin', '(', 'X_2', ')', ')'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-11, 1e-09, uses_negative=False),
                IntegerUniformSampling(1, 100, uses_negative=False),
                UniformSampling(0, 1.5707963267948966, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("lambd", "wavelength"),
                                    "X_1": VariableInfo("n", "number of sources"),
                                    "X_2": VariableInfo("theta", "angle of the first intensity zero from the forward direction")},
                         target=VariableInfo("d", "spacing between neighbouring sources"),
                         description=(
                             "Spacing of a row of n equally spaced coherent sources, inferred from the angle theta at"
                             " which their radiation first vanishes for a given wavelength. It is proportional to the"
                             " wavelength and decreases with the number of sources and with the angle (for 0 < theta "
                             "< pi/2)."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.32.5", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '^2', '*', 'X_1', '^2', '/', '(', '6', '*', 'pi', '*',
                                       f'{self.ELECTRIC_CONSTANT}', '*', f'{self.SPEED_OF_LIGHT}', '^3', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(0.001, 0.1),
                                   LogUniformSampling(100000.0, 10000000.0, uses_negative=False)],
                         variables={"X_0": VariableInfo("q", "electric charge of the particle"),
                                    "X_1": VariableInfo("a", "acceleration of the particle")},
                         target=VariableInfo("Pwr", "total radiated power"),
                         description=(
                             "Total power radiated by a point charge that accelerates while moving slowly compared "
                             "with light, in vacuum. It is non-negative, even in both the charge and the "
                             "acceleration, zero for an unaccelerated charge, and grows with both."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.32.17", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.ELECTRIC_CONSTANT}', '*', f'{self.SPEED_OF_LIGHT}', '*', 'X_0', '^2',
                                       '*', '8', '*', 'pi', '*', 'X_1', '^2', '/', '3', '*', 'X_2', '^4', '/', '(', '(',
                                       'X_2', '^2', '-', 'X_3', '^2', ')', '^2', ')', '/', '2'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(10.0, 1000.0),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
                LogUniformSampling(1000000000.0, 100000000000.0),
                LogUniformSampling(1000000000.0, 100000000000.0),
            ],
                         variables={"X_0": VariableInfo("Ef", "electric field amplitude of the incident wave"),
                                    "X_1": VariableInfo("r", "characteristic radius of the scatterer (sets its cross-section)"),
                                    "X_2": VariableInfo("omega", "angular frequency of the incident wave"),
                                    "X_3": VariableInfo("omega_0", "natural (resonance) angular frequency of the bound charge")},
                         target=VariableInfo("Pwr", "time-averaged scattered power"),
                         description=(
                             "Time-averaged power scattered by an elastically bound, undamped charge illuminated by a"
                             " plane electromagnetic wave. It is proportional to the incident intensity, diverges at "
                             "resonance (omega = omega_0), rises steeply with frequency far below resonance and "
                             "approaches a constant far above it."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.34.10", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', '1', '-', 'X_1', '/', f'{self.SPEED_OF_LIGHT}', ')'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1000000000.0, 100000000000.0, uses_negative=False),
                LogUniformSampling(100000.0, 10000000.0),
            ],
                         variables={"X_0": VariableInfo("omega_0", "angular frequency emitted by the source"),
                                    "X_1": VariableInfo("v", "speed of the source towards the observer")},
                         target=VariableInfo("omega", "observed angular frequency"),
                         description=(
                             "A source emitting at angular frequency omega_0 approaches a stationary observer with "
                             "speed v (source treated non-relativistically). The observed frequency equals omega_0 at"
                             " v = 0, increases with v, diverges as v approaches the wave speed, and is lowered for a"
                             " receding source (v < 0)."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.34.8", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '*', 'X_2', '/', 'X_3'], success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-11, 1e-09),
                LogUniformSampling(100000.0, 10000000.0),
                LogUniformSampling(10.0, 1000.0),
                LogUniformSampling(1000000000.0, 100000000000.0),
            ],
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
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.34.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['(', '1', '+', 'X_0', '/', f'{self.SPEED_OF_LIGHT}', ')', '*', 'X_1', '/',
                                       'sqrt', '(', '1', '-', 'X_0', '^2', '/', f'{self.SPEED_OF_LIGHT}', '^2', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1000000.0, 100000000.0),
                LogUniformSampling(1000000000.0, 100000000000.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("v", "speed of the source towards the observer"),
                                    "X_1": VariableInfo("omega_0", "angular frequency in the rest frame of the source")},
                         target=VariableInfo("omega", "observed angular frequency"),
                         description=(
                             "A light source emitting at angular frequency omega_0 in its own rest frame approaches "
                             "the observer head-on at speed v; special relativity is taken into account. The observed"
                             " frequency equals omega_0 at rest, increases monotonically with v and diverges as v "
                             "approaches the speed of light; reversing the direction of motion gives the reciprocal "
                             "frequency ratio."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_1v, None, dataset_name="SRSD I.34.27", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.DIRAC_CONSTANT}', '*', 'X_0'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(1000000000.0, 100000000000.0, uses_negative=False)],
                         variables={"X_0": VariableInfo("omega", "angular frequency of the radiation")},
                         target=VariableInfo("E_n", "energy of one quantum of radiation"),
                         description=(
                             "Energy carried by a single quantum (photon) of electromagnetic radiation of a given "
                             "angular frequency. It is positive and proportional to the frequency."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.37.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '+', 'X_1', '+', '2', '*', 'sqrt', '(', 'X_0', '*', 'X_1', ')', '*',
                                       'cos', '(', 'X_2', ')'],
                         success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.001, 0.1, uses_negative=False),
                LogUniformSampling(0.001, 0.1, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ],
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
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.38.12", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['4', '*', 'pi', '*', f'{self.ELECTRIC_CONSTANT}', '*', f'{self.DIRAC_CONSTANT}',
                                       '^2', '/', '(', 'X_0', '*', 'X_1', '^2', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-28, 1e-26, uses_negative=False),
                LogUniformSampling(1e-11, 1e-09, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("m", "mass of the orbiting particle"),
                                    "X_1": VariableInfo("q", "magnitude of the charge of the particle and of the nucleus")},
                         target=VariableInfo("r", "radius of the ground-state orbit"),
                         description=(
                             "Characteristic size of the lowest-energy state of a light charged particle bound to an "
                             "opposite, much heavier charge of equal magnitude (a hydrogen-like atom) in quantum "
                             "theory. It decreases with the particle mass and with the charge magnitude."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.39.10", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['1.5', '*', 'X_0', '*', 'X_1'], success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(10000.0, 1000000.0, uses_negative=False),
                LogUniformSampling(1e-05, 0.001, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("pr", "pressure of the gas"),
                                    "X_1": VariableInfo("V", "volume of the gas")},
                         target=VariableInfo("E_n", "internal energy of the gas"),
                         description=(
                             "Internal (thermal) energy of a dilute monatomic gas at pressure pr filling volume V. It"
                             " is proportional to both the pressure and the volume."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.39.11", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_1', '*', 'X_2', '/', '(', 'X_0', '-', '1', ')'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                UniformSampling(1, 2, uses_negative=False),
                LogUniformSampling(10000.0, 1000000.0, uses_negative=False),
                LogUniformSampling(1e-05, 0.001, uses_negative=False),
            ],
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
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.39.22", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.BOLTZMANN_CONSTANT}', '*', 'X_0', '*', 'X_1', '/', 'X_2'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
                LogUniformSampling(1e-05, 0.001, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("n", "number of molecules"),
                                    "X_1": VariableInfo("T", "absolute temperature"),
                                    "X_2": VariableInfo("V", "volume of the gas")},
                         target=VariableInfo("pr", "pressure of the gas"),
                         description=(
                             "Pressure of a dilute gas of n molecules at temperature T in a container of volume V. It"
                             " is proportional to the number of molecules and to the temperature, decreases with the "
                             "volume, and does not depend on the kind of molecule."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.40.1", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'exp', '(', 'u-', f'{self.GRAVITATIONAL_ACCELERATION}', '*', 'X_1',
                                       '*', 'X_2', '/', '(', f'{self.BOLTZMANN_CONSTANT}', '*', 'X_3', ')', ')'],
                         success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e25, 1e27, uses_negative=False),
                LogUniformSampling(1e-24, 1e-22, uses_negative=False),
                LogUniformSampling(0.01, 1.0),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("n_0", "number density at height zero"),
                                    "X_1": VariableInfo("m", "mass of one molecule"),
                                    "X_2": VariableInfo("x", "height"),
                                    "X_3": VariableInfo("T", "absolute temperature")},
                         target=VariableInfo("n", "number density of molecules at height x"),
                         description=(
                             "Number density of gas molecules at height x in an isothermal atmosphere in uniform "
                             "gravity, given the density at the ground. It equals n_0 at x = 0, is proportional to "
                             "n_0, decreases monotonically with height (faster for heavier molecules and lower "
                             "temperature) and tends to zero far up."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.41.16", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.DIRAC_CONSTANT}', '*', 'X_0', '^3', '/', '(', 'pi', '^2', '*',
                                       f'{self.SPEED_OF_LIGHT}', '^2', '*', '(', 'exp', '(', f'{self.DIRAC_CONSTANT}',
                                       '*', 'X_0', '/', '(', f'{self.BOLTZMANN_CONSTANT}', '*', 'X_1', ')', ')', '-',
                                       '1', ')', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("omega", "angular frequency"),
                                    "X_1": VariableInfo("T", "absolute temperature")},
                         target=VariableInfo("L_rad", "spectral intensity of the radiation per unit angular frequency"),
                         description=(
                             "Spectral distribution of the electromagnetic radiation in thermal equilibrium inside a "
                             "cavity whose walls are at temperature T. It is positive, increases with temperature at "
                             "every frequency, vanishes at zero and at very high frequency with a single maximum "
                             "whose position moves up in proportion to the temperature, and at low frequency becomes "
                             "linear in temperature."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.43.16", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '*', 'X_2', '/', 'X_3'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-06, 0.0001),
                LogUniformSampling(1e-11, 1e-09),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.001, 0.1, uses_negative=False),
            ],
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
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.43.31", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.BOLTZMANN_CONSTANT}', '*', 'X_0', '*', 'X_1'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(10000000000000.0, 1000000000000000.0, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("mob", "mobility (drift velocity per unit applied force)"),
                                    "X_1": VariableInfo("T", "absolute temperature")},
                         target=VariableInfo("D", "diffusion coefficient"),
                         description=(
                             "Diffusion coefficient of small particles suspended in a fluid at temperature T, given "
                             "their mobility. It is proportional to both the mobility and the temperature."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.43.43", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.BOLTZMANN_CONSTANT}', '*', 'X_1', '/', '(', '(', 'X_0', '-', '1', ')',
                                       '*', 'X_2', ')'], success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                UniformSampling(1, 2, uses_negative=False),
                LogUniformSampling(100.0, 10000.0, uses_negative=False),
                LogUniformSampling(1e-21, 1e-19, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("gamma", "adiabatic index (ratio of specific heats)"),
                                    "X_1": VariableInfo("v", "mean speed of the molecules"),
                                    "X_2": VariableInfo("A", "collision cross-section of the molecules")},
                         target=VariableInfo("kappa", "thermal conductivity of the gas"),
                         description=(
                             "Thermal conductivity of a dilute gas estimated from the motion and collisions of its "
                             "molecules. It increases with the mean molecular speed, decreases with the collision "
                             "cross-section, does not depend on the gas density, decreases with gamma and diverges as"
                             " gamma approaches one."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.44.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.BOLTZMANN_CONSTANT}', '*', 'X_0', '*', 'X_1', '*', 'ln', '(', 'X_2', '/',
                                       'X_3', ')'], success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e25, 1e27, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
                LogUniformSampling(1e-05, 0.001, uses_negative=False),
                LogUniformSampling(1e-05, 0.001, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("n", "number of molecules"),
                                    "X_1": VariableInfo("T", "absolute temperature"),
                                    "X_2": VariableInfo("V2", "final volume"),
                                    "X_3": VariableInfo("V1", "initial volume")},
                         target=VariableInfo("E_n", "work done by the expanding gas"),
                         description=(
                             "Work done by a dilute gas of n molecules expanding at constant temperature T from "
                             "volume V1 to V2. It is positive for expansion, zero for V1 = V2, changes sign when the "
                             "volumes are exchanged, depends on the volumes only through their ratio, and is "
                             "proportional to n and T."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD I.47.23", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sqrt', '(', 'X_0', '*', 'X_1', '/', 'X_2', ')'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                UniformSampling(1, 2, uses_negative=False),
                UniformSampling(5e-06, 1.5e-05, uses_negative=False),
                UniformSampling(1, 2, uses_negative=False),
            ],
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
        self.add_dataset(sl_2v, None, dataset_name="SRSD I.48.2", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', f'{self.SPEED_OF_LIGHT}', '^2', '/', 'sqrt', '(', '1', '-', 'X_1',
                                       '^2', '/', f'{self.SPEED_OF_LIGHT}', '^2', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-29, 1e-27, uses_negative=False),
                LogUniformSampling(1000000.0, 100000000.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("m", "rest mass of the body"),
                                    "X_1": VariableInfo("v", "speed of the body")},
                         target=VariableInfo("E_n", "total energy of the body"),
                         description=(
                             "Total energy (rest energy included) of a free body moving at speed v, in special "
                             "relativity. It is proportional to the mass, even in v, smallest for a body at rest, and"
                             " increases monotonically without bound as the speed approaches the speed of light."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD I.50.26", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', '(', 'cos', '(', 'X_1', '*', 'X_2', ')', '+', 'X_3', '*', 'cos', '(',
                                       'X_1', '*', 'X_2', ')', '^2', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(10.0, 1000.0),
                LogUniformSampling(0.001, 0.1, uses_negative=False),
                LogUniformSampling(0.001, 0.1),
            ],
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
        self.add_dataset(sl_5v, None, dataset_name="SRSD II.2.42", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', '(', 'X_1', '-', 'X_2', ')', '*', 'X_3', '/', 'X_4'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
                LogUniformSampling(0.0001, 0.01, uses_negative=False),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("kappa", "thermal conductivity of the slab"),
                                    "X_1": VariableInfo("T2", "temperature of the second face"),
                                    "X_2": VariableInfo("T1", "temperature of the first face"),
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
        self.add_dataset(sl_2v, None, dataset_name="SRSD II.3.24", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', '4', '*', 'pi', '*', 'X_1', '^2', ')'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(1.0, 100.0), LogUniformSampling(0.01, 1.0, uses_negative=False)],
                         variables={"X_0": VariableInfo("Pwr", "total power emitted by the source"),
                                    "X_1": VariableInfo("r", "distance from the source")},
                         target=VariableInfo("flux", "energy flux (intensity)"),
                         description=(
                             "Energy flux at distance r from a small source that radiates equally in all directions "
                             "into a non-absorbing medium. It is proportional to the emitted power and decreases "
                             "monotonically with distance."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD II.4.23", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', '4', '*', 'pi', '*', f'{self.ELECTRIC_CONSTANT}', '*', 'X_1',
                                       ')'], success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(0.001, 0.1), LogUniformSampling(0.01, 1.0, uses_negative=False)],
                         variables={"X_0": VariableInfo("q", "point charge"),
                                    "X_1": VariableInfo("r", "distance from the charge")},
                         target=VariableInfo("Volt", "electric potential"),
                         description=(
                             "Electric potential at distance r from a point charge at rest in vacuum, taken as zero "
                             "at infinity. It has the sign of the charge, is proportional to it, and its magnitude "
                             "decreases monotonically with distance, diverging at the charge."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD II.6.11", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'cos', '(', 'X_1', ')', '/', '(', '4', '*', 'pi', '*',
                                       f'{self.ELECTRIC_CONSTANT}', '*', 'X_2', '^2', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-22, 1e-20),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
                LogUniformSampling(1e-10, 1e-08, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("p_d", "electric dipole moment"),
                                    "X_1": VariableInfo("theta", "angle between the dipole axis and the direction to the observation point"),
                                    "X_2": VariableInfo("r", "distance from the dipole")},
                         target=VariableInfo("Volt", "electric potential"),
                         description=(
                             "Electric potential of a point electric dipole in vacuum, at distance r and angle theta "
                             "from the dipole axis. It is proportional to the dipole moment, positive on the side the"
                             " dipole points to, zero in the plane perpendicular to the axis (theta = pi/2), changes "
                             "sign under theta -> pi - theta, and falls off with distance faster than the potential "
                             "of a single charge."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="SRSD II.6.15a", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', '4', '*', 'pi', '*', f'{self.ELECTRIC_CONSTANT}', ')', '*', '3',
                                       '*', 'X_1', '/', '(', 'X_2', '^5', ')', '*', 'sqrt', '(', 'X_3', '^2', '+',
                                       'X_4', '^2', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-22, 1e-20),
                LogUniformSampling(1e-10, 1e-08),
                LogUniformSampling(1e-10, 1e-08, uses_negative=False),
                LogUniformSampling(1e-10, 1e-08),
                LogUniformSampling(1e-10, 1e-08),
            ],
                         variables={"X_0": VariableInfo("p_d", "electric dipole moment (along the z-axis)"),
                                    "X_1": VariableInfo("z", "z-coordinate of the observation point (along the dipole axis)"),
                                    "X_2": VariableInfo("r", "distance from the dipole"),
                                    "X_3": VariableInfo("x", "x-coordinate of the observation point"),
                                    "X_4": VariableInfo("y", "y-coordinate of the observation point")},
                         target=VariableInfo("Ef", "electric field component perpendicular to the dipole axis"),
                         description=(
                             "Electric field component perpendicular to the axis of a point dipole at the origin, at "
                             "the point (x, y, z) whose distance from the dipole is r. It is proportional to the "
                             "dipole moment, vanishes on the axis and in the plane z = 0, is odd in z, depends on x "
                             "and y only through the distance from the axis, and decays rapidly with r."
                         ),
                         data_source=data_source)

        self.add_dataset(sl_3v, None, dataset_name="SRSD II.6.15b", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', '4', '*', 'pi', '*', f'{self.ELECTRIC_CONSTANT}', ')', '*', '3',
                                       '*', 'cos', '(', 'X_1', ')', '*', 'sin', '(', 'X_1', ')', '/', '(', 'X_2', '^3',
                                       ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-22, 1e-20),
                UniformSampling(0, 3.141592653589793, uses_negative=False),
                LogUniformSampling(1e-10, 1e-08, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("p_d", "electric dipole moment"),
                                    "X_1": VariableInfo("theta", "angle between the dipole axis and the direction to the observation point"),
                                    "X_2": VariableInfo("r", "distance from the dipole")},
                         target=VariableInfo("Ef", "electric field component perpendicular to the dipole axis"),
                         description=(
                             "Electric field component perpendicular to the axis of a point dipole, at distance r and"
                             " angle theta from the axis. It is proportional to the dipole moment, zero on the axis "
                             "and in the plane perpendicular to it, largest in magnitude at 45 degrees, changes sign "
                             "under theta -> pi - theta, and decays rapidly with distance."
                         ),
                         data_source=data_source)

        self.add_dataset(sl_2v, None, dataset_name="SRSD II.8.7", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['3', '*', 'X_0', '^2', '/', '(', '5', '*', '4', '*', 'pi', '*',
                                       f'{self.ELECTRIC_CONSTANT}', '*', 'X_1', ')'],
                         success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[LogUniformSampling(1e-11, 1e-09),
                                                                              LogUniformSampling(1e-12, 1e-10,
                                                                                                 uses_negative=False)],
                         variables={"X_0": VariableInfo("q", "total charge"),
                                    "X_1": VariableInfo("d", "radius of the sphere")},
                         target=VariableInfo("E_n", "electrostatic energy of the charged sphere"),
                         description=(
                             "Electrostatic energy needed to assemble a charge spread uniformly through a ball in "
                             "vacuum. It is positive, even in the charge, grows with its magnitude and decreases with"
                             " the radius."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_1v, None, dataset_name="SRSD II.8.31", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.ELECTRIC_CONSTANT}', '*', 'X_0', '^2', '/', '2'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(10.0, 1000.0, uses_negative=False)],
                         variables={"X_0": VariableInfo("Ef", "electric field strength")},
                         target=VariableInfo("E_den", "energy density of the electric field"),
                         description=(
                             "Energy stored per unit volume in an electric field in vacuum. It is non-negative, even "
                             "in the field and zero without field."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD II.10.9", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', f'{self.ELECTRIC_CONSTANT}', '*', '(', '1', '+', 'X_1', ')',
                                       ')'], success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(0.001, 0.1), LogUniformSampling(1.0, 100.0, uses_negative=False)],
                         variables={"X_0": VariableInfo("sigma_den", "free surface charge density on the plates"),
                                    "X_1": VariableInfo("chi", "electric susceptibility of the dielectric")},
                         target=VariableInfo("Ef", "electric field inside the dielectric"),
                         description=(
                             "Electric field inside a linear dielectric that fills the gap of a parallel-plate "
                             "capacitor whose plates carry a given free surface charge density. It is proportional to"
                             " the charge density, equals the empty-capacitor value at chi = 0 and decreases as the "
                             "susceptibility grows."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="SRSD II.11.3", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '/', '(', 'X_2', '*', '(', 'X_3', '^2', '-', 'X_4', '^2', ')',
                                       ')'], success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-11, 1e-09),
                LogUniformSampling(1e-09, 1e-07, uses_negative=False),
                LogUniformSampling(1e-28, 1e-26, uses_negative=False),
                LogUniformSampling(1000000000.0, 100000000000.0),
                LogUniformSampling(1000000000.0, 100000000000.0),
            ],
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
        self.add_dataset(sl_5v, None, dataset_name="SRSD II.11.17", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', '(', '1', '+', 'X_1', '*', 'X_2', '*', 'cos', '(', 'X_3', ')', '/',
                                       '(', f'{self.BOLTZMANN_CONSTANT}', '*', 'X_4', ')', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e27, 1e29, uses_negative=False),
                LogUniformSampling(1e-22, 1e-20),
                LogUniformSampling(10.0, 1000.0),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("n_0", "number density without applied field"),
                                    "X_1": VariableInfo("p_d", "permanent electric dipole moment of a molecule"),
                                    "X_2": VariableInfo("Ef", "applied electric field strength"),
                                    "X_3": VariableInfo("theta", "angle between the dipole moment and the field"),
                                    "X_4": VariableInfo("T", "absolute temperature")},
                         target=VariableInfo("n", "number density of molecules with the given orientation"),
                         description=(
                             "Polar molecules in thermal equilibrium at temperature T in a weak electric field: the "
                             "target is the number per unit volume (and solid angle) whose dipoles point at angle "
                             "theta to the field, to first order in the field. It exceeds n_0 for dipoles along the "
                             "field and falls below it for the opposite orientation; the anisotropy grows with the "
                             "field and the dipole moment and decreases with temperature."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD II.11.20", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '^2', '*', 'X_2', '/', '(', '3', '*',
                                       f'{self.BOLTZMANN_CONSTANT}', '*', 'X_3', ')'],
                         success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(1e-22, 1e-20),
                LogUniformSampling(10.0, 1000.0),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("n_rho", "number density of molecules"),
                                    "X_1": VariableInfo("p_d", "permanent electric dipole moment of a molecule"),
                                    "X_2": VariableInfo("Ef", "applied electric field strength"),
                                    "X_3": VariableInfo("T", "absolute temperature")},
                         target=VariableInfo("Pol", "electric polarization"),
                         description=(
                             "Orientational polarization of a dilute gas of polar molecules in a weak electric field "
                             "at temperature T. It is proportional to the density and to the field, even in the "
                             "dipole moment (growing with its magnitude) and decreases with temperature."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD II.11.27", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.ELECTRIC_CONSTANT}', '*', 'X_0', '*', 'X_1', '*', 'X_2', '/', '(', '1',
                                       '-', 'X_0', '*', 'X_1', '/', '3', ')'],
                         success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(1e-33, 1e-31, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("n", "number density of molecules"),
                                    "X_1": VariableInfo("alpha", "molecular polarizability"),
                                    "X_2": VariableInfo("Ef", "macroscopic electric field strength")},
                         target=VariableInfo("Pol", "electric polarization"),
                         description=(
                             "Polarization of a dense isotropic dielectric, accounting for the field of the "
                             "neighbouring polarized molecules. It is proportional to the field, depends on density "
                             "and polarizability only through their product, grows faster than linearly with that "
                             "product and diverges at a finite value of it."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD II.11.28", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['1', '+', 'X_0', '*', 'X_1', '/', '(', '1', '-', 'X_0', '*', 'X_1', '/', '3',
                                       ')'], success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(1e-33, 1e-31, uses_negative=False),
            ],
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
        self.add_dataset(sl_2v, None, dataset_name="SRSD II.13.17", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['2', '*', 'X_0', '/', '(', '4', '*', 'pi', '*', f'{self.ELECTRIC_CONSTANT}', '*',
                                       f'{self.SPEED_OF_LIGHT}', '^2', '*', 'X_1', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(0.001, 0.1), LogUniformSampling(0.001, 0.1, uses_negative=False)],
                         variables={"X_0": VariableInfo("I", "electric current in the wire"),
                                    "X_1": VariableInfo("r", "distance from the wire")},
                         target=VariableInfo("B", "magnetic field strength"),
                         description=(
                             "Magnetic field at distance r from a long straight wire carrying a steady current in "
                             "vacuum. It is proportional to the current, decreases monotonically with distance and is"
                             " the same at all points of a circle around the wire."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD II.13.23", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', 'sqrt', '(', '1', '-', 'X_1', '^2', '/', f'{self.SPEED_OF_LIGHT}',
                                       '^2', ')'], success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e27, 1e29, uses_negative=False),
                LogUniformSampling(1000000.0, 100000000.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("rho_c_0", "charge density in the rest frame of the charges"),
                                    "X_1": VariableInfo("v", "speed of the charges")},
                         target=VariableInfo("rho_c", "charge density in the frame where the charges move"),
                         description=(
                             "Charge density observed in a frame in which a charge distribution moves at speed v, "
                             "given its density in its own rest frame (special relativity). It equals the rest-frame "
                             "density at v = 0, is proportional to it, even in v, increases with speed and diverges "
                             "as v approaches the speed of light."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD II.13.34", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '/', 'sqrt', '(', '1', '-', 'X_1', '^2', '/',
                                       f'{self.SPEED_OF_LIGHT}', '^2', ')'],
                         success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e27, 1e29, uses_negative=False),
                LogUniformSampling(1000000.0, 100000000.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("rho_c_0", "charge density in the rest frame of the charges"),
                                    "X_1": VariableInfo("v", "velocity of the charges")},
                         target=VariableInfo("j", "electric current density"),
                         description=(
                             "Current density carried by a charge distribution moving at velocity v, given its "
                             "density in its own rest frame (special relativity). It is proportional to the rest-"
                             "frame density, odd in v, close to the classical value for slow motion, and grows "
                             "without bound as the speed approaches the speed of light."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD II.15.4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['u-', 'X_0', '*', 'X_1', '*', 'cos', '(', 'X_2', ')'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-25, 1e-23),
                LogUniformSampling(0.001, 0.1),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ],
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
        self.add_dataset(sl_3v, None, dataset_name="SRSD II.15.5", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['u-', 'X_0', '*', 'X_1', '*', 'cos', '(', 'X_2', ')'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-22, 1e-20),
                LogUniformSampling(10.0, 1000.0),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ],
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
        self.add_dataset(sl_3v, None, dataset_name="SRSD II.21.32", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', '4', '*', 'pi', '*', f'{self.ELECTRIC_CONSTANT}', '*', 'X_1',
                                       '*', '(', '1', '-', 'X_2', '/', f'{self.SPEED_OF_LIGHT}', ')', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.001, 0.1),
                LogUniformSampling(1.0, 100.0, uses_negative=False),
                LogUniformSampling(1000000.0, 100000000.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("q", "point charge"),
                                    "X_1": VariableInfo("r", "distance from the charge (at its retarded position)"),
                                    "X_2": VariableInfo("v", "velocity component of the charge towards the observer")},
                         target=VariableInfo("Volt", "electric potential of the moving charge"),
                         description=(
                             "Electric potential at an observation point produced by a point charge in uniform "
                             "motion, with the distance taken to the position the charge had when the signal left it."
                             " It reduces to the static value at v = 0, is enhanced for an approaching charge and "
                             "reduced for a receding one, and diverges as the approach speed reaches the speed of "
                             "light."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD II.24.17", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sqrt', '(', 'X_0', '^2', '/', f'{self.SPEED_OF_LIGHT}', '^2', '-', 'pi', '^2',
                                       '/', 'X_1', '^2', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(1000000000.0, 100000000000.0),
                                   LogUniformSampling(0.001, 0.1, uses_negative=False)],
                         variables={"X_0": VariableInfo("omega", "angular frequency of the wave"),
                                    "X_1": VariableInfo("d", "transverse width of the guide")},
                         target=VariableInfo("k", "propagation wavenumber along the guide"),
                         description=(
                             "Wavenumber along a rectangular metal waveguide for its lowest mode at angular frequency"
                             " omega. It is real only above a cutoff frequency set by the width (below it the wave "
                             "does not propagate), zero at cutoff, increases with frequency and with the width, and "
                             "approaches the free-space wavenumber at high frequency."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_1v, None, dataset_name="SRSD II.27.16", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.ELECTRIC_CONSTANT}', '*', f'{self.SPEED_OF_LIGHT}', '*', 'X_0', '^2'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[LogUniformSampling(0.1, 10.0, uses_negative=False)],
                         variables={"X_0": VariableInfo("Ef", "electric field strength of the wave")},
                         target=VariableInfo("flux", "energy flux of the wave"),
                         description=(
                             "Energy flux (power per unit area) carried by a plane electromagnetic wave in vacuum. It"
                             " is non-negative, even in the field strength and zero without field."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_1v, None, dataset_name="SRSD II.27.18", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.ELECTRIC_CONSTANT}', '*', 'X_0', '^2'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[LogUniformSampling(0.1, 10.0, uses_negative=False)],
                         variables={"X_0": VariableInfo("Ef", "electric field strength of the wave")},
                         target=VariableInfo("E_den", "electromagnetic energy density of the wave"),
                         description=(
                             "Total electromagnetic energy per unit volume (electric and magnetic parts together) of "
                             "a plane wave in vacuum. It is non-negative, even in the field strength and zero without"
                             " field."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD II.34.2a", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '/', '(', '2', '*', 'pi', '*', 'X_2', ')'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-11, 1e-09),
                LogUniformSampling(100000.0, 10000000.0),
                LogUniformSampling(1e-11, 1e-09, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("q", "charge of the particle"),
                                    "X_1": VariableInfo("v", "orbital speed"),
                                    "X_2": VariableInfo("r", "radius of the orbit")},
                         target=VariableInfo("I", "electric current"),
                         description=(
                             "Average electric current produced by a point charge circulating on a circular orbit. It"
                             " is proportional to the charge and to the speed and decreases with the radius."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD II.34.2", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '*', 'X_2', '/', '2'], success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-11, 1e-09),
                LogUniformSampling(100000.0, 10000000.0),
                LogUniformSampling(1e-11, 1e-09, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("q", "charge of the particle"),
                                    "X_1": VariableInfo("v", "orbital speed"),
                                    "X_2": VariableInfo("r", "radius of the orbit")},
                         target=VariableInfo("mom", "magnetic moment of the orbit"),
                         description=(
                             "Magnetic moment produced by a point charge moving on a circular orbit. It is "
                             "proportional to the charge, the speed and the radius, and its sign follows the charge."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD II.34.11", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '*', 'X_2', '/', '(', '2', '*', 'X_3', ')'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                UniformSampling(-1.0, 1.0, uses_negative=False),
                LogUniformSampling(1e-11, 1e-09),
                LogUniformSampling(1e-09, 1e-07),
                LogUniformSampling(1e-30, 1e-28, uses_negative=False),
            ],
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
        self.add_dataset(sl_2v, None, dataset_name="SRSD II.34.29a", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', f'{self.PLANCK_CONSTANT}', '/', '(', '4', '*', 'pi', '*', 'X_1',
                                       ')'], success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[LogUniformSampling(1e-11, 1e-09),
                                                                              LogUniformSampling(1e-30, 1e-28,
                                                                                                 uses_negative=False)],
                         variables={"X_0": VariableInfo("q", "charge of the particle"),
                                    "X_1": VariableInfo("m", "mass of the particle")},
                         target=VariableInfo("mom", "unit of magnetic moment"),
                         description=(
                             "Natural unit of magnetic moment of a charged particle in quantum theory: the orbital "
                             "magnetic moment that goes with the smallest nonzero unit of angular momentum. It is "
                             "proportional to the charge and decreases with the mass."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD II.34.29b", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', f'{self.BOHR_MAGNETON}', '*', 'X_1', '*', 'X_2', '*', '2', '*', 'pi',
                                       '/', f'{self.PLANCK_CONSTANT}'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                UniformSampling(-1.0, 1.0, uses_negative=False),
                LogUniformSampling(0.001, 0.1),
                LogUniformSampling(1e-26, 1e-22),
            ],
                         variables={"X_0": VariableInfo("g_", "g-factor (Lande factor) of the state"),
                                    "X_1": VariableInfo("B", "magnetic field strength"),
                                    "X_2": VariableInfo("Jz", "component of the angular momentum along the field")},
                         target=VariableInfo("E_n", "magnetic energy of the state"),
                         description=(
                             "Energy shift of an atomic state in a weak uniform magnetic field, given its angular "
                             "momentum component along the field. It is proportional to the g-factor, the magnetic-"
                             "moment unit, the field and the angular momentum component, changes sign with that "
                             "component and vanishes without field."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD II.35.18", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', 'exp', '(', 'X_1', '*', 'X_2', '/', '(',
                                       f'{self.BOLTZMANN_CONSTANT}', '*', 'X_3', ')', ')', '+', 'exp', '(', 'u-', 'X_1',
                                       '*', 'X_2', '/', '(', f'{self.BOLTZMANN_CONSTANT}', '*', 'X_3', ')', ')', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(1e-25, 1e-23, uses_negative=False),
                LogUniformSampling(0.001, 0.1, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("n_0", "total number density of atoms"),
                                    "X_1": VariableInfo("mom", "magnetic moment of an atom"),
                                    "X_2": VariableInfo("B", "magnetic field strength"),
                                    "X_3": VariableInfo("T", "absolute temperature")},
                         target=VariableInfo("n", "density per thermal weight (total density divided by the summed weights of the two states)"),
                         description=(
                             "Atoms with magnetic moment mom that can only point along or against a magnetic field, "
                             "in thermal equilibrium at temperature T. The target is the total density divided by the"
                             " sum of the thermal weights of the two orientations; it equals n_0/2 without field, is "
                             "even in the field, and decreases monotonically to zero as the magnetic energy grows "
                             "relative to the thermal energy."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD II.35.21", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '*', 'tanh', '(', 'X_1', '*', 'X_2', '/', '(',
                                       f'{self.BOLTZMANN_CONSTANT}', '*', 'X_3', ')', ')'],
                         success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(1e-25, 1e-23, uses_negative=False),
                LogUniformSampling(0.001, 0.1, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("n_rho", "number density of atoms"),
                                    "X_1": VariableInfo("mom", "magnetic moment of an atom"),
                                    "X_2": VariableInfo("B", "magnetic field strength"),
                                    "X_3": VariableInfo("T", "absolute temperature")},
                         target=VariableInfo("M", "magnetization"),
                         description=(
                             "Magnetization of a dilute system of atoms whose moments can only point along or against"
                             " an applied field, at temperature T. It is odd in the field, linear for weak fields or "
                             "high temperature, and saturates at full alignment for strong fields or low temperature;"
                             " it increases with the field and decreases with temperature."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="SRSD II.36.38", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '/', '(', f'{self.BOLTZMANN_CONSTANT}', '*', 'X_2', ')', '+',
                                       'X_0', '*', 'X_3', '/', '(', f'{self.ELECTRIC_CONSTANT}', '*',
                                       f'{self.SPEED_OF_LIGHT}', '^2', '*', f'{self.BOLTZMANN_CONSTANT}', '*', 'X_2',
                                       ')', '*', 'X_4'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-25, 1e-23),
                LogUniformSampling(0.001, 0.1),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
                UniformSampling(0, 1, uses_negative=False),
                LogUniformSampling(1e23, 1e25, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("mom", "magnetic moment of an atom"),
                                    "X_1": VariableInfo("H", "applied magnetic field"),
                                    "X_2": VariableInfo("T", "absolute temperature"),
                                    "X_3": VariableInfo("alpha", "coupling constant of the internal field to the magnetization"),
                                    "X_4": VariableInfo("M", "magnetization of the material")},
                         target=VariableInfo("f", "ratio of the magnetic energy of an atom to the thermal energy"),
                         description=(
                             "In a magnetized material the field acting on an atom is the applied field plus an "
                             "internal contribution proportional to the magnetization; the target compares the "
                             "resulting magnetic energy of an atomic moment with the thermal energy at temperature T."
                             " It is linear in the applied field and in the magnetization, proportional to the moment"
                             " and decreases with temperature."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD II.37.1", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', '(', '1', '+', 'X_1', ')', '*', 'X_2'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(1e-25, 1e-23), LogUniformSampling(10000.0, 1000000.0),
                                   LogUniformSampling(0.001, 0.1)],
                         variables={"X_0": VariableInfo("mom", "magnetic moment"),
                                    "X_1": VariableInfo("chi", "magnetic susceptibility of the medium"),
                                    "X_2": VariableInfo("B", "applied magnetic field strength")},
                         target=VariableInfo("E_n", "magnetic energy of the moment"),
                         description=(
                             "Magnetic energy (magnitude) of a moment aligned with the field inside a linear magnetic"
                             " medium. The medium enhances the applied field: the energy is proportional to the "
                             "moment and the field, grows linearly with the susceptibility, and equals the vacuum "
                             "value at chi = 0."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD II.38.3", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '*', 'X_2', '/', 'X_3'], success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(0.0001, 0.01, uses_negative=False),
                LogUniformSampling(0.001, 0.1),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("Y", "Young's modulus of the material"),
                                    "X_1": VariableInfo("A", "cross-sectional area of the rod"),
                                    "X_2": VariableInfo("x", "extension of the rod"),
                                    "X_3": VariableInfo("d", "length of the rod")},
                         target=VariableInfo("F", "stretching force"),
                         description=(
                             "Force needed to stretch an elastic rod by a small extension x. It is proportional to "
                             "the modulus, the cross-sectional area and the extension, and decreases with the length "
                             "of the rod."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD II.38.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', '2', '*', '(', '1', '+', 'X_1', ')', ')'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[LogUniformSampling(0.1, 10.0, uses_negative=False),
                                                                   LogUniformSampling(0.01, 1.0, uses_negative=False)],
                         variables={"X_0": VariableInfo("Y", "Young's modulus of the material"),
                                    "X_1": VariableInfo("sigma", "Poisson's ratio of the material")},
                         target=VariableInfo("mu_S", "shear modulus"),
                         description=(
                             "Shear modulus of an isotropic elastic material, given its Young's modulus and Poisson's"
                             " ratio. It is proportional to Young's modulus, decreases with Poisson's ratio, and is "
                             "half of Young's modulus when Poisson's ratio is zero."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD III.4.32", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['1', '/', '(', 'exp', '(', '(', f'{self.PLANCK_CONSTANT}', '/', '(', '2', '*',
                                       'pi', ')', ')', '*', 'X_0', '/', '(', f'{self.BOLTZMANN_CONSTANT}', '*', 'X_1',
                                       ')', ')', '-', '1', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("omega", "angular frequency of the mode"),
                                    "X_1": VariableInfo("T", "absolute temperature")},
                         target=VariableInfo("n", "mean number of quanta in the mode"),
                         description=(
                             "Mean number of quanta in one radiation (oscillator) mode in thermal equilibrium at "
                             "temperature T. It is positive, decreases with frequency, increases with temperature, "
                             "depends on them only through their ratio, is large at low frequency (approximately "
                             "thermal energy over quantum energy) and exponentially small at high frequency."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD III.4.33", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.DIRAC_CONSTANT}', '*', 'X_0', '/', '(', 'exp', '(',
                                       f'{self.DIRAC_CONSTANT}', '*', 'X_0', '/', '(', f'{self.BOLTZMANN_CONSTANT}',
                                       '*', 'X_1', ')', ')', '-', '1', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("omega", "angular frequency of the oscillator"),
                                    "X_1": VariableInfo("T", "absolute temperature")},
                         target=VariableInfo("E_n", "mean thermal energy of the oscillator"),
                         description=(
                             "Mean thermal energy (zero-point energy excluded) of a quantum oscillator in equilibrium"
                             " at temperature T. When the quantum energy is small compared with the thermal energy it"
                             " approaches the classical value, which is proportional to the temperature and "
                             "independent of frequency; it is exponentially suppressed in the opposite limit, "
                             "increases with temperature and decreases with frequency."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD III.7.38", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['2', '*', 'X_0', '*', 'X_1', '/', '(', f'{self.PLANCK_CONSTANT}', '/', '(', '2',
                                       '*', 'pi', ')', ')'], success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[LogUniformSampling(1e-11, 1e-09), LogUniformSampling(0.001, 0.1)],
                         variables={"X_0": VariableInfo("mom", "magnetic moment of the particle"),
                                    "X_1": VariableInfo("B", "magnetic field strength")},
                         target=VariableInfo("omega", "angular frequency of the spin flip"),
                         description=(
                             "Angular frequency at which a spin-1/2 particle with magnetic moment mom in a uniform "
                             "magnetic field oscillates between (or precesses across) its two spin states. It is "
                             "proportional to the moment and to the field."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD III.8.54", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sin', '(', 'X_0', '*', 'X_1', '/', '(', f'{self.PLANCK_CONSTANT}', '/', '(',
                                       '2', '*', 'pi', ')', ')', ')', '^2'],
                         success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[LogUniformSampling(1e-18, 1e-16),
                                                                              LogUniformSampling(1e-18, 1e-16,
                                                                                                 uses_negative=False)],
                         variables={"X_0": VariableInfo("E_n", "coupling energy between the two states"),
                                    "X_1": VariableInfo("t", "time")},
                         target=VariableInfo("prob", "transition probability"),
                         description=(
                             "Probability that a two-state system with equal unperturbed energies, starting in one "
                             "state, is found in the other after time t when the states are coupled. It oscillates "
                             "periodically between 0 and 1, starts at zero at t = 0, and oscillates faster for "
                             "stronger coupling."
                         ),
                         data_source=data_source)

        self.add_dataset(sl_5v, None, dataset_name="SRSD III.9.52", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '^2', '*', 'X_1', '^2', '*', 'X_2', '^2', '/', '(',
                                       f'{self.PLANCK_CONSTANT}', '/', '(', '2', '*', 'pi', ')', ')', '^2', '*', 'sin',
                                       '(', '(', 'X_3', '-', 'X_4', ')', '*', 'X_2', '/', '2', ')', '^2', '/', '(', '(',
                                       'X_3', '-', 'X_4', ')', '*', 'X_2', '/', '2', ')', '^2'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-22, 1e-20),
                LogUniformSampling(10.0, 1000.0),
                LogUniformSampling(1e-18, 1e-16, uses_negative=False),
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("p_d", "electric dipole moment of the transition"),
                                    "X_1": VariableInfo("Ef", "amplitude of the oscillating electric field"),
                                    "X_2": VariableInfo("t", "duration of the interaction"),
                                    "X_3": VariableInfo("omega", "angular frequency of the driving field"),
                                    "X_4": VariableInfo("omega_0", "resonance angular frequency of the two-state system")},
                         target=VariableInfo("prob", "transition probability"),
                         description=(
                             "Probability that a weakly driven two-state system (resonance omega_0) makes a "
                             "transition while exposed to an oscillating electric field for a time t. It is non-"
                             "negative, peaks sharply at resonance, is even in the detuning omega - omega_0, vanishes"
                             " when the detuning times t is a nonzero multiple of 2*pi, and its peak grows with the "
                             "dipole moment, the field and the interaction time."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD III.10.19", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'sqrt', '(', 'X_1', '^2', '+', 'X_2', '^2', '+', 'X_3', '^2', ')'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-25, 1e-23),
                LogUniformSampling(0.001, 0.1),
                LogUniformSampling(0.001, 0.1),
                LogUniformSampling(0.001, 0.1),
            ],
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
        self.add_dataset(sl_1v, None, dataset_name="SRSD III.12.43", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', f'{self.DIRAC_CONSTANT}'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[IntegerUniformSampling(1, 100, uses_negative=False)],
                         variables={"X_0": VariableInfo("n", "angular momentum quantum number")},
                         target=VariableInfo("L", "angular momentum"),
                         description=(
                             "Allowed values of a component of the angular momentum of a quantum system, labelled by "
                             "a quantum number n. The values are proportional to n and evenly spaced."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD III.13.18", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['2', '*', 'X_0', '*', 'X_1', '^2', '*', 'X_2', '/', '(',
                                       f'{self.PLANCK_CONSTANT}', '/', '(', '2', '*', 'pi', ')', ')'],
                         success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-18, 1e-16),
                LogUniformSampling(1e-10, 1e-08, uses_negative=False),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("E_n", "coupling energy between neighbouring atoms"),
                                    "X_1": VariableInfo("d", "lattice spacing"),
                                    "X_2": VariableInfo("k", "wavenumber of the electron")},
                         target=VariableInfo("v", "velocity of the electron wave packet"),
                         description=(
                             "Velocity of an electron wave packet with small wavenumber moving along a one-"
                             "dimensional crystal lattice. It is zero at k = 0, proportional to k and to the coupling"
                             " energy, and increases with the lattice spacing."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD III.14.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', '(', 'exp', '(', 'X_1', '*', 'X_2', '/', '(',
                                       f'{self.BOLTZMANN_CONSTANT}', '*', 'X_3', ')', ')', '-', '1', ')'],
                         success_threshold=1e-7,
                         seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.001, 0.1),
                LogUniformSampling(1e-22, 1e-20, uses_negative=False),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(10.0, 1000.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("I_0", "reverse saturation current"),
                                    "X_1": VariableInfo("q", "charge of the carriers"),
                                    "X_2": VariableInfo("Volt", "applied voltage (forward bias positive)"),
                                    "X_3": VariableInfo("T", "absolute temperature")},
                         target=VariableInfo("I", "electric current through the junction"),
                         description=(
                             "Current through a semiconductor p-n junction at temperature T under an applied voltage."
                             " It is zero at zero voltage, grows exponentially for forward bias and saturates at -I_0"
                             " for reverse bias (strongly rectifying); it is proportional to I_0."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD III.15.12", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['2', '*', 'X_0', '*', '(', '1', '-', 'cos', '(', 'X_1', '*', 'X_2', ')', ')'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-18, 1e-16, uses_negative=False),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
                LogUniformSampling(1e-10, 1e-08, uses_negative=False),
            ],
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
        self.add_dataset(sl_2v, None, dataset_name="SRSD III.15.14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.DIRAC_CONSTANT}', '^2', '/', '(', '2', '*', 'X_0', '*', 'X_1', '^2',
                                       ')'], success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-18, 1e-16, uses_negative=False),
                LogUniformSampling(1e-10, 1e-08, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("E_n", "coupling energy between neighbouring atoms"),
                                    "X_1": VariableInfo("d", "lattice spacing")},
                         target=VariableInfo("m", "effective mass of the electron"),
                         description=(
                             "Effective mass of an electron near the bottom of an energy band of a one-dimensional "
                             "crystal lattice. It is positive and decreases with both the coupling energy and the "
                             "lattice spacing."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD III.15.27", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['2', '*', 'pi', '*', 'X_0', '/', '(', 'X_1', '*', 'X_2', ')'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                IntegerUniformSampling(1, 100),
                IntegerUniformSampling(1, 100, uses_negative=False),
                LogUniformSampling(1e-10, 1e-08, uses_negative=False),
            ],
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
        self.add_dataset(sl_3v, None, dataset_name="SRSD III.17.37", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', '(', '1', '+', 'X_1', '*', 'cos', '(', 'X_2', ')', ')'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-18, 1e-16, uses_negative=False),
                LogUniformSampling(1e-18, 1e-16),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ],
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
        self.add_dataset(sl_3v, None, dataset_name="SRSD III.19.51", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['u-', 'X_0', '*', 'X_1', '^4', '/', '(', '2', '*', '(', '4', '*', 'pi', '*',
                                       f'{self.ELECTRIC_CONSTANT}', ')', '^2', '*', '(', f'{self.PLANCK_CONSTANT}', '/',
                                       '(', '2', '*', 'pi', ')', ')', '^2', ')', '*', '(', '1', '/', 'X_2', '^2', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-30, 1e-28, uses_negative=False),
                LogUniformSampling(1e-11, 1e-09),
                IntegerUniformSampling(1, 100, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("m", "mass of the orbiting particle"),
                                    "X_1": VariableInfo("q", "magnitude of the charge of the particle and of the nucleus"),
                                    "X_2": VariableInfo("n", "principal quantum number")},
                         target=VariableInfo("E_n", "energy of the bound state"),
                         description=(
                             "Energy levels of a light charged particle bound to an opposite, much heavier charge of "
                             "equal magnitude (a hydrogen-like atom), labelled by the principal quantum number n. The"
                             " energies are negative (bound), rise towards zero as n grows, are proportional to the "
                             "mass, and become deeper for larger charge."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD III.21.20", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['u-', 'X_0', '*', 'X_1', '*', 'X_2', '/', 'X_3'],
                         success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-29, 1e-27, uses_positive=False),
                LogUniformSampling(1e-11, 1e-09, uses_positive=False),
                LogUniformSampling(0.001, 0.1),
                LogUniformSampling(1e-30, 1e-28, uses_negative=False),
            ],
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
        self.add_dataset(sl_4v, None, dataset_name="SRSD Bonus 1", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['(', 'X_0', '*', 'X_1', '*', f'{self.FINE_STRUCTURE_CONSTANT}', '*',
                                       f'{self.DIRAC_CONSTANT}', '*', f'{self.SPEED_OF_LIGHT}', '/', '(', '4', '*',
                                       'X_2', '*', 'sin', '(', 'X_3', '/', '2', ')', '^2', ')', ')', '^2'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                IntegerUniformSampling(1, 10, uses_negative=False),
                IntegerUniformSampling(1, 10, uses_negative=False),
                LogUniformSampling(1e-18, 1e-16, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("Z_1", "charge number of the projectile"),
                                    "X_1": VariableInfo("Z_2", "charge number of the target nucleus"),
                                    "X_2": VariableInfo("E_n", "kinetic energy of the projectile"),
                                    "X_3": VariableInfo("theta", "scattering angle")},
                         target=VariableInfo("A", "differential scattering cross-section"),
                         description=(
                             "Non-relativistic charged projectile scattered by the electrostatic field of a fixed, "
                             "point-like target nucleus, as a function of the scattering angle. The cross-section is "
                             "positive, symmetric in the two charge numbers and independent of their signs, decreases"
                             " with the projectile energy, and is very strongly peaked forward (diverging as theta ->"
                             " 0) with its minimum at backscattering."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_6v, None, dataset_name="SRSD Bonus 2", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'X_1', '/', 'X_2', '^2', '*', '(', '1', '+', 'sqrt', '(', '1', '+',
                                       '2', '*', 'X_3', '*', 'X_2', '^2', '/', '(', 'X_0', '*', 'X_1', '^2', ')', ')',
                                       '*', 'cos', '(', 'X_4', '-', 'X_5', ')', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(1000000000.0, 100000000000.0, uses_negative=False),
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
                LogUniformSampling(1e25, 1e27, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("m", "mass of the orbiting body"),
                                    "X_1": VariableInfo("k_G", "strength constant of the attractive central force"),
                                    "X_2": VariableInfo("L", "angular momentum of the body"),
                                    "X_3": VariableInfo("E_n", "total energy of the body"),
                                    "X_4": VariableInfo("theta1", "polar angle of the body"),
                                    "X_5": VariableInfo("theta2", "polar angle of the point of closest approach")},
                         target=VariableInfo("k", "reciprocal of the radial distance"),
                         description=(
                             "A body moves under an attractive central force that weakens with distance in the same "
                             "way as gravity; the target is the reciprocal of its distance from the centre as a "
                             "function of its polar angle. It depends on the angles only through their difference, is"
                             " 2*pi-periodic and even in it, and is largest at the point of closest approach (theta1 "
                             "= theta2)."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD Bonus 3", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', '(', '1', '-', 'X_1', '^2', ')', '/', '(', '1', '+', 'X_1', '*',
                                       'cos', '(', 'X_2', '-', 'X_3', ')', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
                UniformSampling(0.0, 1.0, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("d", "semi-major axis of the orbit"),
                                    "X_1": VariableInfo("alpha", "eccentricity of the orbit"),
                                    "X_2": VariableInfo("theta1", "polar angle of the body"),
                                    "X_3": VariableInfo("theta2", "polar angle of the point of closest approach")},
                         target=VariableInfo("r", "radial distance from the attracting centre"),
                         description=(
                             "Distance from the attracting centre of a body on a closed elliptical orbit, as a "
                             "function of its polar angle. It is 2*pi-periodic and even in the angle difference, "
                             "smallest at theta1 = theta2 and largest on the opposite side, proportional to the semi-"
                             "major axis, and constant (a circle) for zero eccentricity."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="SRSD Bonus 4", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sqrt', '(', '2', '/', 'X_0', '*', '(', 'X_1', '-', 'X_2', '-', 'X_3', '^2', '/',
                                       '(', '2', '*', 'X_0', '*', 'X_4', '^2', ')', ')', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(1e25, 1e27, uses_negative=False),
                LogUniformSampling(1e25, 1e27, uses_negative=False),
                LogUniformSampling(100000000.0, 10000000000.0),
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("m", "mass of the particle"),
                                    "X_1": VariableInfo("E_n", "total energy of the particle"),
                                    "X_2": VariableInfo("U", "potential energy at the current distance"),
                                    "X_3": VariableInfo("L", "angular momentum of the particle"),
                                    "X_4": VariableInfo("r", "current distance from the centre")},
                         target=VariableInfo("v", "radial velocity"),
                         description=(
                             "Radial speed of a particle moving in a central potential, at the moment its distance "
                             "from the centre is r. It is real only where the energy exceeds the potential plus the "
                             "rotational (centrifugal) energy, is zero at the turning points, increases with the "
                             "total energy and decreases with the potential energy and the angular momentum."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD Bonus 5", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['2', '*', 'pi', '*', 'sqrt', '(', 'X_0', '^3', ')', '/', 'sqrt', '(',
                                       f'{self.GRAVITATIONAL_CONSTANT}', '*', '(', 'X_1', '+', 'X_2', ')', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(1e23, 1e25, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("d", "semi-major axis of the relative orbit"),
                                    "X_1": VariableInfo("m1", "mass of the first body"),
                                    "X_2": VariableInfo("m2", "mass of the second body")},
                         target=VariableInfo("t", "orbital period"),
                         description=(
                             "Period of two bodies in a bound gravitational orbit around each other. It grows faster "
                             "than linearly with the size of the orbit, decreases with the masses, depends on them "
                             "only through their sum, and does not depend on the eccentricity."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_7v, None, dataset_name="SRSD Bonus 6", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sqrt', '(', '1', '+', '2', '*', 'X_0', '^2', '*', 'X_1', '*', 'X_2', '^2', '/',
                                       '(', 'X_3', '*', '(', 'X_4', '*', 'X_5', '*', 'X_6', '^2', ')', '^2', ')', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-18, 1e-16),
                LogUniformSampling(1e-18, 1e-16, uses_negative=False),
                LogUniformSampling(1e-10, 1e-08, uses_negative=False),
                LogUniformSampling(1e-30, 1e-28, uses_negative=False),
                IntegerUniformSampling(1, 10, uses_negative=False),
                IntegerUniformSampling(1, 10, uses_negative=False),
                LogUniformSampling(1e-11, 1e-09),
            ],
                         variables={"X_0": VariableInfo("epsilon", "permittivity"),
                                    "X_1": VariableInfo("E_n", "total energy of the particle"),
                                    "X_2": VariableInfo("L", "angular momentum of the particle"),
                                    "X_3": VariableInfo("m", "mass of the particle"),
                                    "X_4": VariableInfo("Z_1", "charge number of the particle"),
                                    "X_5": VariableInfo("Z_2", "charge number of the fixed centre"),
                                    "X_6": VariableInfo("q", "elementary charge")},
                         target=VariableInfo("alpha", "eccentricity of the orbit"),
                         description=(
                             "Shape of the orbit of a charged particle in the electrostatic field of a fixed charged "
                             "centre. The eccentricity is one for zero total energy, above one for unbound (positive-"
                             "energy) orbits and below one for bound ones; it is symmetric in the two charge numbers "
                             "and grows with the angular momentum when the energy is positive."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD Bonus 7", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sqrt', '(', '8', '*', 'pi', '*', f'{self.GRAVITATIONAL_CONSTANT}', '*', 'X_0',
                                       '/', '3', '-', 'X_1', '*', f'{self.SPEED_OF_LIGHT}', '^2', '/', 'X_2', '^2',
                                       ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-28, 1e-26, uses_negative=False),
                IntegerUniformSampling(-1, 2),
                LogUniformSampling(1e25, 1e27, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("rho", "mass density of the universe"),
                                    "X_1": VariableInfo("alpha", "spatial curvature parameter"),
                                    "X_2": VariableInfo("d", "scale factor of the universe")},
                         target=VariableInfo("H_G", "expansion rate of the universe"),
                         description=(
                             "Expansion rate of a homogeneous, isotropic universe filled with matter of density rho. "
                             "It increases with the density; positive spatial curvature lowers it and negative "
                             "curvature raises it, and for flat space it does not depend on the scale factor."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD Bonus 8", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', '1', '+', 'X_0', '/', '(', f'{self.ELECTRON_MASS}', '*',
                                       f'{self.SPEED_OF_LIGHT}', '^2', ')', '*', '(', '1', '-', 'cos', '(', 'X_1', ')',
                                       ')', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-24, 1e-22, uses_negative=False),
                UniformSampling(-3.141592653589793, 3.141592653589793),
            ],
                         variables={"X_0": VariableInfo("E_n", "energy of the incoming photon"),
                                    "X_1": VariableInfo("theta", "scattering angle")},
                         target=VariableInfo("K", "energy of the scattered photon"),
                         description=(
                             "Energy of a photon after it scatters by angle theta off a free electron initially at "
                             "rest. It equals the incoming energy at theta = 0, decreases monotonically up to "
                             "backscattering (theta = pi), never exceeds the incoming energy, is even and 2*pi-"
                             "periodic in theta, and is almost unchanged for low-energy photons."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD Bonus 9", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['u-', '32', '/', '5', '*', f'{self.GRAVITATIONAL_CONSTANT}', '^4', '/',
                                       f'{self.SPEED_OF_LIGHT}', '^5', '*', 'X_0', '^2', '*', 'X_1', '^2', '*', '(',
                                       'X_0', '+', 'X_1', ')', '/', 'X_2', '^5'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(1e23, 1e25, uses_negative=False),
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("m1", "mass of the first body"),
                                    "X_1": VariableInfo("m2", "mass of the second body"),
                                    "X_2": VariableInfo("r", "orbital separation")},
                         target=VariableInfo("Pwr", "rate of change of the orbital energy"),
                         description=(
                             "Two point masses on a circular orbit lose orbital energy by emitting gravitational "
                             "radiation; the target is the (negative) rate of change of the orbital energy. It is "
                             "symmetric in the masses, its magnitude grows rapidly with the masses and falls off very"
                             " steeply with the separation."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_2v, None, dataset_name="SRSD Bonus 10", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['(', 'cos', '(', 'X_0', ')', '-', 'X_1', '/', f'{self.SPEED_OF_LIGHT}', ')', '/',
                                       '(', '1', '-', 'X_1', '/', f'{self.SPEED_OF_LIGHT}', '*', 'cos', '(', 'X_0', ')',
                                       ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                UniformSampling(0, 3.141592653589793, uses_negative=False),
                LogUniformSampling(1000000.0, 100000000.0),
            ],
                         variables={"X_0": VariableInfo("theta2", "angle between the light ray and the direction of motion in the original frame"),
                                    "X_1": VariableInfo("v", "relative speed of the two frames")},
                         target=VariableInfo("cos_theta1", "cosine of the direction angle of the ray in the moving frame"),
                         description=(
                             "A light ray makes angle theta2 with the direction of relative motion in one inertial "
                             "frame; the target is the cosine of the corresponding angle seen from a frame moving "
                             "with speed v. It equals cos(theta2) at v = 0, stays in [-1, 1], is unchanged for rays "
                             "along the motion (theta2 = 0 or pi), and decreases monotonically with v."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD Bonus 11", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'sin', '(', 'X_1', '/', '2', ')', '^2', '*', 'sin', '(', 'X_2', '*',
                                       'X_3', '/', '2', ')', '^2', '/', '(', 'X_1', '^2', '/', '4', '*', 'sin', '(',
                                       'X_3', '/', '2', ')', '^2', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.001, 0.1, uses_negative=False),
                LogUniformSampling(1e-11, 1e-09, uses_negative=False),
                IntegerUniformSampling(1, 100, uses_negative=False),
                LogUniformSampling(1e-11, 1e-09, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("I_0", "intensity scale"),
                                    "X_1": VariableInfo("alpha", "phase difference across the width of one slit"),
                                    "X_2": VariableInfo("n", "number of slits"),
                                    "X_3": VariableInfo("delta", "phase difference between neighbouring slits")},
                         target=VariableInfo("I", "diffracted intensity"),
                         description=(
                             "Intensity of light diffracted by a grating of n identical, equally spaced slits. It is "
                             "non-negative, proportional to I_0, even in alpha and delta; its dependence on alpha "
                             "forms an envelope with its maximum at alpha = 0 and zeros at nonzero multiples of 2*pi,"
                             " while in delta it has sharp principal maxima at multiples of 2*pi with n - 1 zeros "
                             "between neighbouring ones."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="SRSD Bonus 12", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '/', '(', '4', '*', 'pi', '*', 'X_1', '*', 'X_2', '^2', ')', '*', '(',
                                       '4', '*', 'pi', '*', 'X_1', '*', 'X_3', '*', 'X_4', '-', 'X_0', '*', 'X_4', '*',
                                       'X_2', '^3', '/', '(', 'X_2', '^2', '-', 'X_4', '^2', ')', '^2', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(0.001, 0.1),
                LogUniformSampling(1e-12, 1e-10, uses_negative=False),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("q", "point charge"),
                                    "X_1": VariableInfo("epsilon", "vacuum permittivity (electric constant)"),
                                    "X_2": VariableInfo("y", "distance of the charge from the centre of the sphere"),
                                    "X_3": VariableInfo("Volt", "potential at which the sphere is held"),
                                    "X_4": VariableInfo("d", "radius of the conducting sphere")},
                         target=VariableInfo("F", "radial force on the point charge (positive means repulsion)"),
                         description=(
                             "Force on a point charge outside a conducting sphere held at a fixed potential, in "
                             "vacuum. The part due to the sphere's potential repels a charge of the same sign, while "
                             "the charges induced on the sphere always attract; the attraction dominates near the "
                             "surface, where the force diverges as y approaches d."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="SRSD Bonus 13", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_1', '/', '(', '4', '*', 'pi', '*', 'X_0', '*', 'sqrt', '(', 'X_2', '^2', '+',
                                       'X_3', '^2', '-', '2', '*', 'X_2', '*', 'X_3', '*', 'cos', '(', 'X_4', ')', ')',
                                       ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-12, 1e-10, uses_negative=False),
                LogUniformSampling(0.001, 0.1),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
                UniformSampling(0, 3.141592653589793, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("epsilon", "vacuum permittivity (electric constant)"),
                                    "X_1": VariableInfo("q", "point charge"),
                                    "X_2": VariableInfo("r", "distance of the observation point from the origin"),
                                    "X_3": VariableInfo("d", "distance of the charge from the origin"),
                                    "X_4": VariableInfo("alpha", "angle between the position vectors of the observation point and the charge")},
                         target=VariableInfo("Volt", "electric potential"),
                         description=(
                             "Electric potential in vacuum of a point charge located at distance d from the origin, "
                             "observed at distance r from the origin. It is proportional to the charge, symmetric in "
                             "r and d, even and 2*pi-periodic in alpha, largest in magnitude when the two directions "
                             "coincide, and diverges where the observation point meets the charge."
                         ),
                         data_source=data_source)

        self.add_dataset(sl_5v, None, dataset_name="SRSD Bonus 14", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['X_0', '*', 'cos', '(', 'X_1', ')', '*', '(', 'u-', 'X_2', '+', 'X_3', '^3', '/',
                                       'X_2', '^2', '*', '(', 'X_4', '-', '1', ')', '/', '(', 'X_4', '+', '2', ')',
                                       ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(10.0, 1000.0),
                UniformSampling(0, 3.141592653589793, uses_negative=False),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
                LogUniformSampling(0.01, 1.0, uses_negative=False),
                LogUniformSampling(0.1, 10.0, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("Ef", "strength of the applied uniform electric field"),
                                    "X_1": VariableInfo("theta", "polar angle from the field direction"),
                                    "X_2": VariableInfo("r", "distance from the centre of the sphere"),
                                    "X_3": VariableInfo("d", "radius of the sphere"),
                                    "X_4": VariableInfo("alpha", "relative permittivity (dielectric constant) of the sphere")},
                         target=VariableInfo("Volt", "electric potential outside the sphere"),
                         description=(
                             "Electric potential outside a sphere of linear dielectric placed in an initially uniform"
                             " electric field. It is proportional to the applied field, vanishes in the plane through"
                             " the centre perpendicular to the field and changes sign across it, and reduces to the "
                             "potential of the uniform field when alpha = 1 or far from the sphere."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD Bonus 15", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sqrt', '(', '1', '-', 'X_0', '^2', '/', f'{self.SPEED_OF_LIGHT}', '^2', ')',
                                       '*', 'X_1', '/', '(', '1', '+', 'X_0', '/', f'{self.SPEED_OF_LIGHT}', '*', 'cos',
                                       '(', 'X_2', ')', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(100000.0, 10000000.0, uses_negative=False),
                LogUniformSampling(1000000000.0, 100000000000.0, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("v", "relative speed of the two frames"),
                                    "X_1": VariableInfo("omega", "angular frequency measured in the first frame"),
                                    "X_2": VariableInfo("theta", "angle between the direction of the radiation and the relative velocity")},
                         target=VariableInfo("omega_0", "angular frequency in the second frame"),
                         description=(
                             "Angular frequency of the same electromagnetic wave measured in two inertial frames "
                             "moving with relative speed v. It is proportional to omega, equals omega at v = 0, and "
                             "at fixed speed is smallest for theta = 0 and largest for theta = pi."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_5v, None, dataset_name="SRSD Bonus 16", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['sqrt', '(', '(', 'X_0', '-', 'X_1', '*', 'X_2', ')', '^2', '*',
                                       f'{self.SPEED_OF_LIGHT}', '^2', '+', 'X_3', '^2', '*', f'{self.SPEED_OF_LIGHT}',
                                       '^4', ')', '+', 'X_1', '*', 'X_4'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-09, 1e-07),
                LogUniformSampling(1e-11, 1e-09),
                LogUniformSampling(10.0, 1000.0),
                LogUniformSampling(1e-30, 1e-28, uses_negative=False),
                LogUniformSampling(0.1, 10.0),
            ],
                         variables={"X_0": VariableInfo("p", "canonical momentum of the particle"),
                                    "X_1": VariableInfo("q", "charge of the particle"),
                                    "X_2": VariableInfo("A_vec", "magnetic vector potential"),
                                    "X_3": VariableInfo("m", "rest mass of the particle"),
                                    "X_4": VariableInfo("Volt", "electric scalar potential")},
                         target=VariableInfo("E_n", "total energy of the particle"),
                         description=(
                             "Total energy of a relativistic charged particle in an electromagnetic field described "
                             "by its scalar and vector potentials. It depends on the momentum and the vector "
                             "potential only through the kinetic momentum p - q*A_vec, is at least the rest energy "
                             "plus q*Volt, and is linear in the scalar potential."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_6v, None, dataset_name="SRSD Bonus 17", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['(', 'X_1', '^2', '+', 'X_0', '^2', '*', 'X_2', '^2', '*', 'X_3', '^2', '*', '(',
                                       '1', '+', 'X_4', '*', 'X_3', '/', 'X_5', ')', ')', '/', '(', '2', '*', 'X_0',
                                       ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1e-30, 1e-28, uses_negative=False),
                LogUniformSampling(1e-09, 1e-07),
                LogUniformSampling(1000000000.0, 100000000000.0),
                LogUniformSampling(1e-11, 1e-09),
                LogUniformSampling(0.1, 10.0),
                LogUniformSampling(1e-11, 1e-09, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("m", "mass of the particle"),
                                    "X_1": VariableInfo("p", "momentum of the particle"),
                                    "X_2": VariableInfo("omega", "angular frequency of small oscillations"),
                                    "X_3": VariableInfo("x", "position (displacement from equilibrium)"),
                                    "X_4": VariableInfo("alpha", "strength of the anharmonic correction"),
                                    "X_5": VariableInfo("y", "length scale of the anharmonic correction")},
                         target=VariableInfo("E_n", "energy of the oscillator"),
                         description=(
                             "Energy of a particle in a nearly harmonic potential with a small asymmetric "
                             "(anharmonic) correction. It is even in the momentum, reduces to the pure harmonic "
                             "oscillator for alpha = 0, and the correction breaks the symmetry between positive and "
                             "negative displacements."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD Bonus 18", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['3', '/', '(', '8', '*', 'pi', '*', f'{self.GRAVITATIONAL_CONSTANT}', ')', '*',
                                       '(', f'{self.SPEED_OF_LIGHT}', '^2', '*', 'X_0', '/', 'X_1', '^2', '+', 'X_2',
                                       '^2', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(10.0, 1000.0),
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
                LogUniformSampling(1.0, 100.0),
            ],
                         variables={"X_0": VariableInfo("k_f", "spatial curvature parameter"),
                                    "X_1": VariableInfo("r", "scale factor of the universe"),
                                    "X_2": VariableInfo("H_G", "expansion rate of the universe")},
                         target=VariableInfo("rho_0", "mass density of the universe"),
                         description=(
                             "Mass density of a homogeneous, isotropic universe given its expansion rate and spatial "
                             "curvature. It increases with the expansion rate (even in it); positive curvature raises"
                             " the density and negative curvature lowers it, and for flat space it does not depend on"
                             " the scale factor."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_4v, None, dataset_name="SRSD Bonus 19", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=['u-', '1', '/', '(', '8', '*', 'pi', '*', f'{self.GRAVITATIONAL_CONSTANT}', ')',
                                       '*', '(', f'{self.SPEED_OF_LIGHT}', '^4', '*', 'X_0', '/', 'X_1', '^2', '+',
                                       'X_2', '^2', '*', f'{self.SPEED_OF_LIGHT}', '^2', '*', '(', '1', '-', '2', '*',
                                       'X_3', ')', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(10.0, 1000.0),
                LogUniformSampling(100000000.0, 10000000000.0, uses_negative=False),
                LogUniformSampling(1.0, 100.0, uses_negative=False),
                UniformSampling(-10, 10),
            ],
                         variables={"X_0": VariableInfo("k_f", "spatial curvature parameter"),
                                    "X_1": VariableInfo("r", "scale factor of the universe"),
                                    "X_2": VariableInfo("H_G", "expansion rate of the universe"),
                                    "X_3": VariableInfo("alpha", "deceleration parameter")},
                         target=VariableInfo("pr", "pressure of the cosmic fluid"),
                         description=(
                             "Pressure of the matter filling a homogeneous, isotropic universe, given its expansion "
                             "rate, spatial curvature and deceleration parameter. It is even in the expansion rate, "
                             "increases with the deceleration parameter, and for flat space vanishes when the "
                             "deceleration parameter is one half."
                         ),
                         data_source=data_source)
        self.add_dataset(sl_3v, None, dataset_name="SRSD Bonus 20", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=[f'{self.FINE_STRUCTURE_CONSTANT}', '^2', '*', f'{self.PLANCK_CONSTANT}', '^2',
                                       '/', '(', '4', '*', 'pi', '*', f'{self.ELECTRON_MASS}', '^2', '*',
                                       f'{self.SPEED_OF_LIGHT}', '^2', ')', '*', 'X_0', '^2', '/', 'X_1', '^2', '*',
                                       '(', 'X_0', '/', 'X_1', '+', 'X_1', '/', 'X_0', '-', 'sin', '(', 'X_2', ')',
                                       '^2', ')'],
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata, samplers=[
                LogUniformSampling(1000000000.0, 100000000000.0, uses_negative=False),
                LogUniformSampling(1000000000.0, 100000000000.0, uses_negative=False),
                UniformSampling(0, 6.283185307179586, uses_negative=False),
            ],
                         variables={"X_0": VariableInfo("omega_0", "angular frequency of the scattered photon"),
                                    "X_1": VariableInfo("omega", "angular frequency of the incident photon"),
                                    "X_2": VariableInfo("beta", "scattering angle")},
                         target=VariableInfo("A", "differential scattering cross-section"),
                         description=(
                             "Differential cross-section for scattering a photon off a free electron at rest, as a "
                             "function of the incident and scattered photon frequencies and the scattering angle. It "
                             "is positive and even in beta; when the two frequencies are equal (low-energy limit) it "
                             "is symmetric under beta -> pi - beta, largest forward and backward and smallest at "
                             "right angles."
                         ),
                         data_source=data_source)

    # fmt: on
