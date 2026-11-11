"""Physical units of variables, constants and targets.

A [Unit][SRToolkit.utils.units.Unit] is a product of base dimensions raised to rational exponents,
e.g. metres per second is ``m s^-1``. The predefined units are constants of this module, named by
their SI symbols, and combine with ``*``, ``/`` and ``**``:

```python
from SRToolkit.utils import units as u

velocity = u.m / u.s
entropy = u.J / u.K
u.J == u.kg * u.m**2 / u.s**2  # True: derived units are products of base units
```

A misspelled name (``u.meter``) is an ``AttributeError``. The predefined units are the seven SI base
units ``m``, ``kg``, ``s``, ``A``, ``K``, ``mol``, ``cd`` and the derived units ``N``, ``J``,
``W``, ``Pa``, ``Hz``, ``C`` (coulomb), ``V``, ``Ohm``, ``F``, ``T``. A unit is a dimension, not a
scale, so there are no prefixes: a kilometre has the unit ``m``. State the scale of the data in the
variable's [VariableInfo][SRToolkit.dataset.background_knowledge.VariableInfo] description instead,
e.g. ``VariableInfo("d", "distance in km", u.m)``. Checks such as
[ConsistentUnits][SRToolkit.utils.expression_constraints.ConsistentUnits] treat km and m alike; a
free constant absorbs the conversion factor.

## Custom units

[Unit][SRToolkit.utils.units.Unit] builds a unit from a dict of exponents. Its names must be
predefined units, so a typo such as ``"meter"`` is an error, unless they are confirmed as custom
units, e.g. for information or money:

```python
from SRToolkit.utils.units import Unit

rate = Unit({"bit": 1, "s": -1}, custom_units=["bit"])
BIT = Unit({"bit": 1}, custom_units=["bit"])  # or name it once and combine
rate == BIT / u.s  # True
```

``custom_units`` declares new base dimensions. A unit made of existing ones is built with arithmetic
instead, e.g. a litre is ``u.m**3``: ``Unit({"L": 1}, custom_units=["L"])`` would be a new dimension
that does not match ``m^3``.

## Saving and reading

[to_dict][SRToolkit.utils.units.Unit.to_dict] writes a unit as ``{dimension: exponent}`` over base
dimensions, with the exponents as strings (``{"m": "1", "s": "-1"}``), and
[from_dict][SRToolkit.utils.units.Unit.from_dict] rebuilds it without confirmation: a saved unit
was valid when it was made. ``str`` is for reading only: ``"kg m^2 s^-2"``, ``"m^(1/2)"``, and
``"1"`` for the dimensionless unit ``Unit()``. ``str``, ``to_dict`` and
[dimensions][SRToolkit.utils.units.Unit.dimensions] list base dimensions by name in Python string
order, so uppercase comes first: ``A K cd kg m mol s``.

## How symbols act on units

A [UnitRule][SRToolkit.utils.units.UnitRule] says how an operator or function combines the units
of its arguments, e.g. ``*`` multiplies them and ``sin`` needs a dimensionless argument. Every
``"op"`` and ``"fn"`` symbol of a [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary]
has one (see [add_symbol][SRToolkit.utils.symbol_library.SymbolLibrary.add_symbol]), so a custom
symbol is described once, where it is defined:

```python
from fractions import Fraction

from SRToolkit.utils import FN_PREFIX, SymbolLibrary
from SRToolkit.utils.units import UnitRule

sl = SymbolLibrary.default_symbols(2)
sl.add_symbol("cbrt", "fn", FN_PREFIX, "np.cbrt({})", unit_rule=UnitRule.power(Fraction(1, 3)))
```

Units are used by [ConsistentUnits][SRToolkit.utils.expression_constraints.ConsistentUnits], by
[VariableInfo][SRToolkit.dataset.background_knowledge.VariableInfo] and by the grammar constraint
[DimensionalConsistency][SRToolkit.utils.grammar.constraints.DimensionalConsistency].
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import ClassVar, Dict, Iterable, Mapping, Optional, Tuple, Union

_BASE_UNITS = ("m", "kg", "s", "A", "K", "mol", "cd")
# Derived units as exponents of base units.
_DERIVED_UNITS: Dict[str, Dict[str, int]] = {
    "N": {"kg": 1, "m": 1, "s": -2},
    "J": {"kg": 1, "m": 2, "s": -2},
    "W": {"kg": 1, "m": 2, "s": -3},
    "Pa": {"kg": 1, "m": -1, "s": -2},
    "Hz": {"s": -1},
    "C": {"A": 1, "s": 1},
    "V": {"kg": 1, "m": 2, "s": -3, "A": -1},
    "Ohm": {"kg": 1, "m": 2, "s": -3, "A": -2},
    "F": {"kg": -1, "m": -2, "s": 4, "A": 2},
    "T": {"kg": 1, "s": -2, "A": -1},
}


def _parse_exponent(value: Union[int, str, Fraction], where: str) -> Fraction:
    # Exponent of a saved unit or rule: int, Fraction or a string Fraction accepts (not a float).
    if isinstance(value, (int, Fraction)) and not isinstance(value, bool):
        return Fraction(value)
    if isinstance(value, str):
        try:
            return Fraction(value)
        except (ValueError, ZeroDivisionError):
            pass
    raise ValueError(f"{where}: invalid exponent {value!r}; expected an integer or a fraction such as '1/2'.")


@dataclass(frozen=True, init=False, repr=False)
class Unit:
    """A physical unit: base dimensions with rational exponents.

    Immutable (assigning an attribute raises ``AttributeError``) and hashable; units can be copied
    and pickled. Two units are equal when they have the same base dimensions with the same
    exponents, however they were built.

    Args:
        units: Exponents by unit name, e.g. ``{"m": 1, "s": -1}``. A name is a predefined unit
            (see the module docstring; a derived unit such as ``"J"`` stands for its base
            dimensions) or a name in ``custom_units``. Exponents are ``int`` or ``Fraction``; zero
            exponents are dropped. ``None`` (the default) gives the dimensionless unit.
        custom_units: Names of custom base dimensions that ``units`` may use, e.g. ``["bit"]``; a
            single string is one name. Each must be a valid Python identifier. A predefined name
            always means the predefined unit.

    Raises:
        ValueError: If a name in ``units`` is neither predefined nor in ``custom_units`` (the
            message names it and lists the predefined units), or a name in ``custom_units`` is not
            a valid Python identifier.
        TypeError: If an exponent is not an ``int`` or a ``Fraction``.

    Examples:
        >>> from fractions import Fraction
        >>> from SRToolkit.utils import units as u
        >>> v = Unit({"m": 1, "s": -1})
        >>> v == u.m / u.s
        True
        >>> str(Unit({"J": 1, "K": -1}))
        'K^-1 kg m^2 s^-2'
        >>> str(Unit({"bit": 1, "s": -1}, custom_units=["bit"]))
        'bit s^-1'
        >>> str(u.m ** Fraction(1, 2))
        'm^(1/2)'
    """

    _dims: Tuple[Tuple[str, Fraction], ...]

    def __init__(
        self, units: Optional[Mapping[str, Union[int, Fraction]]] = None, custom_units: Iterable[str] = ()
    ) -> None:
        custom = {custom_units} if isinstance(custom_units, str) else set(custom_units)
        invalid = sorted((n for n in custom if not (isinstance(n, str) and n.isidentifier())), key=repr)
        if invalid:
            raise ValueError(f"Unit: custom unit names must be valid Python identifiers, got {invalid}.")
        dims: Dict[str, Fraction] = {}
        for name, exponent in (units or {}).items():
            if not isinstance(exponent, (int, Fraction)) or isinstance(exponent, bool):
                raise TypeError(
                    f"Unit: the exponent of {name!r} must be an int or a Fraction, got {type(exponent).__name__}."
                )
            if name in _DERIVED_UNITS:
                base: Mapping[str, Union[int, Fraction]] = _DERIVED_UNITS[name]
            elif name in _BASE_UNITS or name in custom:
                base = {name: 1}
            else:
                raise ValueError(
                    f"Unit: {name!r} is not a predefined unit ({', '.join((*_BASE_UNITS, *_DERIVED_UNITS))}); "
                    f"confirm a custom unit with custom_units=[{name!r}]."
                )
            for dim, k in base.items():
                dims[dim] = dims.get(dim, Fraction(0)) + Fraction(exponent) * k
        object.__setattr__(self, "_dims", tuple(sorted((name, exp) for name, exp in dims.items() if exp != 0)))

    @property
    def dimensions(self) -> Mapping[str, Fraction]:
        """The non-zero exponents by dimension name, read-only, in name order; empty if dimensionless."""
        return MappingProxyType(dict(self._dims))

    def __mul__(self, other: Unit) -> Unit:
        """Return the product of two units: exponents of the same dimension add.

        Args:
            other: A unit.

        Returns:
            The product.
        """
        if not isinstance(other, Unit):
            return NotImplemented
        dims = dict(self._dims)
        for name, exp in other._dims:
            dims[name] = dims.get(name, Fraction(0)) + exp
        return Unit(dims, custom_units=dims)

    def __truediv__(self, other: Unit) -> Unit:
        """Return the quotient of two units: the exponents of ``other`` are subtracted.

        Args:
            other: A unit.

        Returns:
            The quotient.
        """
        if not isinstance(other, Unit):
            return NotImplemented
        return self * other**-1

    def __pow__(self, exponent: Union[int, Fraction]) -> Unit:
        """Return the unit raised to a rational power: every exponent is multiplied by it.

        Args:
            exponent: An ``int`` or a ``Fraction``. ``0`` gives the dimensionless unit.

        Returns:
            The power.

        Raises:
            TypeError: If ``exponent`` is not an ``int`` or a ``Fraction`` (e.g. a ``float``).
        """
        if not isinstance(exponent, (int, Fraction)) or isinstance(exponent, bool):
            raise TypeError(f"Unit: the power must be an int or a Fraction, got {type(exponent).__name__}.")
        dims = {name: exp * exponent for name, exp in self._dims}
        return Unit(dims, custom_units=dims)

    def __str__(self) -> str:
        """Return the unit as text: ``"1"`` if dimensionless, otherwise the factors in name order.

        A factor is ``<name>`` for exponent 1, ``<name>^<k>`` for another integer (``s^-2``) and
        ``<name>^(<p>/<q>)`` for a fraction, the sign inside the brackets (``m^(-1/2)``), separated
        by single spaces.

        Returns:
            The text, e.g. ``"kg m^2 s^-2"``.
        """
        if not self._dims:
            return "1"
        factors = []
        for name, exp in self._dims:
            if exp == 1:
                factors.append(name)
            else:
                factors.append(f"{name}^{exp.numerator if exp.denominator == 1 else f'({exp})'}")
        return " ".join(factors)

    def __repr__(self) -> str:
        """Return ``Unit('<str(self)>')``.

        Returns:
            The representation.
        """
        return f"Unit('{self}')"

    def to_dict(self) -> dict:
        """Return the unit as a JSON-safe dictionary.

        Returns:
            ``{dimension: str(exponent)}`` in name order, e.g. ``{"m": "1", "s": "-1"}`` or
            ``{"m": "1/2"}``; ``{}`` if dimensionless.

        Examples:
            >>> from SRToolkit.utils import units as u
            >>> u.J.to_dict()
            {'kg': '1', 'm': '2', 's': '-2'}
        """
        return {name: str(exp) for name, exp in self._dims}

    @classmethod
    def from_dict(cls, d: Mapping[str, Union[int, str, Fraction]]) -> Unit:
        """Rebuild a unit from the output of ``to_dict``.

        Every name is accepted without confirmation (a saved unit was valid when it was made): a
        predefined name means the predefined unit, any other valid Python identifier a custom one.
        Exponents may be ``int``, ``Fraction`` or strings such as ``"-2"`` and ``"1/2"``.

        Args:
            d: A mapping from unit name to exponent.

        Returns:
            The unit.

        Raises:
            ValueError: If a name is not a valid Python identifier, or an exponent is a ``float`` or
                a string that is not a number.
        """
        exponents = {name: _parse_exponent(value, "Unit.from_dict") for name, value in d.items()}
        return cls(exponents, custom_units=exponents)


# Predefined units, named by their SI symbols.
m = Unit({"m": 1})
"""Metre, SI base unit of length."""
kg = Unit({"kg": 1})
"""Kilogram, SI base unit of mass."""
s = Unit({"s": 1})
"""Second, SI base unit of time."""
A = Unit({"A": 1})
"""Ampere, SI base unit of electric current."""
K = Unit({"K": 1})
"""Kelvin, SI base unit of temperature."""
mol = Unit({"mol": 1})
"""Mole, SI base unit of amount of substance."""
cd = Unit({"cd": 1})
"""Candela, SI base unit of luminous intensity."""
N = Unit({"N": 1})
"""Newton (force), ``kg m s^-2``."""
J = Unit({"J": 1})
"""Joule (energy), ``kg m^2 s^-2``."""
W = Unit({"W": 1})
"""Watt (power), ``kg m^2 s^-3``."""
Pa = Unit({"Pa": 1})
"""Pascal (pressure), ``kg m^-1 s^-2``."""
Hz = Unit({"Hz": 1})
"""Hertz (frequency), ``s^-1``."""
C = Unit({"C": 1})
"""Coulomb (electric charge), ``A s``."""
V = Unit({"V": 1})
"""Volt (voltage), ``A^-1 kg m^2 s^-3``."""
Ohm = Unit({"Ohm": 1})
"""Ohm (resistance), ``A^-2 kg m^2 s^-3``."""
F = Unit({"F": 1})
"""Farad (capacitance), ``A^2 kg^-1 m^-2 s^4``."""
T = Unit({"T": 1})
"""Tesla (magnetic flux density), ``A^-1 kg s^-2``."""


@dataclass(frozen=True, repr=False)
class UnitRule:
    """How an operator or function combines the units of its arguments.

    Immutable (assigning an attribute raises ``AttributeError``) and hashable; rules can be copied
    and pickled, and are compared by kind and exponent. Use the class attributes
    and [power][SRToolkit.utils.units.UnitRule.power]; the constructor is not part of the API.

    | Rule | Arity | Result | Conflict when |
    |---|---|---|---|
    | ``SAME`` | 1 or 2 | the arguments' unit | with two arguments, their units differ |
    | ``MULTIPLY`` | 2 | product of the units | never |
    | ``DIVIDE`` | 2 | left unit divided by right unit | never |
    | ``EXPONENT`` | 2 | if the exponent is a single number token (defined in [ConsistentUnits][SRToolkit.utils.expression_constraints.ConsistentUnits]), the base unit to that power; otherwise dimensionless | the exponent is not a number token and the base or the exponent is not dimensionless |
    | ``power(k)`` | 1 | the argument's unit to the power ``k`` | never |
    | ``DIMENSIONLESS`` | 1 or 2 | dimensionless | an argument is not dimensionless |

    What counts as a number token, how conflicts are counted and the messages are defined by
    [ConsistentUnits][SRToolkit.utils.expression_constraints.ConsistentUnits]. The rules of the
    predefined symbols are listed in
    [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].

    Attributes:
        SAME: Arguments must share a unit, which the result has (``+``, ``floor``).
        MULTIPLY: The result's unit is the product of the arguments' units (``*``).
        DIVIDE: The result's unit is the left unit divided by the right one (``/``).
        EXPONENT: A number exponent scales the base's unit; otherwise both must be dimensionless
            (``^``).
        DIMENSIONLESS: Arguments must be dimensionless, and so is the result (``sin``, ``exp``).
        kind: The rule's name: ``"same"``, ``"multiply"``, ``"divide"``, ``"exponent"``,
            ``"power"`` or ``"dimensionless"``.
        exponent: ``k`` as a ``Fraction`` for ``power(k)``, otherwise ``None``.

    Examples:
        >>> UnitRule.power(2).exponent
        Fraction(2, 1)
        >>> UnitRule.SAME.arities
        (1, 2)
    """

    SAME: ClassVar[UnitRule]
    MULTIPLY: ClassVar[UnitRule]
    DIVIDE: ClassVar[UnitRule]
    EXPONENT: ClassVar[UnitRule]
    DIMENSIONLESS: ClassVar[UnitRule]

    _ARITIES = {
        "same": (1, 2),
        "multiply": (2,),
        "divide": (2,),
        "exponent": (2,),
        "power": (1,),
        "dimensionless": (1, 2),
    }

    kind: str
    exponent: Optional[Fraction] = None

    @property
    def arities(self) -> Tuple[int, ...]:
        """The numbers of arguments the rule accepts, as a tuple in increasing order, e.g. ``(1, 2)``."""
        return UnitRule._ARITIES[self.kind]

    @classmethod
    def power(cls, k: Union[int, Fraction]) -> UnitRule:
        """Return the rule of a function that raises its argument's unit to the power ``k``.

        Args:
            k: The power, an ``int`` or a ``Fraction``.

        Returns:
            The rule.

        Raises:
            TypeError: If ``k`` is not an ``int`` or a ``Fraction`` (e.g. a ``float``).

        Examples:
            >>> from fractions import Fraction
            >>> UnitRule.power(Fraction(1, 2)) == UnitRule.power(Fraction(2, 4))
            True
        """
        if not isinstance(k, (int, Fraction)) or isinstance(k, bool):
            raise TypeError(f"UnitRule.power: the power must be an int or a Fraction, got {type(k).__name__}.")
        return cls("power", Fraction(k))

    def __repr__(self) -> str:
        """Return the expression that builds the rule, e.g. ``UnitRule.SAME``.

        A power is ``UnitRule.power(<k>)`` for an integer ``k`` and
        ``UnitRule.power(Fraction(<p>, <q>))`` otherwise.

        Returns:
            The representation, e.g. ``"UnitRule.power(2)"`` or
            ``"UnitRule.power(Fraction(1, 2))"``.
        """
        if self.exponent is None:
            return f"UnitRule.{self.kind.upper()}"
        k = self.exponent
        return f"UnitRule.power({k.numerator if k.denominator == 1 else f'Fraction({k.numerator}, {k.denominator})'})"

    def to_dict(self) -> dict:
        """Return the rule as a JSON-safe dictionary.

        Returns:
            ``{"kind": kind}``, plus ``"exponent": str(exponent)`` for ``power(k)``, e.g.
            ``{"kind": "power", "exponent": "1/2"}``.
        """
        if self.exponent is None:
            return {"kind": self.kind}
        return {"kind": self.kind, "exponent": str(self.exponent)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Union[int, str, Fraction]]) -> UnitRule:
        """Rebuild a rule from the output of ``to_dict``.

        Other keys are ignored.

        Args:
            d: A dictionary with the key ``"kind"``, and ``"exponent"`` for ``"power"``, read as in
                [Unit.from_dict][SRToolkit.utils.units.Unit.from_dict].

        Returns:
            The rule.

        Raises:
            ValueError: If ``"kind"`` is missing or not one of the rules' names, or a power's
                exponent is missing or invalid.
        """
        kind = d.get("kind")
        if kind not in cls._ARITIES:
            raise ValueError(f"UnitRule.from_dict: unknown kind {kind!r}; expected one of {', '.join(cls._ARITIES)}.")
        if kind != "power":
            return cls(str(kind))
        if "exponent" not in d:
            raise ValueError("UnitRule.from_dict: a power rule needs an 'exponent'.")
        return cls("power", _parse_exponent(d["exponent"], "UnitRule.from_dict"))


UnitRule.SAME = UnitRule("same")
UnitRule.MULTIPLY = UnitRule("multiply")
UnitRule.DIVIDE = UnitRule("divide")
UnitRule.EXPONENT = UnitRule("exponent")
UnitRule.DIMENSIONLESS = UnitRule("dimensionless")
