from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from math import isclose
from typing import Any


__all__ = [
    'approx_floats'
]


class approx_floats:
    """Generic comparator that uses approximate equality for floats.

        The comparator is applied recursively to dataclasses and collections.
        All other types are compared exactly."""

    def __init__(self, value: Any, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> None:
        if isinstance(value, approx_floats):
            raise TypeError()

        self._value = value
        self._rel_tol = rel_tol
        self._abs_tol = abs_tol

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, approx_floats):
            raise TypeError()

        if {type(self._value), type(other)} in ({float}, {float, int}):
            # Float comparing against float or int - use tolerance.
            return isclose(self._value, other, rel_tol=self._rel_tol, abs_tol=self._abs_tol)
        elif isinstance(self._value, str) and isinstance(other, str):
            # str is a sequence of str, annoyingly, so must be handled separately.
            return self._value == other
        elif isinstance(self._value, Sequence) and isinstance(other, Sequence) and type(self._value) == type(other):
            # Comparing two sequences of same type - recurse into elements.
            return (len(self._value) == len(other) and
                    all(type(self)(e1) == e2 for e1, e2 in zip(self._value, other)))
        elif isinstance(self._value, Mapping) and isinstance(other, Mapping) and type(self._value) == type(other):
            # Comparing two mappings of same type - recurse into values.
            # Note floats as keys are not compared with tolerance. Floats probably shouldn't be used as keys anyway.
            return (self._value.keys() == other.keys() and
                    all(type(self)(self._value[key]) == other[key] for key in self._value.keys()))
        elif is_dataclass(self._value) and is_dataclass(other) and type(self._value) == type(other):
            # Comparing two dataclasses of same type - recurse into fields.
            return all(type(self)(getattr(self._value, field.name)) == getattr(other, field.name)
                        for field in fields(self._value))
        elif type(self._value) == type(other):
            # Comparing any other values of the same type - must be exactly equal.
            return self._value == other
        else:
            # Comparing two values of differing types - in a unit test, probably a mistake.
            raise TypeError()

    def __str__(self) -> str:
        return f'{type(self).__name__}({self._value})'

    def __repr__(self) -> str:
        return str(self)
