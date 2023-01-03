from abc import abstractmethod
from collections.abc import Iterable
from datetime import date
from heapq import merge
from typing import Any, Protocol, TypeVar


__all__ = [
    'float_approx_eq',
    'merge_by_date',
    'Ordered'
]


TOrdered = TypeVar('TOrdered', bound='Ordered')


class Ordered(Protocol):
    """A type with a total ordering."""

    @abstractmethod
    def __eq__(self, other: Any, /) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __lt__(self: TOrdered, other: TOrdered, /) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __le__(self: TOrdered, other: TOrdered, /) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError()


class HasDate(Protocol):
    date: date


THasDate = TypeVar('THasDate', bound=HasDate)


def merge_by_date(iterables: Iterable[Iterable[THasDate]]) -> Iterable[THasDate]:
    return merge(*iterables, key=lambda item: item.date)


def float_approx_eq(a: float, b: float, /, *, tolerance: float = 1e-6) -> bool:
    """Checks if two floating point values are equal within a given tolerance."""

    if tolerance < 0:
        raise ValueError('tolerance must be nonnegative')
    return abs(a - b) < tolerance
