from abc import abstractmethod
from collections.abc import Iterable
from datetime import date
from heapq import merge
from typing import Any, Protocol, TypeVar


__all__ = [
    'merge_by_date',
    'Ordered'
]


T_Ordered = TypeVar('T_Ordered', bound='Ordered')


class Ordered(Protocol):
    """A type with a total ordering."""

    @abstractmethod
    def __eq__(self, other: Any, /) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __lt__(self: T_Ordered, other: T_Ordered, /) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __le__(self: T_Ordered, other: T_Ordered, /) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError()


class HasDate(Protocol):
    date: date


T_HasDate = TypeVar('T_HasDate', bound=HasDate)


def merge_by_date(iterables: Iterable[Iterable[T_HasDate]]) -> Iterable[T_HasDate]:
    """Merges multiple sorted iterables into one sorted iterable.
        Objects are ordered by their `date` attribute."""

    return merge(*iterables, key=lambda item: item.date)
