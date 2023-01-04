from dataclasses import dataclass
from datetime import date

from cashflow.utility import merge_by_date


@dataclass(order=False)
class TypeWithDate:
    date: date
    value: complex        # Something that's definitely not ordered


def test_merge_by_date_zero_iterables() -> None:
    assert tuple(merge_by_date(())) == ()

def test_merge_by_date_empty_iterables() -> None:
    assert tuple(merge_by_date(((), [], {}))) == ()

def test_merge_by_date_single_iterable() -> None:
    items = (
        TypeWithDate(date(2000, 1, 2), 1+1j),
        TypeWithDate(date(2000, 1, 3), 4+3j),
        TypeWithDate(date(2000, 4, 3), 2+2j))
    assert tuple(merge_by_date((items,))) == items

def test_merge_by_date_multiple_iterables() -> None:
    items1 = (
        TypeWithDate(date(2000, 1, 2), 1+1j),
        TypeWithDate(date(2000, 1, 3), 4+3j),
        TypeWithDate(date(2000, 4, 3), 2+2j))
    items2 = (
        TypeWithDate(date(2000, 2, 7), 4+3j),
        TypeWithDate(date(2000, 5, 9), 2+2j))
    items3 = (
        TypeWithDate(date(2000, 1, 2), 1+1j),
        TypeWithDate(date(2001, 4, 1), 4+2j),
        TypeWithDate(date(2002, 5, 1), 6+3j))
    items4 = (TypeWithDate(date(2000, 3, 3), 3+3j),)
    expected = (
        TypeWithDate(date(2000, 1, 2), 1+1j),
        TypeWithDate(date(2000, 1, 2), 1+1j),
        TypeWithDate(date(2000, 1, 3), 4+3j),
        TypeWithDate(date(2000, 2, 7), 4+3j),
        TypeWithDate(date(2000, 3, 3), 3+3j),
        TypeWithDate(date(2000, 4, 3), 2+2j),
        TypeWithDate(date(2000, 5, 9), 2+2j),
        TypeWithDate(date(2001, 4, 1), 4+2j),
        TypeWithDate(date(2002, 5, 1), 6+3j)
    )
    assert tuple(merge_by_date((items1, items2, items3, items4))) == expected
