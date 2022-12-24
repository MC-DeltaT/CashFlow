from datetime import date

from cashflow.date_time import DateRange
from cashflow.probability import DiscreteDistribution
from cashflow.schedule import Never, Once


def test_never_iterate_empty_range() -> None:
    events = tuple(Never().iterate(DateRange.half_open(date(2022, 12, 24), date(2022, 12, 24))))
    assert events == ()

def test_never_iterate_nonempty_range() -> None:
    events = tuple(Never().iterate(DateRange.half_open(date(2022, 12, 24), date(2023, 12, 24))))
    assert events == ()

def test_never_iterate_all() -> None:
    events = tuple(Never().iterate(DateRange.all()))
    assert events == ()


def test_once_iterate_date_in_range() -> None:
    s = Once(date(2022, 12, 25))
    events = tuple(s.iterate(DateRange.half_open(date(2022, 12, 1), date(2023, 1, 4))))
    assert events == (DiscreteDistribution.singular(date(2022, 12, 25)),)

def test_once_iterate_date_not_in_range() -> None:
    s = Once(date(2022, 12, 25))
    events = tuple(s.iterate(DateRange.half_open(date(2022, 9, 19), date(2022, 12, 25))))
    assert events == ()

def test_once_iterate_distribution_in_range() -> None:
    d = DiscreteDistribution.uniformly_of(date(2013, 7, 3), date(2013, 7, 10), date(2013, 7, 15))
    s = Once(d)
    events = tuple(s.iterate(DateRange.half_open(date(2013, 7, 8), date(2013, 7, 14))))
    assert events == (d,)

def test_once_iterate_distribution_not_in_range() -> None:
    d = DiscreteDistribution.uniformly_of(date(2013, 7, 3), date(2013, 7, 10), date(2013, 7, 15))
    s = Once(d)
    events = tuple(s.iterate(DateRange.half_open(date(2013, 8, 10), date(2013, 12, 31))))
    assert events == ()


# TODO
