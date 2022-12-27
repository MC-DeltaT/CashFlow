from datetime import date

from cashflow.date_time import DateRange
from cashflow.probability import DiscreteDistribution
from cashflow.schedule import Daily, Never, Once, Weekdays


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

def test_once_iterate_all() -> None:
    s = Once(date(2022, 12, 27))
    events = tuple(s.iterate(DateRange.all()))
    assert events == (DiscreteDistribution.singular(date(2022, 12, 27)),)

def test_once_iterate_distribution_in_range() -> None:
    d = DiscreteDistribution.uniformly_of(date(2013, 7, 3), date(2013, 7, 10), date(2013, 7, 15))
    s = Once(d)
    events = tuple(s.iterate(DateRange.half_open(date(2013, 7, 8), date(2013, 7, 14))))
    assert events == (d,)

def test_once_iterate_distribution_all() -> None:
    d = DiscreteDistribution.uniformly_of(date(2013, 7, 3), date(2013, 7, 10), date(2013, 7, 15))
    s = Once(d)
    events = tuple(s.iterate(DateRange.all()))
    assert events == (d,)

def test_once_iterate_distribution_not_in_range() -> None:
    d = DiscreteDistribution.uniformly_of(date(2013, 7, 3), date(2013, 7, 10), date(2013, 7, 15))
    s = Once(d)
    events = tuple(s.iterate(DateRange.half_open(date(2013, 8, 10), date(2013, 12, 31))))
    assert events == ()


def test_daily_iterate_no_exceptions() -> None:
    s = Daily(DateRange.inclusive(date(2027, 1, 7), date(2028, 1, 6)))
    events = tuple(s.iterate(DateRange.inclusive(date(2027, 12, 28), date(2028, 7, 1))))
    expected = (
        DiscreteDistribution.singular(date(2027, 12, 28)),
        DiscreteDistribution.singular(date(2027, 12, 29)),
        DiscreteDistribution.singular(date(2027, 12, 30)),
        DiscreteDistribution.singular(date(2027, 12, 31)),
        DiscreteDistribution.singular(date(2028, 1, 1)),
        DiscreteDistribution.singular(date(2028, 1, 2)),
        DiscreteDistribution.singular(date(2028, 1, 3)),
        DiscreteDistribution.singular(date(2028, 1, 4)),
        DiscreteDistribution.singular(date(2028, 1, 5)),
        DiscreteDistribution.singular(date(2028, 1, 6))
    )
    assert events == expected

def test_daily_iterate_exceptions() -> None:
    s = Daily(range=DateRange.half_open(date(2027, 1, 1), date(2027, 7, 1)),
        exceptions=(date(2027, 1, 1), DateRange.inclusive(date(2027, 1, 5), date(2027, 2, 5)), date(2027, 1, 31),
            date(2006, 10, 2)))
    events = tuple(s.iterate(DateRange.inclusive(date(2026, 1, 6), date(2027, 2, 8))))
    expected = (
        DiscreteDistribution.singular(date(2027, 1, 2)),
        DiscreteDistribution.singular(date(2027, 1, 3)),
        DiscreteDistribution.singular(date(2027, 1, 4)),
        DiscreteDistribution.singular(date(2027, 2, 6)),
        DiscreteDistribution.singular(date(2027, 2, 7)),
        DiscreteDistribution.singular(date(2027, 2, 8))
    )
    assert events == expected


def test_weekdays_iterate_no_exceptions() -> None:
    s = Weekdays(DateRange.inclusive(date(2022, 4, 3), date(2022, 8, 2)))
    events = tuple(s.iterate(DateRange.inclusive(date(2022, 3, 23), date(2022, 4, 18))))
    expected = (
        DiscreteDistribution.singular(date(2022, 4, 4)),
        DiscreteDistribution.singular(date(2022, 4, 5)),
        DiscreteDistribution.singular(date(2022, 4, 6)),
        DiscreteDistribution.singular(date(2022, 4, 7)),
        DiscreteDistribution.singular(date(2022, 4, 8)),
        DiscreteDistribution.singular(date(2022, 4, 11)),
        DiscreteDistribution.singular(date(2022, 4, 12)),
        DiscreteDistribution.singular(date(2022, 4, 13)),
        DiscreteDistribution.singular(date(2022, 4, 14)),
        DiscreteDistribution.singular(date(2022, 4, 15)),
        DiscreteDistribution.singular(date(2022, 4, 18)),
    )
    assert events == expected

def test_weekdays_iterate_exceptions() -> None:
    s = Weekdays(range=DateRange.inclusive(date(2015, 6, 8), date(2015, 12, 7)),
        exceptions=(DateRange.inclusive(date(2015, 7, 15), date(2015, 9, 16)), date(2015, 9, 26), date(2015, 9, 25),
            date(2015, 10, 1), DateRange.beginning_at(date(2015, 11, 1))))
    events = tuple(s.iterate(DateRange.inclusive(date(2015, 7, 11), date(2015, 9, 30))))
    expected = (
        DiscreteDistribution.singular(date(2015, 7, 13)),
        DiscreteDistribution.singular(date(2015, 7, 14)),
        DiscreteDistribution.singular(date(2015, 9, 17)),
        DiscreteDistribution.singular(date(2015, 9, 18)),
        DiscreteDistribution.singular(date(2015, 9, 21)),
        DiscreteDistribution.singular(date(2015, 9, 22)),
        DiscreteDistribution.singular(date(2015, 9, 23)),
        DiscreteDistribution.singular(date(2015, 9, 24)),
        DiscreteDistribution.singular(date(2015, 9, 28)),
        DiscreteDistribution.singular(date(2015, 9, 29)),
        DiscreteDistribution.singular(date(2015, 9, 30)),
    )
    assert events == expected

# TODO
