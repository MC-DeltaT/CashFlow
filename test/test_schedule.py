from datetime import date

from cashflow.date_time import DateRange, Month, Week
from cashflow.probability import DiscreteDistribution
from cashflow.schedule import (
    Daily, DayOfMonthDistribution, DayOfWeekDistribution, Never, Once, SimpleDayOfMonthSchedule,
    SimpleDayOfWeekSchedule, Weekdays, Weekends)


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


def test_weekends_iterate_no_exceptions() -> None:
    s = Weekends(DateRange.inclusive(date(2023, 1, 8), date(2023, 3, 13)))
    events = tuple(s.iterate(DateRange.inclusive(date(2022, 12, 13), date(2023, 2, 9))))
    expected = (
        DiscreteDistribution.singular(date(2023, 1, 8)),
        DiscreteDistribution.singular(date(2023, 1, 14)),
        DiscreteDistribution.singular(date(2023, 1, 15)),
        DiscreteDistribution.singular(date(2023, 1, 21)),
        DiscreteDistribution.singular(date(2023, 1, 22)),
        DiscreteDistribution.singular(date(2023, 1, 28)),
        DiscreteDistribution.singular(date(2023, 1, 29)),
        DiscreteDistribution.singular(date(2023, 2, 4)),
        DiscreteDistribution.singular(date(2023, 2, 5))
    )
    assert events == expected

def test_weekends_iterate_exceptions() -> None:
    s = Weekends(range=DateRange.inclusive(date(2023, 1, 1), date(2023, 4, 1)),
        exceptions=(date(2023, 1, 3), DateRange.inclusive(date(2023, 1, 27), date(2023, 2, 11)), date(2023, 2, 26)))
    events = tuple(s.iterate(DateRange.inclusive(date(2023, 1, 1), date(2023, 3, 1))))
    expected = (
        DiscreteDistribution.singular(date(2023, 1, 1)),
        DiscreteDistribution.singular(date(2023, 1, 7)),
        DiscreteDistribution.singular(date(2023, 1, 8)),
        DiscreteDistribution.singular(date(2023, 1, 14)),
        DiscreteDistribution.singular(date(2023, 1, 15)),
        DiscreteDistribution.singular(date(2023, 1, 21)),
        DiscreteDistribution.singular(date(2023, 1, 22)),
        DiscreteDistribution.singular(date(2023, 2, 12)),
        DiscreteDistribution.singular(date(2023, 2, 18)),
        DiscreteDistribution.singular(date(2023, 2, 19)),
        DiscreteDistribution.singular(date(2023, 2, 25))
    )
    assert events == expected


def test_simple_day_of_week_schedule() -> None:
    d1 = DayOfWeekDistribution.singular(5)
    d2 = DayOfWeekDistribution.uniformly_of(0, 2)
    s = SimpleDayOfWeekSchedule((
        d1,
        DayOfWeekDistribution.null(),
        d2
    ))
    expected = (d1, d2)
    assert tuple(s.iterate(Week.of(date(1900, 1, 2)))) == expected
    assert tuple(s.iterate(Week.of(date(2022, 12, 4)))) == expected
    assert tuple(s.iterate(Week.of(date(2099, 5, 13)))) == expected


# TODO: test Weekly


def test_simple_day_of_month_schedule_normal() -> None:
    d1 = DayOfMonthDistribution.uniformly_of(30)
    d2 = DayOfMonthDistribution.uniformly_of(1, 3, 5, 29)
    s = SimpleDayOfMonthSchedule((
        DayOfMonthDistribution.null(),
        d1,
        d2
    ))
    expected = (d1, d2)
    assert tuple(s.iterate(Month(1900, 1))) == expected
    assert tuple(s.iterate(Month(2022, 12))) == expected
    assert tuple(s.iterate(Month(2222, 9))) == expected

def test_simple_day_of_month_schedule_invalid_dates() -> None:
    s = SimpleDayOfMonthSchedule((
        DayOfMonthDistribution.uniformly_of(29, 30, 31),
        DayOfMonthDistribution.uniformly_of(31)
    ))
    # Leap year.
    result1 = tuple(s.iterate(Month(2020, 2)))
    assert len(result1) == 1 and result1[0].approx_eq(DayOfMonthDistribution.from_probabilities({29: 1/3}))
    # Non-leap year.
    assert tuple(s.iterate(Month(2023, 2))) == ()
    result2 = tuple(s.iterate(Month(2023, 4)))
    assert len(result2) == 1 and result2[0].approx_eq(DayOfMonthDistribution.from_probabilities({29: 1/3, 30: 1/3}))


# TODO: test Monthly
