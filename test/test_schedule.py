from datetime import date

from cashflow.date_time import DateRange, Month, Week
from cashflow.probability import DiscreteOutcome
from cashflow.schedule import (
    Daily, DateDistribution, DayOfMonthDistribution, DayOfWeekDistribution, Monthly, Never, Once,
    SimpleDayOfMonthSchedule, SimpleDayOfWeekSchedule, Weekdays, Weekends, Weekly)


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
    assert events == (DateDistribution.singular(date(2022, 12, 25)),)

def test_once_iterate_date_not_in_range() -> None:
    s = Once(date(2022, 12, 25))
    events = tuple(s.iterate(DateRange.half_open(date(2022, 9, 19), date(2022, 12, 25))))
    assert events == ()

def test_once_iterate_all() -> None:
    s = Once(date(2022, 12, 27))
    events = tuple(s.iterate(DateRange.all()))
    assert events == (DateDistribution.singular(date(2022, 12, 27)),)

def test_once_iterate_distribution_in_range() -> None:
    d = DateDistribution.uniformly_of(date(2013, 7, 3), date(2013, 7, 10), date(2013, 7, 15))
    s = Once(d)
    events = tuple(s.iterate(DateRange.half_open(date(2013, 7, 8), date(2013, 7, 14))))
    assert events == (d,)

def test_once_iterate_distribution_all() -> None:
    d = DateDistribution.uniformly_of(date(2013, 7, 3), date(2013, 7, 10), date(2013, 7, 15))
    s = Once(d)
    events = tuple(s.iterate(DateRange.all()))
    assert events == (d,)

def test_once_iterate_distribution_not_in_range() -> None:
    d = DateDistribution.uniformly_of(date(2013, 7, 3), date(2013, 7, 10), date(2013, 7, 15))
    s = Once(d)
    events = tuple(s.iterate(DateRange.half_open(date(2013, 8, 10), date(2013, 12, 31))))
    assert events == ()


def test_daily_iterate_no_excludes() -> None:
    s = Daily(DateRange.inclusive(date(2027, 1, 7), date(2028, 1, 6)))
    events = tuple(s.iterate(DateRange.inclusive(date(2027, 12, 28), date(2028, 7, 1))))
    expected = (
        DateDistribution.singular(date(2027, 12, 28)),
        DateDistribution.singular(date(2027, 12, 29)),
        DateDistribution.singular(date(2027, 12, 30)),
        DateDistribution.singular(date(2027, 12, 31)),
        DateDistribution.singular(date(2028, 1, 1)),
        DateDistribution.singular(date(2028, 1, 2)),
        DateDistribution.singular(date(2028, 1, 3)),
        DateDistribution.singular(date(2028, 1, 4)),
        DateDistribution.singular(date(2028, 1, 5)),
        DateDistribution.singular(date(2028, 1, 6))
    )
    assert events == expected

def test_daily_iterate_excludes() -> None:
    s = Daily(range=DateRange.half_open(date(2027, 1, 1), date(2027, 7, 1)),
        exclude=(date(2027, 1, 1), DateRange.inclusive(date(2027, 1, 5), date(2027, 2, 5)), date(2027, 1, 31),
            date(2006, 10, 2)))
    events = tuple(s.iterate(DateRange.inclusive(date(2026, 1, 6), date(2027, 2, 8))))
    expected = (
        # 2027/1/1 excluded
        DateDistribution.singular(date(2027, 1, 2)),
        DateDistribution.singular(date(2027, 1, 3)),
        DateDistribution.singular(date(2027, 1, 4)),
        # 2027/1/5 to 2027/2/5 excluded
        DateDistribution.singular(date(2027, 2, 6)),
        DateDistribution.singular(date(2027, 2, 7)),
        DateDistribution.singular(date(2027, 2, 8))
    )
    assert events == expected


def test_weekdays_iterate_no_excludes() -> None:
    s = Weekdays(DateRange.inclusive(date(2022, 4, 3), date(2022, 8, 2)))
    events = tuple(s.iterate(DateRange.inclusive(date(2022, 3, 23), date(2022, 4, 18))))
    expected = (
        DateDistribution.singular(date(2022, 4, 4)),
        DateDistribution.singular(date(2022, 4, 5)),
        DateDistribution.singular(date(2022, 4, 6)),
        DateDistribution.singular(date(2022, 4, 7)),
        DateDistribution.singular(date(2022, 4, 8)),
        DateDistribution.singular(date(2022, 4, 11)),
        DateDistribution.singular(date(2022, 4, 12)),
        DateDistribution.singular(date(2022, 4, 13)),
        DateDistribution.singular(date(2022, 4, 14)),
        DateDistribution.singular(date(2022, 4, 15)),
        DateDistribution.singular(date(2022, 4, 18))
    )
    assert events == expected

def test_weekdays_iterate_excludes() -> None:
    s = Weekdays(range=DateRange.inclusive(date(2015, 6, 8), date(2015, 12, 7)),
        exclude=(DateRange.inclusive(date(2015, 7, 15), date(2015, 9, 16)), date(2015, 9, 26), date(2015, 9, 25),
            date(2015, 10, 1), DateRange.beginning_at(date(2015, 11, 1))))
    events = tuple(s.iterate(DateRange.inclusive(date(2015, 7, 11), date(2015, 9, 30))))
    expected = (
        DateDistribution.singular(date(2015, 7, 13)),
        DateDistribution.singular(date(2015, 7, 14)),
        # 2015/7/15 to 2015/9/16 excluded
        DateDistribution.singular(date(2015, 9, 17)),
        DateDistribution.singular(date(2015, 9, 18)),
        DateDistribution.singular(date(2015, 9, 21)),
        DateDistribution.singular(date(2015, 9, 22)),
        DateDistribution.singular(date(2015, 9, 23)),
        DateDistribution.singular(date(2015, 9, 24)),
        # 2015/9/25 excluded
        DateDistribution.singular(date(2015, 9, 28)),
        DateDistribution.singular(date(2015, 9, 29)),
        DateDistribution.singular(date(2015, 9, 30))
    )
    assert events == expected


def test_weekends_iterate_no_excludes() -> None:
    s = Weekends(DateRange.inclusive(date(2023, 1, 8), date(2023, 3, 13)))
    events = tuple(s.iterate(DateRange.inclusive(date(2022, 12, 13), date(2023, 2, 9))))
    expected = (
        DateDistribution.singular(date(2023, 1, 8)),
        DateDistribution.singular(date(2023, 1, 14)),
        DateDistribution.singular(date(2023, 1, 15)),
        DateDistribution.singular(date(2023, 1, 21)),
        DateDistribution.singular(date(2023, 1, 22)),
        DateDistribution.singular(date(2023, 1, 28)),
        DateDistribution.singular(date(2023, 1, 29)),
        DateDistribution.singular(date(2023, 2, 4)),
        DateDistribution.singular(date(2023, 2, 5))
    )
    assert events == expected

def test_weekends_iterate_excludes() -> None:
    s = Weekends(range=DateRange.inclusive(date(2023, 1, 1), date(2023, 4, 1)),
        exclude=(date(2023, 1, 3), DateRange.inclusive(date(2023, 1, 27), date(2023, 2, 11)), date(2023, 2, 26)))
    events = tuple(s.iterate(DateRange.inclusive(date(2023, 1, 1), date(2023, 3, 1))))
    expected = (
        DateDistribution.singular(date(2023, 1, 1)),
        DateDistribution.singular(date(2023, 1, 7)),
        DateDistribution.singular(date(2023, 1, 8)),
        DateDistribution.singular(date(2023, 1, 14)),
        DateDistribution.singular(date(2023, 1, 15)),
        DateDistribution.singular(date(2023, 1, 21)),
        DateDistribution.singular(date(2023, 1, 22)),
        # 2023/1/28 & 2023/1/29 excluded
        # 2023/2/4 & 2023/2/5 excluded
        # 2023/2/11 excluded
        DateDistribution.singular(date(2023, 2, 12)),
        DateDistribution.singular(date(2023, 2, 18)),
        DateDistribution.singular(date(2023, 2, 19)),
        DateDistribution.singular(date(2023, 2, 25))
        # 2023/2/26 excluded
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


def test_weekly_iterate_day() -> None:
    s = Weekly(
        day=4,
        range=DateRange.inclusive(date(2022, 12, 25), date(2023, 6, 12)),
        period=3,
        exclude=(date(2023, 2, 3), DateRange.inclusive(date(2023, 3, 1), date(2023, 4, 19)), date(2024, 3, 21)))
    actual = tuple(s.iterate(DateRange.inclusive(date(2022, 10, 4), date(2023, 5, 18))))
    expected = (
        # 2022/12/23 outside range
        DateDistribution.singular(date(2023, 1, 13)),
        # 2023/2/3  excluded
        DateDistribution.singular(date(2023, 2, 24)),
        # 2023/3/17 excluded
        # 2023/4/7  excluded
        DateDistribution.singular(date(2023, 4, 28))
    )
    assert actual == expected

def test_weekly_iterate_distribution() -> None:
    s = Weekly(
        day=DayOfWeekDistribution.from_probabilities({1: 0.4, 3: 0.2}),
        range=DateRange.inclusive(date(2023, 1, 5), date(2023, 3, 16)),
        period=2,
        exclude=(date(2023, 1, 19), DateRange.inclusive(date(2023, 1, 23), date(2023, 2, 5)),
            DateRange.half_open(date(2023, 2, 14), date(2023, 2, 14))))
    actual = tuple(s.iterate(DateRange.inclusive(date(2009, 10, 11), date(2023, 2, 28))))
    assert len(actual) == 4
    # 2023/1/3 outside range
    assert actual[0].approx_eq(DateDistribution((DiscreteOutcome(date(2023, 1, 5), 0.2, 0.6),)))
    assert actual[1].approx_eq(DateDistribution((DiscreteOutcome(date(2023, 1, 17), 0.4, 0.4),)))
    # 2023/1/19 excluded
    # 2023/1/31 & 2023/2/2 excluded
    assert actual[2].approx_eq(DateDistribution((
        DiscreteOutcome(date(2023, 2, 14), 0.4, 0.4), DiscreteOutcome(date(2023, 2, 16), 0.2, 0.4 + 0.2))))
    assert actual[3].approx_eq(DateDistribution((DiscreteOutcome(date(2023, 2, 28), 0.4, 0.4),)))
    # 2023/3/2 outside range

def test_weekly_iterate_schedule() -> None:
    s = Weekly(
        day=SimpleDayOfWeekSchedule((DayOfWeekDistribution.from_probabilities({1: 0.4, 3: 0.2}),)),
        range=DateRange.inclusive(date(2023, 1, 5), date(2023, 3, 16)),
        period=2,
        exclude=(date(2023, 1, 19), DateRange.inclusive(date(2023, 1, 23), date(2023, 2, 5)),
            DateRange.half_open(date(2023, 2, 14), date(2023, 2, 14))))
    actual = tuple(s.iterate(DateRange.inclusive(date(2009, 10, 11), date(2023, 2, 28))))
    assert len(actual) == 4
    # 2023/1/3 outside range
    assert actual[0].approx_eq(DateDistribution((DiscreteOutcome(date(2023, 1, 5), 0.2, 0.6),)))
    assert actual[1].approx_eq(DateDistribution((DiscreteOutcome(date(2023, 1, 17), 0.4, 0.4),)))
    # 2023/1/19 excluded
    # 2023/1/31 & 2023/2/2 excluded
    assert actual[2].approx_eq(DateDistribution((
        DiscreteOutcome(date(2023, 2, 14), 0.4, 0.4), DiscreteOutcome(date(2023, 2, 16), 0.2, 0.4 + 0.2))))
    assert actual[3].approx_eq(DateDistribution((DiscreteOutcome(date(2023, 2, 28), 0.4, 0.4),)))
    # 2023/3/2 outside range


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


def test_monthly_iterate_day() -> None:
    s = Monthly(
        day=31,
        range=DateRange.inclusive(date(2022, 1, 6), date(2028, 10, 2)),
        period=5,
        exclude=(DateRange.inclusive(date(2025, 4, 2), date(2025, 11, 1)), date(2026, 8, 31), date(2030, 2, 10)))
    actual = tuple(s.iterate(DateRange.inclusive(date(2021, 5, 8), date(2027, 1, 30))))
    expected = (
        DateDistribution.singular(date(2022, 1, 31)),
        # 2022/6/31 invalid
        # 2022/11/31 invalid
        # 2023/4/31 invalid
        # 2023/9/31 invalid
        # 2024/2/31 invalid
        DateDistribution.singular(date(2024, 7, 31)),
        DateDistribution.singular(date(2024, 12, 31)),
        # 2025/5/31 excluded
        # 2025/10/31 excluded
        DateDistribution.singular(date(2026, 3, 31)),
        # 2026/8/31 excluded
        # 2027/1/31 outside range
    )
    assert actual == expected

def test_monthly_iterate_schedule() -> None:
    ... # TODO

def test_monthly_iterate_leap_year() -> None:
    ... # TODO
