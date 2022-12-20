from datetime import date
from pytest import raises

from cashflow.date_time import DateRange, Month, Week


def test_month_construct_invalid_month() -> None:
    with raises(ValueError):
        Month(2022, -1)
    with raises(ValueError):
        Month(2022, 13)

def test_month_of_first_day() -> None:
    assert Month.of(date(2022, 12, 1)) == Month(2022, 12)

def test_month_of_not_first_day() -> None:
    assert Month.of(date(2022, 12, 15)) == Month(2022, 12)

def test_month_date_range() -> None:
    m = Month(2022, 12)
    expected = DateRange(inclusive_lower_bound=date(2022, 12, 1),exclusive_upper_bound=date(2023, 1, 1))
    assert m.date_range == expected

def test_month_day_valid() -> None:
    assert Month(2022, 7).day(12) == date(2022, 7, 12)

def test_month_day_invalid_day() -> None:
    m = Month(2022, 5)
    with raises(ValueError):
        m.day(-3)

def test_month_day_invalid_date() -> None:
    m = Month(2022, 2)
    with raises(ValueError):
        m.day(29)

def test_month_contains_true() -> None:
    assert date(2022, 1, 4) in Month(2022, 1)

def test_month_contains_false() -> None:
    assert date(2022, 3, 4) not in Month(2022, 1)

def test_month_add_positive_same_year() -> None:
    assert Month(2010, 6) + 4 == Month(2010, 10)

def test_month_add_positive_different_year() -> None:
    assert Month(2010, 6) + 26 == Month(2012, 8)

def test_month_add_negative_same_year() -> None:
    assert Month(2000, 4) + -3 == Month(2000, 1)

def test_month_add_negative_different_year() -> None:
    assert Month(2000, 6) + -20 == Month(1998, 10)

def test_month_sub_same_year_positive() -> None:
    assert Month(2017, 8) - Month(2017, 3) == 5

def test_month_sub_different_year_positive() -> None:
    assert Month(2018, 4) - Month(2014, 3) == 49

def test_month_sub_same_year_negative() -> None:
    assert Month(2021, 3) - Month(2021, 12) == -9

def test_month_sub_different_year_negative() -> None:
    assert Month(2021, 11) - Month(2022, 12) == -13

def test_month_eq() -> None:
    assert Month(2002, 12) == Month(2002, 12)

def test_month_neq() -> None:
    assert Month(2002, 12) != Month(2002, 4)


def test_week_start_not_monday() -> None:
    with raises(ValueError):
        Week(date(2022, 12, 18))

def test_week_of_monday() -> None:
    assert Week.of(date(2022, 12, 19)) == Week(date(2022, 12, 19))

def test_week_of_not_monday() -> None:
    assert Week.of(date(2022, 12, 18)) == Week(date(2022, 12, 12))

def test_week_date_range() -> None:
    w = Week(date(2022, 12, 12))
    expected = DateRange(inclusive_lower_bound=date(2022, 12, 12), exclusive_upper_bound=date(2022, 12, 19))
    assert w.date_range == expected

def test_week_day_same_month() -> None:
    w = Week(date(2022, 12, 19))
    assert w.day(3) == date(2022, 12, 22)

def test_week_day_next_month() -> None:
    w = Week(date(2022, 11, 28))
    assert w.day(4) == date(2022, 12, 2)

def test_week_day_next_year() -> None:
    w = Week(date(2021, 12, 27))
    assert w.day(6) == date(2022, 1, 2)

def test_week_day_invalid() -> None:
    w = Week(date(2022, 12, 19))
    with raises(Exception):
        w.day(-1)
    with raises(Exception):
        w.day(7)

def test_week_contains_true() -> None:
    assert date(2022, 12, 25) in Week(date(2022, 12, 19))

def test_week_contains_false() -> None:
    assert date(2022, 12, 18) not in Week(date(2022, 12, 19))

def test_week_add_positive_same_month() -> None:
    assert Week(date(2022, 12, 12)) + 1 == Week(date(2022, 12, 19))

def test_week_add_positive_different_month() -> None:
    assert Week(date(2022, 12, 19)) + 5 == Week(date(2023, 1, 23))

def test_week_add_negative_same_month() -> None:
    assert Week(date(2022, 12, 19)) + -1 == Week(date(2022, 12, 12))

def test_week_add_negative_different_month() -> None:
    assert Week(date(2022, 12, 19)) + -3 == Week(date(2022, 11, 28))

def test_week_sub_same_month_positive() -> None:
    assert Week(date(2022, 1, 24)) - Week(date(2022, 1, 10)) == 2

def test_week_sub_different_month_positive() -> None:
    assert Week(date(2023, 1, 23)) - Week(date(2022, 12, 19)) == 5

def test_week_sub_same_month_negative() -> None:
    assert Week(date(2022, 12, 12)) - Week(date(2022, 12, 26)) == -2

def test_week_sub_different_month_negative() -> None:
    assert Week(date(2022, 11, 28)) - Week(date(2022, 12, 19)) == -3

def test_week_eq() -> None:
    assert Week(date(2022, 12, 19)) == Week(date(2022, 12, 19))

def test_week_neq() -> None:
    assert Week(date(2022, 12, 19)) != Week(date(2022, 12, 5))


# TODO: test DateRange
