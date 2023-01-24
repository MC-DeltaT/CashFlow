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

def test_month_has_day() -> None:
    m = Month(2020, 2)
    assert m.has_day(1)
    assert m.has_day(10)
    assert m.has_day(28)
    assert m.has_day(29)
    assert not m.has_day(30)
    assert not m.has_day(31)
    assert not m.has_day(32)

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


def test_date_range_construct_invalid() -> None:
    with raises(ValueError):
        DateRange(inclusive_lower_bound=date(2022, 12, 20), exclusive_upper_bound=date(2022, 12, 19))

def test_date_range_inclusive() -> None:
    d = DateRange.inclusive(date(2022, 8, 7), date(2026, 3, 31))
    assert d.inclusive_lower_bound == date(2022, 8, 7)
    assert d.exclusive_upper_bound == date(2026, 4, 1)

def test_date_range_half_open() -> None:
    d = DateRange.half_open(date(2000, 12, 4), date(2004, 10, 15))
    assert d.inclusive_lower_bound == date(2000, 12, 4)
    assert d.exclusive_upper_bound == date(2004, 10, 15)

def test_date_range_singular() -> None:
    d = DateRange.singular(date(2020, 2, 20))
    assert d.inclusive_lower_bound == date(2020, 2, 20)
    assert d.exclusive_upper_bound == date(2020, 2, 21)

def test_date_range_around_nonzero_radius() -> None:
    d = DateRange.around(date(2022, 5, 28), 7)
    assert d.inclusive_lower_bound == date(2022, 5, 21)
    assert d.exclusive_upper_bound == date(2022, 6, 5)

def test_date_range_around_zero_radius() -> None:
    d = DateRange.around(date(2023, 1, 8), 0)
    assert d.inclusive_lower_bound == date(2023, 1, 8)
    assert d.exclusive_upper_bound == date(2023, 1, 9)

def test_date_range_around_invalid() -> None:
    with raises(ValueError):
        DateRange.around(date(2023, 1, 5), -1)

def test_date_range_beginning_at() -> None:
    d = DateRange.beginning_at(date(1980, 1, 5))
    assert d.inclusive_lower_bound
    assert not d.has_proper_upper_bound

def test_date_range_before() -> None:
    d = DateRange.before(date(2006, 4, 2))
    assert not d.has_proper_lower_bound
    assert d.exclusive_upper_bound == date(2006, 4, 2)

def test_date_range_up_to() -> None:
    d = DateRange.up_to(date(2007, 8, 20))
    assert not d.has_proper_lower_bound
    assert d.exclusive_upper_bound == date(2007, 8, 21)

def test_date_range_all() -> None:
    d = DateRange.all()
    assert not d.has_proper_lower_bound
    assert not d.has_proper_upper_bound

def test_date_range_empty() -> None:
    d = DateRange.empty()
    assert d.is_empty
    # These should probably be true, else things may break.
    assert d.has_proper_lower_bound
    assert d.has_proper_upper_bound

def test_date_range_days_valid() -> None:
    d = DateRange.inclusive(date(1999, 11, 14), date(2000, 1, 3))
    assert d.days == 17 + 31 + 3

def test_date_range_day_invalid() -> None:
    d1 = DateRange.beginning_at(date(2022, 12, 25))
    with raises(ValueError):
        d1.days

    d2 = DateRange.up_to(date(2022, 12, 25))
    with raises(ValueError):
        d2.days

def test_date_range_empty_true() -> None:
    d = DateRange.half_open(date(2022, 6, 7), date(2022, 6, 7))
    assert d.is_empty

def test_date_range_empty_false() -> None:
    d = DateRange.singular(date(2022, 8, 9))
    assert not d.is_empty

def test_date_range_first_day_valid() -> None:
    d = DateRange.inclusive(date(2022, 11, 1), date(2022, 12, 19))
    assert d.first_day == date(2022, 11, 1)

def test_date_range_first_day_invalid() -> None:
    d = DateRange.up_to(date(2022, 12, 18))
    with raises(ValueError):
        d.first_day

def test_date_range_last_day_valid() -> None:
    d = DateRange.inclusive(date(2022, 11, 1), date(2022, 12, 19))
    assert d.last_day == date(2022, 12, 19)

def test_date_range_last_day_invalid() -> None:
    d = DateRange.beginning_at(date(2022, 12, 18))
    with raises(ValueError):
        d.last_day

def test_date_range_inclusive_upper_bound() -> None:
    d = DateRange.inclusive(date(2012, 12, 12), date(2013, 11, 15))
    assert d.inclusive_upper_bound == date(2013, 11, 15)

def test_date_range_iter() -> None:
    d = DateRange.inclusive(date(2002, 12, 29), date(2003, 1, 5))
    expected = [
        date(2002, 12, 29),
        date(2002, 12, 30),
        date(2002, 12, 31),
        date(2003, 1, 1),
        date(2003, 1, 2),
        date(2003, 1, 3),
        date(2003, 1, 4),
        date(2003, 1, 5)
    ]
    assert list(d) == expected

def test_date_range_len() -> None:
    d = DateRange.inclusive(date(2002, 12, 29), date(2003, 1, 5))
    assert len(d) == 8

def test_date_range_contains_bounded() -> None:
    d = DateRange.inclusive(date(2002, 12, 29), date(2003, 1, 5))
    assert date(2002, 12, 29) in d
    assert date(2002, 12, 30) in d
    assert date(2003, 1, 5) in d
    assert date(2002, 12, 28) not in d
    assert date(2003, 1, 6) not in d

def test_date_range_contains_unbounded_upper() -> None:
    d = DateRange.beginning_at(date(2022, 12, 27))
    assert date(2022, 12, 27) in d
    assert date(2022, 12, 31) in d
    assert date(2100, 1, 1) in d
    assert date(2022, 12, 26) not in d
    assert date(2000, 1, 1) not in d

def test_date_range_contains_unbounded_lower() -> None:
    d = DateRange.up_to(date(2022, 12, 27))
    assert date(2022, 12, 27) in d
    assert date(2022, 6, 8) in d
    assert date(1900, 5, 18) in d
    assert date(2022, 12, 28) not in d
    assert date(2050, 5, 1) not in d

def test_date_range_contains_unbounded() -> None:
    d = DateRange.all()
    assert date(2022, 12, 27) in d
    assert date(2022, 6, 8) in d
    assert date(1900, 5, 18) in d
    assert date(2022, 12, 28) in d
    assert date(2050, 5, 1) in d
    assert date(2022, 12, 31) in d
    assert date(2100, 1, 1) in d
    assert date(2022, 12, 26) in d
    assert date(2000, 1, 1) in d
    assert date(2002, 12, 29) in d
    assert date(2002, 12, 30) in d
    assert date(2003, 1, 5) in d
    assert date(2002, 12, 28) in d
    assert date(2003, 1, 6) in d

def test_date_range_and_overlapping1() -> None:
    d1 = DateRange.inclusive(date(2000, 5, 1), date(2000, 6, 1))
    d2 = DateRange.inclusive(date(2000, 5, 20), date(2000, 7, 3))
    expected = DateRange.inclusive(date(2000, 5, 20), date(2000, 6, 1))
    assert d1 & d2 == expected

def test_date_range_and_overlapping2() -> None:
    d1 = DateRange.inclusive(date(2000, 5, 20), date(2000, 7, 3))
    d2 = DateRange.inclusive(date(2000, 5, 1), date(2000, 6, 1))
    expected = DateRange.inclusive(date(2000, 5, 20), date(2000, 6, 1))
    assert d1 & d2 == expected

def test_date_range_and_overlapping3() -> None:
    d1 = DateRange.inclusive(date(2016, 4, 1), date(2017, 9, 16))
    d2 = DateRange.inclusive(date(2016, 4, 1), date(2017, 9, 16))
    expected = DateRange.inclusive(date(2016, 4, 1), date(2017, 9, 16))
    assert d1 & d2 == expected

def test_date_range_and_disjoint() -> None:
    d1 = DateRange.half_open(date(2000, 1, 1), date(2000, 2, 1))
    d2 = DateRange.half_open(date(2000, 2, 1), date(2000, 3, 1))
    expected = DateRange(inclusive_lower_bound=date(2000, 2, 1), exclusive_upper_bound=date(2000, 2, 1))
    assert d1 & d2 == expected
    assert d2 & d1 == expected

def test_date_range_eq() -> None:
    d1 = DateRange.half_open(date(2022, 12, 24), date(2024, 4, 30))
    d2 = DateRange.inclusive(date(2022, 12, 24), date(2024, 4, 29))
    assert d1 == d2

def test_date_range_neq() -> None:
    d1 = DateRange.inclusive(date(2022, 12, 24), date(2024, 4, 30))
    d2 = DateRange.inclusive(date(2022, 12, 24), date(2024, 4, 29))
    assert d1 != d2
