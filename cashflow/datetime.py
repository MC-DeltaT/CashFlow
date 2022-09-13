"""Date and time-related utilities."""


from calendar import MONDAY
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterator, Literal, cast

from dateutil.relativedelta import relativedelta


__all__ = [
    'DateRange',
    'DayOfMonthNumeral',
    'DayOfWeekNumeral',
    'Month',
    'MonthNumeral',
    'Week'
]


DayOfWeekNumeral = Literal[0, 1, 2, 3, 4, 5, 6]
DayOfMonthNumeral = Literal[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

MonthNumeral = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


@dataclass(frozen=True, order=True)
class Month:
    year: int
    month: MonthNumeral

    @classmethod
    def of(cls, d: date, /):
        """Takes the month that a date is within."""

        return cls(d.year, cast(MonthNumeral, d.month))

    @property
    def date_range(self) -> 'DateRange':
        """Gets the range of dates this month spans."""

        first_day = self.day(1)
        return DateRange.half_open(first_day, first_day + relativedelta(months=1))

    def day(self, day: DayOfMonthNumeral, /) -> date:
        """Creates a date within this month.

            :except ValueError: If `day` is not a valid day within this month (e.g. 30th of February)."""

        return date(self.year, self.month, day)

    def __contains__(self, d: date | datetime, /) -> bool:
        """Checks if a date or datetime is within this month.
            If the value is a datetime, the time part is simply ignored."""

        if isinstance(d, datetime):
            d = d.date()
        return d.year == self.year and d.month == self.month

    def __add__(self, months: int):
        """Adds a number of months."""

        return self.of(self.day(1) + relativedelta(months=months))

    def __sub__(self, other: 'Month') -> int:
        """Finds the number of months between two months."""

        return (self.year - other.year) * 12 + self.month - other.month


@dataclass(frozen=True, order=True)
class Week:
    """A seven-day week starting Monday."""

    start: date     # Always a Monday

    def __post_init__(self) -> None:
        if self.start.weekday() != MONDAY:
            raise ValueError('start must be a Monday')

    @classmethod
    def of(cls, d: date, /):
        """Takes the week that a date is within."""

        return cls(d + relativedelta(weekday=MONDAY))

    @property
    def date_range(self) -> 'DateRange':
        """Gets the range of dates this week spans."""

        return DateRange.half_open(self.start, self.start + relativedelta(weeks=1))

    def day(self, day: DayOfWeekNumeral) -> date:
        """Creates a date within this week."""

        return self.start + relativedelta(weekday=day)

    def __contains__(self, d: date | datetime, /) -> bool:
        """Checks if a date or datetime is within this week.
            If the value is a datetime, the time part is simply ignored."""

        if isinstance(d, datetime):
            d = d.date()
        return d in self.date_range

    def __add__(self, weeks: int):
        """Adds a number of weeks."""

        return self.of(self.start + relativedelta(weeks=weeks))

    def __sub__(self, other: 'Week') -> int:
        """Finds the number of weeks between two weeks."""

        return (self.start - other.start).days // 7


@dataclass(frozen=True, kw_only=True)
class DateRange:
    """A contiguous sequence of dates."""

    inclusive_lower_bound: date
    exclusive_upper_bound: date

    def __post_init__(self) -> None:
        if self.inclusive_lower_bound > self.exclusive_upper_bound:
            raise ValueError('inclusive_lower_bound must be <= exclusive_upper_bound')

    @classmethod
    def inclusive(cls, inclusive_lower_bound: date, inclusive_upper_bound: date):
        """Creates a range from an inclusive lower bound and inclusive upper bound."""

        return cls(
            inclusive_lower_bound=inclusive_lower_bound,
            exclusive_upper_bound=inclusive_upper_bound + timedelta(days=1))

    @classmethod
    def half_open(cls, inclusive_lower_bound: date, exclusive_upper_bound: date):
        """Creates a range from an inclusive lower bound and exclusive upper bound."""

        return cls(inclusive_lower_bound=inclusive_lower_bound, exclusive_upper_bound=exclusive_upper_bound)

    @classmethod
    def singular(cls, d: date, /):
        """Creates a range containing exactly one date."""

        return cls.inclusive(d, d)

    @classmethod
    def around(cls, centre: date, days_radius: int):
        """Creates a range centred on `centre` and containing `days_radius` days on either side."""

        return cls.inclusive(centre + timedelta(days=-days_radius), centre + timedelta(days=days_radius))

    @classmethod
    def beginning_at(cls, inclusive_lower_bound: date):
        """Creates a range containing all representable dates beginning from an inclusive lower bound (i.e no upper
            bound)."""

        return cls.half_open(inclusive_lower_bound, date.max)

    @classmethod
    def before(cls, exclusive_upper_bound: date):
        """Creates a range containing all representable dates less than an exclusive upper bound (i.e. no lower
            bound)."""

        return cls.half_open(date.min, exclusive_upper_bound)

    @classmethod
    def up_to(cls, inclusive_upper_bound: date):
        """Creates a range containing all representable dates up to an inclusive upper bound (i.e. no lower bound)."""

        return cls.inclusive(date.min, inclusive_upper_bound)

    @classmethod
    def all(cls):
        """Creates a range containing all representable dates (i.e. no lower or upper bounds)."""

        return cls.half_open(date.min, date.max)

    @property
    def has_proper_lower_bound(self) -> bool:
        """Checks if the range has a lower bound that is not `date.min`."""

        return self.inclusive_lower_bound != date.min

    @property
    def has_proper_upper_bound(self) -> bool:
        """Checks if the range has an upper bound that is not `date.max`."""

        return self.exclusive_upper_bound != date.max

    @property
    def days(self) -> int:
        """The number of days contained in the range.

            :except ValueError: If the range is missing a lower or upper bound."""

        if self.has_proper_lower_bound and self.has_proper_upper_bound:
            return (self.exclusive_upper_bound - self.inclusive_lower_bound).days
        else:
            raise ValueError('Range is not fully bounded')

    @property
    def is_empty(self) -> bool:
        """Checks if the range contains zero dates."""

        return self.inclusive_lower_bound == self.exclusive_upper_bound

    @property
    def first_day(self) -> date:
        """Gets the first day within the range.

            :except ValueError: If the range is empty or has no lower bound."""

        if self.is_empty or not self.has_proper_lower_bound:
            raise ValueError('Empty range does not have a first day')
        else:
            return self.inclusive_lower_bound

    @property
    def last_day(self) -> date:
        """Gets the last day within the range.

            :except ValueError: If the range is empty or has no upper bound."""

        if self.is_empty or not self.has_proper_upper_bound:
            raise ValueError('Empty range does not have a last day')
        else:
            return self.exclusive_upper_bound + timedelta(days=-1)

    def __iter__(self) -> Iterator[date]:
        """Iterates all dates within the range, in chronological order."""

        d = self.inclusive_lower_bound
        while d < self.exclusive_upper_bound:
            yield d
            d += timedelta(days=1)

    def __len__(self) -> int:
        return self.days

    def __contains__(self, d: date | datetime, /) -> bool:
        """Checks if a date or datetime is contained within the range.
            If the value is a datetime, the time part is simply ignored."""

        if isinstance(d, datetime):
            d = d.date()
        return self.inclusive_lower_bound <= d < self.exclusive_upper_bound

    def __and__(self, other: 'DateRange', /):
        """Intersection of two ranges."""

        lower_bound = max(self.inclusive_lower_bound, other.inclusive_lower_bound)
        upper_bound = min(self.exclusive_upper_bound, other.exclusive_upper_bound)
        # Clamp the lower bound so it doesn't exceed the upper bound.
        lower_bound = min(lower_bound, upper_bound)
        return self.half_open(lower_bound, upper_bound)
