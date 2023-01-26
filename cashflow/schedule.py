from abc import ABC, abstractmethod
from calendar import FRIDAY, SATURDAY
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Callable

from .date_time import DateRange, DayOfMonthNumeral, DayOfWeekNumeral, Month, Week
from .probability import DiscreteDistribution


__all__ = [
    'Daily',
    'DateDistribution',
    'DayOfMonthDistribution',
    'DayOfMonthSchedule',
    'DayOfWeekDistribution',
    'DayOfWeekSchedule',
    'EventSchedule',
    'Monthly',
    'Never',
    'Once',
    'SimpleDayOfMonthSchedule',
    'SimpleDayOfWeekSchedule',
    'Weekdays',
    'Weekends',
    'Weekly'
]


DateDistribution = DiscreteDistribution[date]


class EventSchedule(ABC):
    """A rule which specifies when an event may occur.
        Each event is given as a distribution of dates on which it might occur."""

    @abstractmethod
    def iterate(self, date_range: DateRange, /) -> Iterable[DateDistribution]:
        """Iterates possible events in the schedule within the specified range of dates.

            Events are ordered roughly in chronological order, but their distributions may overlap.

            Each event is guaranteed to have a nonzero probability of occurring within `date_range`, however the events
            are not restricted to contain outcomes only within `date_range`."""

        raise NotImplementedError()

    # TODO? cache schedule iteration


class Never(EventSchedule):
    """Event never occurs."""

    def iterate(self, date_range: DateRange, /) -> tuple[()]:
        return ()


@dataclass(frozen=True, eq=False)
class Once(EventSchedule):
    """Event occurs exactly once, on a specified date or somewhere in a distribution of dates."""

    date: date | DateDistribution

    def iterate(self, date_range: DateRange, /) -> tuple[DateDistribution] | tuple[()]:
        match self.date:
            case DiscreteDistribution() as distribution \
                    if distribution.probability_in(
                        date_range.inclusive_lower_bound, date_range.exclusive_upper_bound) > 0:
                return (distribution,)
            case date() as d if d in date_range:
                return (DateDistribution.singular(d),)
            case _:
                return ()


@dataclass(frozen=True, eq=False)
class Daily(EventSchedule):
    """Event occurs on all days."""

    range: DateRange = DateRange.all()
    exclude: Collection[date | DateRange] = ()

    def iterate(self, date_range: DateRange, /) -> Iterable[DateDistribution]:
        return (DateDistribution.singular(occurrence) for occurrence in date_range & self.range
                if not _is_occurrence_excluded(occurrence, self.exclude))


@dataclass(frozen=True, eq=False)
class Weekdays(EventSchedule):
    """Event occurs on all Mondays to Fridays."""

    range: DateRange = DateRange.all()
    exclude: Collection[date | DateRange] = ()

    def iterate(self, date_range: DateRange, /) -> Iterable[DateDistribution]:
        return (DateDistribution.singular(occurrence) for occurrence in date_range & self.range
                if occurrence.weekday() <= FRIDAY and not _is_occurrence_excluded(occurrence, self.exclude))


@dataclass(frozen=True, eq=False)
class Weekends(EventSchedule):
    """Event occurs on all Saturdays and Sundays."""

    range: DateRange = DateRange.all()
    exclude: Collection[date | DateRange] = ()

    def iterate(self, date_range: DateRange, /) -> Iterable[DateDistribution]:
        return (DateDistribution.singular(occurrence) for occurrence in date_range & self.range
                if occurrence.weekday() >= SATURDAY and not _is_occurrence_excluded(occurrence, self.exclude))


DayOfWeekDistribution = DiscreteDistribution[DayOfWeekNumeral]


class DayOfWeekSchedule(ABC):
    """A rule which specifies when an event may occur within a particular week.
        Each event is given as a distribution of days on which it might occur."""

    @abstractmethod
    def iterate(self, week: Week, /) -> Iterable[DayOfWeekDistribution]:
        """Iterates possible occurrences in the schedule within the specified week.

            Occurrences are ordered roughly in chronological order, but their distributions may overlap."""

        raise NotImplementedError()


@dataclass(frozen=True, eq=False)
class SimpleDayOfWeekSchedule(DayOfWeekSchedule):
    """A basic schedule where each week has the same distribution of occurrences."""

    distributions: Sequence[DayOfWeekDistribution]

    def iterate(self, week: Week, /) -> Iterable[DayOfWeekDistribution]:
        return (distribution for distribution in self.distributions if distribution.has_possible_outcomes)


@dataclass(frozen=True, eq=False)
class Weekly(EventSchedule):
    """Event occurs on specified days of week every `period` number of weeks."""

    day: DayOfWeekNumeral | DayOfWeekDistribution | DayOfWeekSchedule
    range: DateRange = DateRange.all()
    period: int = 1
    exclude: Collection[date | DateRange] = ()

    def __post_init__(self) -> None:
        if isinstance(self.day, int) and not 0 <= self.day <= 6:
            raise ValueError('day must be in the range [0, 6]')
        if self.period < 1:
            raise ValueError('period must be >= 1')
        if self.period != 1 and not self.range.has_proper_lower_bound:
            raise ValueError('range must have a lower bound if period is not 1')

    def iterate(self, date_range: DateRange, /) -> Iterable[DateDistribution]:
        # Methodology is to iterate over weeks, then within each week, iterate the specified days.

        date_range &= self.range
        if date_range:
            day_schedule = self._day_schedule

            start_week = Week.of(self.range.inclusive_lower_bound)
            week = Week.of(date_range.first_day)
            last_week = Week.of(date_range.inclusive_upper_bound)

            while week <= last_week:
                period_diff = (week - start_week) % self.period
                if period_diff == 0:
                    for day_distribution in day_schedule.iterate(week):
                        date_distribution = day_distribution.map_values(week.day)
                        # Note that the probabilities of other occurences are not affected by the excluded occurences.
                        date_distribution = date_distribution.subset(_excluded_occurrence_filter(self.exclude))
                        if date_distribution.probability_in(
                                date_range.inclusive_lower_bound, date_range.exclusive_upper_bound) > 0:
                            yield date_distribution
                week += self.period - period_diff

    @property
    def _day_schedule(self) -> DayOfWeekSchedule:
        match self.day:
            case DayOfWeekSchedule() as schedule:
                return schedule
            case DiscreteDistribution() as distribution:
                return SimpleDayOfWeekSchedule((distribution,))
            case int(day):
                return SimpleDayOfWeekSchedule((DayOfWeekDistribution.singular(day),))
            case _:
                raise TypeError('day')


DayOfMonthDistribution = DiscreteDistribution[DayOfMonthNumeral]


class DayOfMonthSchedule(ABC):
    """A rule which specifies when an event may occur within a particular month.
        Each event is given as a distribution of days on which it might occur."""

    @abstractmethod
    def iterate(self, month: Month, /) -> Iterable[DayOfMonthDistribution]:
        """Iterates possible occurrences in the schedule within the specified month.

            Occurrences are ordered roughly in chronological order, but their distributions may overlap.

            Distributions will never contain days which are invalid for the specified month (e.g. 30 for February)."""

        raise NotImplementedError()


@dataclass(frozen=True, eq=False)
class SimpleDayOfMonthSchedule(DayOfMonthSchedule):
    """A basic schedule where each month has the same distribution of occurrences.
        However, for a given month, days in the distribution which would form invalid dates (e.g. February 30th) are
        removed."""

    distributions: Sequence[DayOfMonthDistribution]

    def iterate(self, month: Month, /) -> Iterable[DayOfMonthDistribution]:
        # Note that the probabilities of other days are not affected by removing the invalid dates.
        distributions = (distribution.subset(month.has_day) for distribution in self.distributions)
        return (distribution for distribution in distributions if distribution.has_possible_outcomes)


@dataclass(frozen=True, eq=False)
class Monthly(EventSchedule):
    """Event occurs on specified days of month every `period` number of months.

        Occurences on days that form invalid dates (e.g. February 30th) are excluded."""

    day: DayOfMonthNumeral | DayOfMonthSchedule
    range: DateRange = DateRange.all()
    period: int = 1
    exclude: Collection[date | DateRange] = ()

    def __post_init__(self) -> None:
        if isinstance(self.day, int) and not 1 <= self.day <= 31:
            raise ValueError('day must be in the range [1, 31]')
        if self.period < 1:
            raise ValueError('period must be >= 1')
        if self.period != 1 and not self.range.has_proper_lower_bound:
            raise ValueError('range must have a lower bound if period is not 1')

    def iterate(self, date_range: DateRange, /) -> Iterable[DateDistribution]:
        # Methodology is to iterate over possible months, then within each month, iterate the specified days.

        date_range &= self.range
        if date_range:
            day_schedule = self._day_schedule

            start_month = Month.of(self.range.inclusive_lower_bound)
            month = Month.of(date_range.first_day)
            last_month = Month.of(date_range.inclusive_upper_bound)

            while month <= last_month:
                period_diff = (month - start_month) % self.period
                if period_diff == 0:
                    for day_distribution in day_schedule.iterate(month):
                        date_distribution = day_distribution.map_values(month.day)
                        # Note that the probabilities of other occurences are not affected by the excluded occurences.
                        date_distribution = date_distribution.subset(_excluded_occurrence_filter(self.exclude))
                        if date_distribution.probability_in(
                                date_range.inclusive_lower_bound, date_range.exclusive_upper_bound) > 0:
                            yield date_distribution
                month += self.period - period_diff

    @property
    def _day_schedule(self) -> DayOfMonthSchedule:
        match self.day:
            case DayOfMonthSchedule() as schedule:
                return schedule
            case int(day):
                return SimpleDayOfMonthSchedule((DayOfMonthDistribution.singular(day),))
            case _:
                raise TypeError('day')


def _is_occurrence_excluded(occurrence: date, excluded: Collection[date | DateRange]) -> bool:
    for exclude in excluded:
        match exclude:
            case date() as d if occurrence == d:
                return True
            case DateRange() as date_range if occurrence in date_range:
                return True
            case _:
                pass
    return False


def _excluded_occurrence_filter(exclude: Collection[date | DateRange], /) -> Callable[[date], bool]:
    def inner(occurrence: date):
        return not _is_occurrence_excluded(occurrence, exclude)

    return inner
