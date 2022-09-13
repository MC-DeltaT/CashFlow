"""Event schedules."""


from abc import ABC, abstractmethod
from calendar import FRIDAY, SATURDAY
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Collection, Iterable, Sequence

from .datetime import DateRange, DayOfMonthNumeral, DayOfWeekNumeral, Month, Week
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
        """Iterates occurrences in the schedule within the specified range of dates.
            Occurences are ordered roughly in chronological order, but their distributions may overlap."""

        raise NotImplementedError()


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
            case DiscreteDistribution() as distribution if distribution.could_occur_in(date_range):
                return (distribution,)
            case date() as d if d in date_range:
                return (DateDistribution.singular(d),)
            case _:
                return ()


@dataclass(frozen=True, eq=False)
class Daily(EventSchedule):
    """Event occurs on all days."""

    range: DateRange = DateRange.all()
    exceptions: Collection[date | DateRange] = ()

    def iterate(self, date_range: DateRange, /) -> Iterable[DateDistribution]:
        return (DateDistribution.singular(occurrence) for occurrence in date_range & self.range
                if not self._is_excepted(occurrence))

    def _is_excepted(self, occurrence: date) -> bool:
        for exception in self.exceptions:
            if occurrence == exception or (isinstance(exception, DateRange) and occurrence in exception):
                return True
        return False


@dataclass(frozen=True, eq=False)
class Weekdays(EventSchedule):
    """Event occurs on all Mondays to Fridays."""

    range: DateRange = DateRange.all()
    exceptions: Collection[date | DateRange] = ()

    def iterate(self, date_range: DateRange, /) -> Iterable[DateDistribution]:
        return (DateDistribution.singular(occurrence) for occurrence in date_range & self.range
                if occurrence.weekday() <= FRIDAY and not self._is_excepted(occurrence))

    def _is_excepted(self, occurrence: date) -> bool:
        for exception in self.exceptions:
            if occurrence == exception or (isinstance(exception, DateRange) and occurrence in exception):
                return True
        return False


@dataclass(frozen=True, eq=False)
class Weekends(EventSchedule):
    """Event occurs on all Saturdays and Sundays."""

    range: DateRange = DateRange.all()
    exceptions: Collection[date | DateRange] = ()

    def iterate(self, date_range: DateRange, /) -> Iterable[DateDistribution]:
        return (DateDistribution.singular(occurrence) for occurrence in date_range & self.range
                if occurrence.weekday() >= SATURDAY and not self._is_excepted(occurrence))

    def _is_excepted(self, occurrence: date) -> bool:
        for exception in self.exceptions:
            if occurrence == exception or (isinstance(exception, DateRange) and occurrence in exception):
                return True
        return False


DayOfWeekDistribution = DiscreteDistribution[DayOfWeekNumeral]


class DayOfWeekSchedule(ABC):
    """A rule which specifies when an event may occur within a particular week.
        Each event is given as a distribution of days on which it might occur."""

    @abstractmethod
    def iterate(self, week: Week, /) -> Iterable[DayOfWeekDistribution]:
        """Iterates occurrences in the schedule within the specified week.
            Occurrences are ordered roughly in chronological order, but their distributions may overlap."""

        raise NotImplementedError()


@dataclass(frozen=True, eq=False)
class SimpleDayOfWeekSchedule(DayOfWeekSchedule):
    """A basic schedule where each week has the same distribution of occurrences."""

    distributions: Sequence[DayOfWeekDistribution]

    def iterate(self, week: Week, /) -> Iterable[DayOfWeekDistribution]:
        return self.distributions


@dataclass(frozen=True, eq=False)
class Weekly(EventSchedule):
    """Event occurs on specified days of week every `period` number of weeks."""

    day: DayOfWeekNumeral | DayOfWeekDistribution | DayOfWeekSchedule
    range: DateRange = DateRange.all()
    period: int = 1
    exceptions: Collection[date | DateRange] = ()

    def __post_init__(self) -> None:
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
            last_week = Week.of(date_range.exclusive_upper_bound + timedelta(days=-1))

            while week <= last_week:
                period_diff = (week - start_week) % self.period
                if period_diff == 0:
                    for day_distribution in day_schedule.iterate(week):
                        date_distribution = day_distribution.map_values(week.day)
                        # Don't filter out occurrences that aren't in the requested date range, so that the
                        # probabilities of the other occurrences are not affected. (Occurrences outside the range can
                        # still occur, they just won't be observed.)
                        date_distribution = date_distribution.drop(self._is_excepted)
                        if date_distribution.could_occur_in(date_range):
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

    def _is_excepted(self, occurrence: date) -> bool:
        for exception in self.exceptions:
            if occurrence == exception or (isinstance(exception, DateRange) and occurrence in exception):
                return True
        return False


DayOfMonthDistribution = DiscreteDistribution[DayOfMonthNumeral]


class DayOfMonthSchedule(ABC):
    """A rule which specifies when an event may occur within a particular month.
        Each event is given as a distribution of days on which it might occur."""

    @abstractmethod
    def iterate(self, month: Month, /) -> Iterable[DayOfMonthDistribution]:
        """Iterates occurrences in the schedule within the specified month.
            Occurrences are ordered roughly in chronological order, but their distributions may overlap."""

        raise NotImplementedError()


@dataclass(frozen=True, eq=False)
class SimpleDayOfMonthSchedule(DayOfMonthSchedule):
    """A basic schedule where each month has the same distribution of occurrences."""

    distributions: Sequence[DayOfMonthDistribution]

    def iterate(self, month: Month, /) -> Iterable[DayOfMonthDistribution]:
        return self.distributions


@dataclass(frozen=True, eq=False)
class Monthly(EventSchedule):
    """Event occurs on specified days of month every `period` number of months."""

    day: DayOfMonthNumeral | DayOfMonthDistribution | DayOfMonthSchedule
    range: DateRange = DateRange.all()
    period: int = 1
    exceptions: Collection[date | DateRange] = ()

    def __post_init__(self) -> None:
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
            last_month = Month.of(date_range.exclusive_upper_bound + timedelta(days=-1))

            while month <= last_month:
                period_diff = (month - start_month) % self.period
                if period_diff == 0:
                    for day_distribution in day_schedule.iterate(month):
                        date_distribution = day_distribution.map_values(month.day)
                        # Don't filter out occurrences that aren't in the requested date range, so that the
                        # probabilities of the other occurrences are not affected. (Occurrences outside the range can
                        # still occur, they just won't be observed.)
                        date_distribution = date_distribution.drop(self._is_excepted)
                        if date_distribution.could_occur_in(date_range):
                            yield date_distribution
                month += self.period - period_diff

    @property
    def _day_schedule(self) -> DayOfMonthSchedule:
        match self.day:
            case DayOfMonthSchedule() as schedule:
                return schedule
            case DiscreteDistribution() as distribution:
                return SimpleDayOfMonthSchedule((distribution,))
            case int(day):
                return SimpleDayOfMonthSchedule((DayOfMonthDistribution.singular(day),))
            case _:
                raise TypeError('day')

    def _is_excepted(self, occurrence: date) -> bool:
        for exception in self.exceptions:
            if occurrence == exception or (isinstance(exception, DateRange) and occurrence in exception):
                return True
        return False
