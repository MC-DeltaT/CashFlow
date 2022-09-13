"""Core, necessary functionality."""


from abc import ABC
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from heapq import merge
from itertools import groupby
from typing import Iterable, Mapping

from .datetime import DateRange
from .probability import (
    DEFAULT_CERTAINTY_TOLERANCE, DiscreteDistribution, DiscreteOutcome, FloatDistribution, effectively_certain)
from .schedule import EventSchedule


__all__ = [
    'Account',
    'AccountBalance',
    'AccountBalanceEvent',
    'AccountBalancesRecord',
    'CashDelta',
    'CashEndpoint',
    'CashFlow',
    'CashFlowEvent',
    'CashFlowOccurrence',
    'CashSink',
    'CashSource',
    'generate_account_balance_events',
    'iterate_cash_flow_occurrences',
    'ScheduledCashFlow',
    'simulate_account_balance_events'
]


@dataclass(frozen=True)
class CashEndpoint(ABC):
    """Somewhere that funds come from or go to."""

    label: str
    tags: frozenset[str] = frozenset()  # Arbitrary tags that can be used for custom categorisation or analysis.


@dataclass(frozen=True)
class CashSource(CashEndpoint, ABC):
    """Somewhere that funds ccan be sourced from."""


@dataclass(frozen=True)
class CashSink(CashEndpoint, ABC):
    """Somewhere that funds can be deposited to."""


@dataclass(frozen=True, eq=False)
class Account(CashSource, CashSink):
    """An endpoint with a measurable balance."""


@dataclass(frozen=True, eq=False)
class CashFlow:
    """Specification of a "cash flow": something that causes an amount of funds to flow from one place to another."""

    label: str
    source: CashSource
    sink: CashSink
    amount: FloatDistribution
    tags: frozenset[str] = frozenset()      # Arbitrary tags that can be used for custom categorisation or analysis.


@dataclass(frozen=True, eq=False)
class CashFlowEvent:
    """An instance of a cash flow whose date is uncertain."""

    cash_flow: CashFlow
    distribution: DiscreteDistribution[date]

    def iterate(self, date_range: DateRange, /) -> Iterable['CashFlowOccurrence']:
        return (CashFlowOccurrence(**asdict(occurrence), event=self)
                for occurrence in self.distribution.iterate(date_range))


@dataclass(frozen=True, eq=False)
class CashFlowOccurrence(DiscreteOutcome[date]):
    event: CashFlowEvent

    @property
    def date(self) -> date:
        return self.value


@dataclass(frozen=True, eq=False)
class ScheduledCashFlow:
    """Specification of a cash flow which occurs on some schedule."""

    cash_flow: CashFlow
    schedule: EventSchedule

    def iterate_events(self, date_range: DateRange, /) -> Iterable[CashFlowEvent]:
        return (CashFlowEvent(self.cash_flow, distribution) for distribution in self.schedule.iterate(date_range))

    def iterate_occurrences(self, date_range: DateRange, /) -> Iterable[CashFlowOccurrence]:
        return iterate_cash_flow_occurrences(self.iterate_events(date_range), date_range)


def iterate_cash_flow_occurrences(events: CashFlowEvent | Iterable[CashFlowEvent], date_range: DateRange) \
        -> Iterable[CashFlowOccurrence]:
    """Iterates all `CashFlowOccurrence` occurring from `events`, in chronological order."""

    if isinstance(events, CashFlowEvent):
        events = (events,)
    occurrence_iterators = (event.iterate(date_range) for event in events)
    return merge(*occurrence_iterators, key=lambda occurrence: occurrence.date)


@dataclass(frozen=True)
class AccountBalance:
    min: float = 0
    max: float = 0
    mean: float = 0

    @classmethod
    def exactly(cls, amount: float):
        return cls(min=amount, max=amount, mean=amount)

    def __add__(self, other: 'CashDelta', /):
        return type(self)(min=self.min + other.min, max=self.max + other.max, mean=self.mean + other.mean)


@dataclass(frozen=True, kw_only=True)
class CashDelta:
    """A change in cash balance, with some uncertainty on the amount."""

    min: float = 0
    max: float = 0
    mean: float = 0

    def __add__(self, other: 'CashDelta', /):
        return type(self)(min=self.min + other.min, max=self.max + other.max, mean=self.mean + other.mean)


@dataclass(frozen=True)
class AccountBalanceEvent:
    # The delta applies at the start of the day given by `date` (alternatively, at the end of the previous day).
    date: date
    account: Account
    delta: CashDelta


def generate_account_balance_events(cash_flows: Iterable[CashFlowOccurrence], /,
        certainty_tolerance: float = DEFAULT_CERTAINTY_TOLERANCE) -> Iterable[AccountBalanceEvent]:
    """Iterates all `AccountBalanceEvent` resulting from a sequence of `CashFlowOccurrence`.
        `cash_flows` must be presorted in chronological order."""

    seen_cash_flow_events: set[CashFlowEvent] = set()
    defer_queue: list[AccountBalanceEvent] = []
    for occurrence in cash_flows:
        # Yield previously produced events for past days.
        while defer_queue and defer_queue[0].date <= occurrence.date:
            yield defer_queue.pop(0)

        cash_flow = occurrence.event.cash_flow
        source = cash_flow.source
        sink = cash_flow.sink
        amount = cash_flow.amount
        source_is_account = isinstance(source, Account)
        sink_is_account = isinstance(sink, Account)
        following_date = occurrence.date + timedelta(days=1)
        if occurrence.event not in seen_cash_flow_events:
            # Date lower bound - first time the event could occur (within the timeframe we're interested in).
            # Lower bound is at the start of the day of occurrence.
            if source_is_account:
                yield AccountBalanceEvent(occurrence.date, source, CashDelta(min=-amount.max))
            if sink_is_account:
                yield AccountBalanceEvent(occurrence.date, sink, CashDelta(max=amount.max))
        # Mean increases linearly up to the end of the day of occurrence (i.e. the following day).
        if source_is_account:
            # Since this event occurs on the following day, we have to defer yielding until later to preserve
            # chronological order.
            defer_queue.append(
                AccountBalanceEvent(following_date, source, CashDelta(mean=-amount.mean * occurrence.probability)))
        if sink_is_account:
            defer_queue.append(
                AccountBalanceEvent(following_date, sink, CashDelta(mean=amount.mean * occurrence.probability)))
        # FIXME? If the cumulative probability is ~1 for more than one occurrence then this will break (but that
        # shouldn't occur in normal usage)
        if effectively_certain(occurrence.cumulative_probability, certainty_tolerance):
            # Date upper bound - event must have occurred by now.
            # Upper bound is at the end of the day of occurrence (i.e. the following day).
            if source_is_account:
                defer_queue.append(AccountBalanceEvent(following_date, source, CashDelta(max=-amount.min)))
            if sink_is_account:
                defer_queue.append(AccountBalanceEvent(following_date, sink, CashDelta(min=amount.min)))
        seen_cash_flow_events.add(occurrence.event)


@dataclass(frozen=True)
class AccountBalancesRecord:
    # The balances are taken at the start of the day given by `date` (alternatively, at the end of the previous day).
    date: date
    balances: Mapping[Account, AccountBalance]


def simulate_account_balance_events(events: Iterable[AccountBalanceEvent],
        initial_account_balances: Mapping[Account, AccountBalance]) -> Iterable[AccountBalancesRecord]:
    """Simulates the effect of a sequence events on account balances, yielding account balances records for each day.
        `events` must be presorted in chronological order."""

    account_balances = dict(initial_account_balances)
    for day, day_events in groupby(events, lambda event: event.date):
        for event in day_events:
            account_balances[event.account] += event.delta
        # Account balances are taken at the end of the day, but stored as start of day (because it's easier to use).
        # Note that event.date is the start of the day the event occurs on.
        yield AccountBalancesRecord(day, account_balances.copy())
