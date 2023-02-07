from abc import ABC
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from heapq import merge
from itertools import groupby

from .date_time import DateRange
from .probability import DEFAULT_CERTAINTY_TOLERANCE, FloatDistribution, clamp_certain, effectively_certain
from .schedule import DateDistribution, EventSchedule
from .utility import merge_by_date


__all__ = [
    'accumulate_endpoint_balances',
    'CashBalanceDelta',
    'CashBalanceUpdate',
    'CashBalanceRecord',
    'CashEndpoint',
    'CashFlowLog',
    'CashSink',
    'CashSource',
    'generate_balance_updates',
    'generate_cash_flow_logs',
    'ScheduledCashFlow',
    'simulate_cash_balances',
    'summarise_total_cash_flow'
]


@dataclass(frozen=True)
class CashEndpoint(ABC):
    """Somewhere that funds come from or go to."""

    label: str


@dataclass(frozen=True)
class CashSource(CashEndpoint, ABC):
    """Somewhere that funds can be sourced from."""


@dataclass(frozen=True)
class CashSink(CashEndpoint, ABC):
    """Somewhere that funds can be deposited to."""


@dataclass(frozen=True, eq=False)
class ScheduledCashFlow:
    """A scheduled event that causes an amount of funds to flow from one place to another."""

    label: str
    source: CashSource
    sink: CashSink
    amount: FloatDistribution
    schedule: EventSchedule
    tags: frozenset[str] = frozenset()      # Arbitrary tags that can be used for custom categorisation or analysis.

    def __post_init__(self) -> None:
        if self.amount.min < 0:
            raise ValueError('amount must be nonnegative')


@dataclass(frozen=True, kw_only=True)
class CashBalanceDelta:
    """A change in an uncertain cash balance."""

    min: float = 0      # Change in the minimum balance.
    max: float = 0      # Change in the maximum balance.
    mean: float = 0     # Change in the mean balance.

    def __add__(self, other: 'CashBalanceDelta', /) -> 'CashBalanceDelta':
        if isinstance(other, CashBalanceDelta):
            return type(self)(min=self.min + other.min, max=self.max + other.max, mean=self.mean + other.mean)
        else:
            return NotImplemented

    def __radd__(self, other: FloatDistribution, /):
        return other.from_inexact(min=other.min + self.min, max=other.max + self.max, mean=other.mean + self.mean)


@dataclass(frozen=True)
class CashBalanceUpdate:
    """Describes a change in the relative cash balance of a `CashEndpoint` as a result of a possible cash flow event."""

    # The delta applies at the start of the day given by `date` (alternatively, at the end of the previous day).
    date: date
    endpoint: CashEndpoint
    delta: CashBalanceDelta
    cause: ScheduledCashFlow


def generate_balance_updates(cash_flow: ScheduledCashFlow, date_range: DateRange, /,
        certainty_tolerance: float = DEFAULT_CERTAINTY_TOLERANCE) -> Iterable[CashBalanceUpdate]:
    """Generates `CashBalanceUpdate` resulting from occurrences of `cash_flow` within `date_range`, in chronological
        order.

        This operation assumes `date_range` span the entire timeframe you're interested in. This is due to the way min
        and max balances are updated when looking at a subset of all possible event occurrences. Therefore it is not
        safe to splice or concatenate resulting sequences of `CashBalanceUpdate`."""

    source = cash_flow.source
    sink = cash_flow.sink
    amount = cash_flow.amount

    def generate_event_updates(event: DateDistribution, /) -> Iterable[CashBalanceUpdate]:
        # Date lower bound - first time the event could possibly occur (within the timeframe we're interested in).
        # Lower bound is at the start of the day of occurrence.
        first_occurrence = event.lower_bound_inclusive(date_range.inclusive_lower_bound)
        assert first_occurrence is not None
        yield CashBalanceUpdate(first_occurrence.value, source, CashBalanceDelta(min=-amount.max), cash_flow)
        yield CashBalanceUpdate(first_occurrence.value, sink, CashBalanceDelta(max=amount.max), cash_flow)

        probability_in_range = event.probability_in(date_range.inclusive_lower_bound, date_range.exclusive_upper_bound)
        assert probability_in_range > 0

        for occurrence in event.iterate(date_range.inclusive_lower_bound, date_range.exclusive_upper_bound):
            # Mean increases linearly up to the end of the day of occurrence (i.e. start of the following day).
            # The following day could be outside the requested date range, but we'll allow it because it's equivalent to
            # the end of the last day in the range.
            following_date = occurrence.value + timedelta(days=1)

            # If we count the occurrence as certain, then it doesn't make much sense to adjust the mean by any
            # probability other than 1.
            update_amount = amount.mean * clamp_certain(occurrence.probability, tolerance=certainty_tolerance)

            yield CashBalanceUpdate(
                following_date, source, CashBalanceDelta(mean=-update_amount), cash_flow)
            yield CashBalanceUpdate(
                following_date, sink, CashBalanceDelta(mean=update_amount), cash_flow)

        last_occurrence = event.upper_bound_inclusive(date_range.inclusive_upper_bound)
        assert last_occurrence is not None
        has_upper_bound = effectively_certain(event.cumulative_probability(date_range.inclusive_upper_bound),
            tolerance=certainty_tolerance)
        if has_upper_bound:
            # Date upper bound - event must have occurred by now.
            # Upper bound is at the end of the day of occurrence (i.e. start of the following day).
            # The following day could be outside the requested date range, but we'll allow it because it's equivalent to
            # the end of the last day in the range.
            following_date = last_occurrence.value + timedelta(days=1)

            # Need to scale the update amount by the probability that the event occurs within the specified range to
            # ensure consistent distributions when accumulating account balances.
            # Consider the case where the event is possible to occur before the date range, the source's max balance
            # must not fall below its mean balance (and the sink's min must not rise above its mean).
            update_amount = amount.min * clamp_certain(probability_in_range, tolerance=certainty_tolerance)
            yield CashBalanceUpdate(following_date, source, CashBalanceDelta(max=-update_amount), cash_flow)
            yield CashBalanceUpdate(following_date, sink, CashBalanceDelta(min=update_amount), cash_flow)

    events = cash_flow.schedule.iterate(date_range)
    event_update_iterators = map(generate_event_updates, events)
    return merge_by_date(event_update_iterators)


@dataclass(frozen=True)
class CashBalanceRecord:
    # The balance is taken at the start of the day given by `date` (alternatively, at the end of the previous day).
    date: date
    amount: FloatDistribution


def accumulate_endpoint_balances(updates: Iterable[CashBalanceUpdate], /,
        initial_balances: Mapping[CashEndpoint, FloatDistribution] = {}) -> dict[CashEndpoint, list[CashBalanceRecord]]:
    """Generates cumulative balances for all endpoints resulting from a sequence of `CashBalanceUpdate`.

        The initial balance for an endpoint is taken from `initial_balances`, if an entry is present. Otherwise, the
        initial balance taken as min=0 mean=0 max=0.

        `updates` must be presorted in chronological order."""

    endpoint_balances: defaultdict[CashEndpoint, FloatDistribution] = defaultdict(lambda: FloatDistribution.singular(0))
    endpoint_balances.update(initial_balances)

    result: defaultdict[CashEndpoint, list[CashBalanceRecord]] = defaultdict(list)
    prev_day = date.min
    for day, day_updates in groupby(updates, lambda update: update.date):
        if day < prev_day:
            raise ValueError('updates must be in chronological order')

        endpoints_changed: set[CashEndpoint] = set()
        for update in day_updates:
            endpoint_balances[update.endpoint] += update.delta
            endpoints_changed.add(update.endpoint)
        # Accumulated balances are taken at the end of the day, but stored as start of day (because it's easier to use).
        # Note that update.date is the start of the day the update occurs on.
        for endpoint in endpoints_changed:
            result[endpoint].append(
                CashBalanceRecord(day, amount=endpoint_balances[endpoint]))

        prev_day = day

    return dict(result)


def simulate_cash_balances(cash_flows: Iterable[ScheduledCashFlow], date_range: DateRange,
        initial_balances: Mapping[CashEndpoint, FloatDistribution] = {},
        certainty_tolerance: float = DEFAULT_CERTAINTY_TOLERANCE) \
        -> dict[CashEndpoint, list[CashBalanceRecord]]:
    """Simulates cash balances of endpoints resulting from cash flows over the specified timeframe."""

    balance_updates = merge_by_date(
        generate_balance_updates(cash_flow, date_range, certainty_tolerance=certainty_tolerance)
        for cash_flow in cash_flows)
    balance_records = accumulate_endpoint_balances(balance_updates, initial_balances)

    if not date_range.is_empty:
        # Prepend opening account balances at the start of the specified timeframe.
        # The opening balance will never already be present, since the result of accumulate_endpoint_balances() will
        # start after the first balance update.
        for endpoint, balance in initial_balances.items():
            records = balance_records.setdefault(endpoint, [])
            if not records or date_range.inclusive_lower_bound < records[0].date:
                records.insert(0, CashBalanceRecord(date_range.inclusive_lower_bound, balance))

        # Append closing balances at the end of the specified timeframe (if not already present).
        for records in balance_records.values():
            if not records or date_range.exclusive_upper_bound > records[-1].date:
                records.append(CashBalanceRecord(date_range.exclusive_upper_bound, records[-1].amount))

    return balance_records


def summarise_total_cash_flow(cash_flow: ScheduledCashFlow, date_range: DateRange, /,
        certainty_tolerance: float = DEFAULT_CERTAINTY_TOLERANCE) -> FloatDistribution:
    """Calculates the distribution of the total amount of cash transferred by `cash_flow` within `date_range`."""

    if date_range.is_empty:
        return FloatDistribution(min=0, max=0, mean=0)

    events = tuple(cash_flow.schedule.iterate(date_range))

    # Minimum cash total happens when only the events which are certain to occur in the timeframe do occur.
    certain_events = sum(1 for event in events
        if effectively_certain(event.probability_in(date_range.inclusive_lower_bound, date_range.exclusive_upper_bound),
            tolerance=certainty_tolerance))
    min_amount = cash_flow.amount.min * certain_events

    # Maximum cash total happens when each possible event does occur. Note that all events from schedule.iterate() are
    # guaranteed to have a nonzero probability of occurring within the requested timeframe.
    max_amount = cash_flow.amount.max * len(events)

    # Mean cash total is simply the mean amount scaled by the total probability of occurrence within the timeframe.
    mean_amount = (cash_flow.amount.mean *
        sum(event.probability_in(date_range.inclusive_lower_bound, date_range.exclusive_upper_bound)
            for event in events))

    return FloatDistribution(min=min_amount, max=max_amount, mean=mean_amount)


@dataclass(frozen=True)
class CashFlowLog:
    """Creates a human-readable description for a cash flow event."""

    date: date
    bound_type: int     # < 0 is lower bound, > 0 is upper bound, 0 is both lower and upper bounds.
    exact_bound: bool   # If true, event is certain to occur within the bound.
    amount: FloatDistribution
    source: CashEndpoint
    sink: CashEndpoint

    def __lt__(self, other: 'CashFlowLog', /) -> bool:
        return (self.date, self.bound_type) < (other.date, other.bound_type)

    def __str__(self) -> str:
        return f'{self.date} {self._bound_marker} | ${self.amount.to_str(2)} from "{self.source.label}" to "{self.sink.label}"'

    @property
    def _bound_marker(self) -> str:
        if self.bound_type < 0:
            if self.exact_bound:
                return 'vv'
            else:
                return '~v'
        elif self.bound_type > 0:
            if self.exact_bound:
                return '^^'
            else:
                return '~^'
        else:
            if self.exact_bound:
                return '=='
            else:
                return '~~'


def generate_cash_flow_logs(cash_flow: ScheduledCashFlow, date_range: DateRange, /,
        certainty_tolerance: float = DEFAULT_CERTAINTY_TOLERANCE) -> Iterable[CashFlowLog]:
    """Generates a human-readable description for each cash flow event within the given timeframe.

        Logs are sorted by date."""

    def generate_event_logs(event: DateDistribution, /) -> Iterable[CashFlowLog]:
        first_occurrence = event.lower_bound_inclusive(date_range.inclusive_lower_bound)
        assert first_occurrence is not None
        last_occurrence = event.upper_bound_inclusive(date_range.inclusive_upper_bound)
        assert last_occurrence is not None

        exact_lower_bound = first_occurrence == event.outcomes[0]
        exact_upper_bound = (last_occurrence == event.outcomes[-1]
            and effectively_certain(event.cumulative_probability(last_occurrence.value), tolerance=certainty_tolerance))

        if first_occurrence.value == last_occurrence.value:
            exact = exact_lower_bound and exact_upper_bound
            yield CashFlowLog(first_occurrence.value, 0, exact, cash_flow.amount, cash_flow.source, cash_flow.sink)
        else:
            yield CashFlowLog(
                first_occurrence.value, -1, exact_lower_bound, cash_flow.amount, cash_flow.source, cash_flow.sink)
            yield CashFlowLog(
                last_occurrence.value, 1, exact_upper_bound, cash_flow.amount, cash_flow.source, cash_flow.sink)

    events = cash_flow.schedule.iterate(date_range)
    event_log_iterators = map(generate_event_logs, events)
    return merge(*event_log_iterators)
