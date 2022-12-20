from abc import ABC
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from heapq import merge
from itertools import groupby

from matplotlib import pyplot

from .date_time import DateRange
from .probability import DEFAULT_CERTAINTY_TOLERANCE, FloatDistribution, effectively_certain
from .schedule import DateDistribution, EventSchedule
from .utility import merge_by_date


__all__ = [
    'accumulate_endpoint_balances',
    'CashBalance',
    'CashBalanceDelta',
    'CashBalanceUpdate',
    'CashBalanceRecord',
    'CashEndpoint',
    'CashFlowLog',
    'CashSink',
    'CashSource',
    'generate_balance_updates',
    'generate_cash_flow_logs',
    'plot_balances_over_time',
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

    def __add__(self, other: 'CashBalanceDelta', /):
        return type(self)(min=self.min + other.min, max=self.max + other.max, mean=self.mean + other.mean)


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

        assert event.possible_in(date_range.inclusive_lower_bound, date_range.exclusive_upper_bound)
        for occurrence in event.iterate(date_range.inclusive_lower_bound, date_range.exclusive_upper_bound):
            # Mean increases linearly up to the end of the day of occurrence (i.e. the following day).
            # Since this event occurs on the following day, we have to defer yielding until later to preserve
            # chronological order.
            following_date = occurrence.value + timedelta(days=1)
            yield CashBalanceUpdate(
                following_date, source, CashBalanceDelta(mean=-amount.mean * occurrence.probability), cash_flow)
            yield CashBalanceUpdate(
                following_date, sink, CashBalanceDelta(mean=amount.mean * occurrence.probability), cash_flow)

        last_occurrence = event.upper_bound_inclusive(date_range.inclusive_upper_bound)
        assert last_occurrence is not None
        has_upper_bound = effectively_certain(last_occurrence.cumulative_probability, tolerance=certainty_tolerance)
        if has_upper_bound:
            # Date upper bound - event must have occurred by now.
            # Upper bound is at the end of the day of occurrence (i.e. the following day).
            following_date = last_occurrence.value + timedelta(days=1)
            yield CashBalanceUpdate(following_date, source, CashBalanceDelta(max=-amount.min), cash_flow)
            yield CashBalanceUpdate(following_date, sink, CashBalanceDelta(min=amount.min), cash_flow)

    events = cash_flow.schedule.iterate(date_range)
    event_update_iterators = map(generate_event_updates, events)
    return merge_by_date(event_update_iterators)


@dataclass(frozen=True, kw_only=True)
class CashBalance:
    min: float = 0
    max: float = 0
    mean: float = 0

    @classmethod
    def exactly(cls, amount: float, /):
        return cls(min=amount, max=amount, mean=amount)

    def __add__(self, other: CashBalanceDelta, /):
        return type(self)(min=self.min + other.min, max=self.max + other.max, mean=self.mean + other.mean)


@dataclass(frozen=True)
class CashBalanceRecord:
    # The balance is taken at the start of the day given by `date` (alternatively, at the end of the previous day).
    date: date
    amount: CashBalance


def accumulate_endpoint_balances(updates: Iterable[CashBalanceUpdate], /,
        initial_balances: Mapping[CashEndpoint, CashBalance] = {}) -> dict[CashEndpoint, list[CashBalanceRecord]]:
    """Generates cumulative balances for all endpoints resulting from a sequence of `CashBalanceUpdate`.

        The initial balance for an endpoint is taken from `initial_balances`, if an entry is present. Otherwise, the
        initial balance taken as min=0 mean=0 max=0.

        `updates` must be presorted in chronological order."""

    endpoint_balances: defaultdict[CashEndpoint, CashBalance] = defaultdict(CashBalance)
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
    return result


def simulate_cash_balances(cash_flows: Iterable[ScheduledCashFlow], date_range: DateRange,
        initial_balances: Mapping[CashEndpoint, float] = {}, certainty_tolerance: float = DEFAULT_CERTAINTY_TOLERANCE) \
        -> dict[CashEndpoint, list[CashBalanceRecord]]:
    """Simulates cash balances of endpoints resulting from cash flows over the specified timeframe."""

    initial_balances_ = {endpoint: CashBalance.exactly(balance) for endpoint, balance in initial_balances.items()}

    balance_updates = merge_by_date(
        generate_balance_updates(cash_flow, date_range, certainty_tolerance=certainty_tolerance)
        for cash_flow in cash_flows)
    balance_records = accumulate_endpoint_balances(balance_updates, initial_balances=initial_balances_)

    # Prepend opening account balances at the start of the specified timeframe.
    for endpoint, balance in initial_balances_.items():
        balance_records[endpoint].insert(0, CashBalanceRecord(date_range.inclusive_lower_bound, balance))

    # Append closing balances at the end of the specified timeframe (if not already present).
    if not date_range.is_empty and balance_records:
        for records in balance_records.values():
            if records and date_range.exclusive_upper_bound > records[-1].date:
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
        if event.certain_in(date_range.inclusive_lower_bound, date_range.exclusive_upper_bound,
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
            return 'v'
        elif self.bound_type > 0:
            return '^'
        else:
            return '-'


def generate_cash_flow_logs(cash_flow: ScheduledCashFlow, date_range: DateRange, /,
        certainty_tolerance: float = DEFAULT_CERTAINTY_TOLERANCE) -> Iterable[CashFlowLog]:
    """Generates a human-readable description for each cash flow event within the given timeframe."""

    def generate_event_logs(event: DateDistribution, /) -> Iterable[CashFlowLog]:
        first_occurrence = event.lower_bound_inclusive(date_range.inclusive_lower_bound)
        assert first_occurrence is not None
        last_occurrence = event.upper_bound_inclusive(date_range.inclusive_upper_bound)
        assert last_occurrence is not None
        has_upper_bound = effectively_certain(last_occurrence.cumulative_probability, tolerance=certainty_tolerance)

        if not has_upper_bound or first_occurrence.value == last_occurrence.value:
            yield CashFlowLog(first_occurrence.value, 0, cash_flow.amount, cash_flow.source, cash_flow.sink)
        else:
            yield CashFlowLog(first_occurrence.value, -1, cash_flow.amount, cash_flow.source, cash_flow.sink)
            yield CashFlowLog(last_occurrence.value, 1, cash_flow.amount, cash_flow.source, cash_flow.sink)

    events = cash_flow.schedule.iterate(date_range)
    event_log_iterators = map(generate_event_logs, events)
    return merge(*event_log_iterators)


def plot_balances_over_time(endpoint_balances: Mapping[CashEndpoint, Sequence[CashBalanceRecord]], /) -> None:
    """Plots `CashEndpoint` balances over time.

        `endpoint_balances` must be presorted in chronological order."""

    if not endpoint_balances:
        raise ValueError('endpoint_balances must not be empty')

    def extract_individual_series(balances: Sequence[CashBalanceRecord]):
        return (
            [balance.date for balance in balances],
            [balance.amount.min for balance in balances],
            [balance.amount.max for balance in balances],
            [balance.amount.mean for balance in balances]
        )

    def plot_balances(dates: Sequence[date], min_balances: Sequence[float], max_balances: Sequence[float],
            mean_balances: Sequence[float], label: str) -> None:
        min_widths = [mean - min_ for mean, min_ in zip(mean_balances, min_balances, strict=True)]
        max_widths = [max_ - mean for mean, max_ in zip(mean_balances, max_balances, strict=True)]
        plot = pyplot.errorbar(dates, mean_balances, yerr=(min_widths, max_widths), label=label)
        plot[-1][0].set_linestyle('--')

    extracted_series = {
        endpoint: extract_individual_series(balances) for endpoint, balances in endpoint_balances.items()}

    min_date = min(min(data[0]) for data in extracted_series.values())
    max_date = max(max(data[0]) for data in extracted_series.values())

    for endpoint, balances in endpoint_balances.items():
        plot_balances(*extract_individual_series(balances), endpoint.label)

    pyplot.title(f'Funds from {min_date} to {max_date}')
    pyplot.xlabel('Date')
    pyplot.ylabel('Funds ($)')
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.show()
