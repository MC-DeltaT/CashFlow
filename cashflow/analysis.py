"""Functionality for high-level analysis and visualisation."""


from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Callable, Collection, DefaultDict, Iterable, Mapping, Sequence

from matplotlib import pyplot
import numpy

from .core import (
    Account, AccountBalance, AccountBalancesRecord, CashFlow, CashFlowEvent, CashFlowOccurrence,
    generate_account_balance_events, simulate_account_balance_events)
from .datetime import DateRange
from .probability import DEFAULT_CERTAINTY_TOLERANCE, effectively_certain


__all__ = [
    'CashFlowSummary',
    'extract_account_balance_series',
    'ExtractedAccountBalanceSeries',
    'generate_cash_flow_logs',
    'plot_funds_over_time',
    'simulate_cash_flows',
    'summarise_cash_flows'
]


def generate_cash_flow_logs(cash_flows: Iterable[CashFlowOccurrence], /,
        certainty_tolerance: float = DEFAULT_CERTAINTY_TOLERANCE) -> Iterable[str]:
    """Generates a human-readable description for each cash flow event.
        `cash_flows` must be presorted in chronological order."""

    def format_log(cash_flow: CashFlow, event_date: date, bound_type: int) -> str:
        if bound_type < 0:
            # Lower bound.
            marker = 'v'
        elif bound_type > 0:
            # Upper bound.
            marker = '^'
        else:
            # Combined lower and upper bound.
            marker = '-'

        if cash_flow.amount.lower_bound == cash_flow.amount.upper_bound:
            amount_str = f'${round(cash_flow.amount.lower_bound, 2)}'
        else:
            amount_str = f'${cash_flow.amount.lower_bound}-${cash_flow.amount.upper_bound}'

        return f'{event_date} {marker} | {amount_str} from "{cash_flow.source.label}" to "{cash_flow.sink.label}"'

    seen_events: set[CashFlowEvent] = set()
    for occurrence in cash_flows:
        is_lower_bound = occurrence.event not in seen_events
        # FIXME? If the cumulative probability is ~1 for more than one occurrence then this will break (but that
        # shouldn't occur in normal usage)
        is_upper_bound = effectively_certain(occurrence.cumulative_probability, tolerance=certainty_tolerance)
        if is_lower_bound and is_upper_bound:
            bound_type = 0
        elif is_lower_bound:
            bound_type = -1
        elif is_upper_bound:
            bound_type = 1
        else:
            continue
        yield format_log(occurrence.event.cash_flow, occurrence.date, bound_type)
        seen_events.add(occurrence.event)


@dataclass(frozen=True)
class CashFlowSummary:
    label: str
    min_total: float
    max_total: float

    def __str__(self) -> str:
        min_total = round(self.min_total, 2)
        max_total = round(self.max_total, 2)
        if max_total - min_total < 0.01:
            total_str = f'${min_total}'
        else:
            total_str = f'${min_total} to ${max_total}'
        return f'Total {self.label}: {total_str}'


def summarise_cash_flows(cash_flows: Collection[CashFlowOccurrence], label: str, /,
        certainty_tolerance: float = DEFAULT_CERTAINTY_TOLERANCE) -> CashFlowSummary:
    """Calculates bounds on the total amount of cash transferred by `cash_flows`."""

    events = {occurrence.event for occurrence in cash_flows}
    max_amount = sum(event.cash_flow.amount.upper_bound for event in events)
    # Floating point error can cause cumulative probability to be slightly less than 1 even for certain events.
    certain_occurrences = (occurrence for occurrence in cash_flows
                           if effectively_certain(occurrence.cumulative_probability, tolerance=certainty_tolerance))
    min_amount = sum(occurrence.event.cash_flow.amount.lower_bound for occurrence in certain_occurrences)
    return CashFlowSummary(label, min_amount, max_amount)


def simulate_cash_flows(cash_flows: Sequence[CashFlowOccurrence], accounts: Iterable[Account], timeframe: DateRange,
        initial_account_balances: Mapping[Account, float] = {}, /,
        certainty_tolerance: float = DEFAULT_CERTAINTY_TOLERANCE) -> list[AccountBalancesRecord]:
    """High level simulation of cash flows.
        `cash_flows` must be presorted in chronological order."""

    initial_account_balances_ = {
        account: AccountBalance.exactly(initial_account_balances.get(account, 0)) for account in accounts}

    account_balance_events = generate_account_balance_events(cash_flows, certainty_tolerance=certainty_tolerance)
    account_balance_records = list(simulate_account_balance_events(account_balance_events, initial_account_balances_))

    # Prepend opening account balances at the start of the specified timeframe.
    account_balance_records.insert(0, AccountBalancesRecord(timeframe.inclusive_lower_bound, initial_account_balances_))

    # Append closing balances at the end of the specified timeframe (if not already present).
    if not timeframe.is_empty and account_balance_records:
        if timeframe.exclusive_upper_bound > account_balance_records[-1].date:
            account_balance_records.append(
                AccountBalancesRecord(timeframe.exclusive_upper_bound, account_balance_records[-1].balances))

    return account_balance_records


@dataclass
class ExtractedAccountBalanceSeries:
    # Each balance value occurs at the start of the day given by the corresponding date.
    dates: list[date]
    min_balances: dict[Account, list[float]]
    max_balances: dict[Account, list[float]]
    mean_balances: dict[Account, list[float]]


def extract_account_balance_series(account_records: Iterable[AccountBalancesRecord], /) -> ExtractedAccountBalanceSeries:
    """Extracts individual arrays of dates, min balances, max balances, and mean balances."""

    dates: list[date] = []
    min_balances: DefaultDict[Account, list[float]] = defaultdict(list)
    max_balances: DefaultDict[Account, list[float]] = defaultdict(list)
    mean_balances: DefaultDict[Account, list[float]] = defaultdict(list)
    for record in account_records:
        dates.append(record.date)
        # FIXME? There is an issue here if not all records have the same set of accounts (but that shouldn't occur in
        # normal usage)
        for account, balance in record.balances.items():
            min_balances[account].append(balance.min)
            max_balances[account].append(balance.max)
            mean_balances[account].append(balance.mean)
    return ExtractedAccountBalanceSeries(dates, min_balances, max_balances, mean_balances)


def plot_funds_over_time(account_records: Sequence[AccountBalancesRecord],
        *custom_groups: tuple[str, Callable[[Account], bool]]) -> None:
    """Plots account balances over time.

        `account_records` must be presorted in chronological order.

        `custom_groups` are specifications of groups of account balances that are summed together to form custom balance
        plots. Each item is a tuple of the group label and function to filter accounts."""

    if not account_records:
        raise ValueError('account_records must not be empty')

    extracted_series = extract_account_balance_series(account_records)
    accounts = extracted_series.min_balances.keys()

    def plot_balances(dates: Sequence[date], min_balances: Sequence[float], max_balances: Sequence[float],
            mean_balances: Sequence[float], label: str) -> None:
        min_widths = [mean - min_ for mean, min_ in zip(mean_balances, min_balances)]
        max_widths = [max_ - mean for mean, max_ in zip(mean_balances, max_balances)]
        plot = pyplot.errorbar(dates, mean_balances, yerr=(min_widths, max_widths), label=label)
        plot[-1][0].set_linestyle('--')

    for account in accounts:
        plot_balances(extracted_series.dates, extracted_series.min_balances[account],
            extracted_series.max_balances[account], extracted_series.mean_balances[account], account.label)

    for label, account_filter in custom_groups:
        group_accounts = [account for account in accounts if account_filter(account)]
        if group_accounts:
            min_balances = numpy.sum([extracted_series.min_balances[account] for account in group_accounts], axis=0)
            max_balances = numpy.sum([extracted_series.max_balances[account] for account in group_accounts], axis=0)
            mean_balances = numpy.sum([extracted_series.mean_balances[account] for account in group_accounts], axis=0)
        else:
            min_balances = numpy.zeros_like(extracted_series.dates)
            max_balances = numpy.zeros_like(extracted_series.dates)
            mean_balances = numpy.zeros_like(extracted_series.dates)
        plot_balances(extracted_series.dates, min_balances, max_balances, mean_balances, label)

    pyplot.title(f'Funds from {account_records[0].date} to {account_records[-1].date}')
    pyplot.xlabel('Date')
    pyplot.ylabel('Funds ($)')
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.show()
