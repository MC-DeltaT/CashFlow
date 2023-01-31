from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from heapq import merge
from typing import Callable

from matplotlib import pyplot

from .cash_flow import (
    CashBalanceRecord, CashEndpoint, CashSink, CashSource, ScheduledCashFlow, generate_cash_flow_logs,
    simulate_cash_balances, summarise_total_cash_flow)
from .date_time import DateRange
from .probability import DEFAULT_CERTAINTY_TOLERANCE, FloatDistribution
from .schedule import EventSchedule


__all__ = [
    'Account',
    'CashFlowAnalysis',
    'ExpenseSink',
    'IncomeSource',
    'plot_balances_over_time',
    'ScheduleBuilder',
    'tagset'
]


@dataclass(frozen=True)
class Account(CashSource, CashSink):
    """E.g. a bank account, savings account, investment portfolio, etc."""

    tags: frozenset[str] = frozenset()  # Arbitrary tags that can be used for custom categorisation or analysis.


def tagset(*tags: str) -> frozenset[str]:
    """Helper for creating a set of tags, because frozenset's constructor is annoying."""

    return frozenset(tags)


@dataclass(frozen=True)
class IncomeSource(CashSource):
    pass


@dataclass(frozen=True)
class ExpenseSink(CashSink):
    pass


class ScheduleBuilder:
    """Helper for defining scheduled cash flows."""

    def __init__(self, general_account: Account | None = None) -> None:
        """
            :param general_account: The default account for incomes and expenses.
        """

        self.general_account = general_account
        self.scheduled_cash_flows: list[ScheduledCashFlow] = []

    def cash_flow(self, label: str, schedule: EventSchedule, amount: float | FloatDistribution, source: CashSource,
            sink: CashSink, tags: frozenset[str] = frozenset()) -> ScheduledCashFlow:
        if not isinstance(amount, FloatDistribution):
            amount = FloatDistribution.singular(amount)
        flow = ScheduledCashFlow(label, source, sink, amount, schedule, tags)
        self.scheduled_cash_flows.append(flow)
        return flow

    def income(self, label: str, schedule: EventSchedule, amount: float | FloatDistribution,
            account: Account | None = None, tags: frozenset[str] = frozenset()) -> ScheduledCashFlow:
        return self.cash_flow(label, schedule, amount, IncomeSource(label), self._account_or_default(account), tags)

    def expense(self, label: str, schedule: EventSchedule, amount: float | FloatDistribution,
            account: Account | None = None, tags: frozenset[str] = frozenset()) -> ScheduledCashFlow:
        return self.cash_flow(label, schedule, amount, self._account_or_default(account), ExpenseSink(label), tags)

    def transfer(self, label: str, schedule: EventSchedule, amount: float | FloatDistribution,
            source_account: Account, destination_account: Account, tags: frozenset[str] = frozenset()) \
            -> ScheduledCashFlow:
        return self.cash_flow(label, schedule, amount, source_account, destination_account, tags)

    def make_analysis(self, date_range: DateRange, certainty_tolerance: float = DEFAULT_CERTAINTY_TOLERANCE) \
            -> 'CashFlowAnalysis':
        return CashFlowAnalysis(self.scheduled_cash_flows, date_range, certainty_tolerance=certainty_tolerance)

    def _account_or_default(self, account: Account | None) -> Account:
        account = account or self.general_account
        if account is None:
            raise ValueError('Either account or general_account must be provided')
        else:
            return account


class CashFlowAnalysis:
    """Helper for high-level analysis of cash flows."""

    def __init__(self, cash_flows: Iterable[ScheduledCashFlow], date_range: DateRange,
            certainty_tolerance: float = DEFAULT_CERTAINTY_TOLERANCE) -> None:
        self._cash_flows = tuple(cash_flows)
        self._date_range = date_range
        self._certainty_tolerance = certainty_tolerance

    def log_cash_flows(self) -> None:
        """Prints a human-readable description for each cash flow event."""

        log_iterators = (
            generate_cash_flow_logs(cash_flow, self._date_range, certainty_tolerance=self._certainty_tolerance)
            for cash_flow in self._cash_flows)
        logs = merge(*log_iterators)
        for log in logs:
            print(log)

    def summarise_cash_flows(self, label: str, cash_flow_filter: Callable[[ScheduledCashFlow], bool]) -> None:
        cash_flows = tuple(filter(cash_flow_filter, self._cash_flows))
        total = sum(
            (summarise_total_cash_flow(cash_flow, self._date_range, certainty_tolerance=self._certainty_tolerance)
             for cash_flow in cash_flows),
            FloatDistribution(min=0, mean=0, max=0))
        print(f'Total {label}: ${total.to_str(2)}')

    # TODO: fix the issue with Mapping variance
    def plot_balances_over_time(self, endpoints: Collection[CashEndpoint],
            initial_balances: Mapping[CashEndpoint, float] = {}) -> None:
        initial_balance_dists = {
                endpoint: FloatDistribution.singular(balance) for endpoint, balance in initial_balances.items()}
        endpoint_balances = simulate_cash_balances(
            self._cash_flows, self._date_range, initial_balance_dists, self._certainty_tolerance)
        endpoint_balances = {
            endpoint: balances for endpoint, balances in endpoint_balances.items() if endpoint in endpoints}
        plot_balances_over_time(endpoint_balances)


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

    for endpoint, extracted in extracted_series.items():
        plot_balances(*extracted, endpoint.label)

    pyplot.title(f'Funds from {min_date} to {max_date}')
    pyplot.xlabel('Date')
    pyplot.ylabel('Funds ($)')
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.show()
