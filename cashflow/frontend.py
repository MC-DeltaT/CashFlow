from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from heapq import merge
from typing import Callable

from .cash_flow import (
    CashEndpoint, CashSink, CashSource, ScheduledCashFlow, generate_cash_flow_logs, plot_balances_over_time,
    simulate_cash_balances, summarise_total_cash_flow)
from .date_time import DateRange
from .probability import DEFAULT_CERTAINTY_TOLERANCE, FloatDistribution
from .schedule import EventSchedule


__all__ = [
    'Account',
    'CashFlowAnalysis',
    'ExpenseSink',
    'IncomeSource',
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

    """
        :param general_account: The default account for incomes and expenses.
    """
    def __init__(self, general_account: Account | None = None) -> None:
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
        endpoint_balances = simulate_cash_balances(
            self._cash_flows, self._date_range, initial_balances, self._certainty_tolerance)
        endpoint_balances = {
            endpoint: balances for endpoint, balances in endpoint_balances.items() if endpoint in endpoints}
        plot_balances_over_time(endpoint_balances)
