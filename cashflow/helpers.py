"""Functionality that is not core, but is useful for building real code."""


from dataclasses import dataclass
from itertools import chain
from typing import Iterable

from .core import (
    Account, CashFlow, CashFlowOccurrence, CashSink, CashSource, ScheduledCashFlow, iterate_cash_flow_occurrences)
from .datetime import DateRange
from .probability import FloatDistribution
from .schedule import EventSchedule


__all__ = [
    'ExpenseSink',
    'IncomeSource',
    'ScheduleBuilder',
    'tagset'
]


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
    """
        :param general_account: The default account for incomes and expenses.
    """
    def __init__(self, general_account: Account | None = None) -> None:
        self.general_account = general_account
        self.scheduled_cash_flows: list[ScheduledCashFlow] = []

    def income(self, label: str, schedule: EventSchedule, amount: float | FloatDistribution,
            account: Account | None = None, tags: frozenset[str] = frozenset()) -> ScheduledCashFlow:
        if isinstance(amount, (float, int)):
            amount = FloatDistribution.singular(amount)
        account = account or self.general_account
        if account is None:
            raise ValueError('Either account or general_account must be provided')
        income = ScheduledCashFlow(CashFlow(label, IncomeSource(label), account, amount, tags), schedule)
        self.scheduled_cash_flows.append(income)
        return income

    def expense(self, label: str, schedule: EventSchedule, amount: float | FloatDistribution,
            account: Account | None = None, tags: frozenset[str] = frozenset()) -> ScheduledCashFlow:
        if isinstance(amount, (float, int)):
            amount = FloatDistribution.singular(amount)
        account = account or self.general_account
        if account is None:
            raise ValueError('Either account or general_account must be provided')
        expense = ScheduledCashFlow(CashFlow(label, account, ExpenseSink(label), amount), schedule)
        self.scheduled_cash_flows.append(expense)
        return expense

    def transfer(self, label: str, schedule: EventSchedule, amount: float | FloatDistribution,
            source_account: Account, destination_account: Account, tags: frozenset[str] = frozenset()) \
            -> ScheduledCashFlow:
        if isinstance(amount, (float, int)):
            amount = FloatDistribution.singular(amount)
        transfer = ScheduledCashFlow(CashFlow(label, source_account, destination_account, amount), schedule)
        self.scheduled_cash_flows.append(transfer)
        return transfer

    def iterate_occurrences(self, date_range: DateRange, /) -> Iterable[CashFlowOccurrence]:
        """Iterates all `CashFlowOccurrence` from all scheduled cash flows, in chronological order."""

        event_iterators = (scheduled.iterate_events(date_range) for scheduled in self.scheduled_cash_flows)
        return iterate_cash_flow_occurrences(chain.from_iterable(event_iterators), date_range)
