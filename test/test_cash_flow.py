from datetime import date

from pytest import approx, raises

from cashflow.cash_flow import (
    CashBalanceDelta, CashBalanceUpdate, CashEndpoint, CashFlowLog, CashSink, CashSource, ScheduledCashFlow,
    accumulate_endpoint_balances, generate_cash_flow_logs, summarise_total_cash_flow)
from cashflow.date_time import DateRange
from cashflow.probability import FloatDistribution
from cashflow.schedule import DateDistribution, DayOfMonthDistribution, Monthly, Once, SimpleDayOfMonthSchedule


def test_scheduled_cash_flow_construct_invalid() -> None:
    with raises(ValueError):
        ScheduledCashFlow(
            'foo',
            CashSource('source'), CashSink('sink'),
            FloatDistribution(min=-0.1, mean=1, max=2),
            Once(date(2023, 1, 1)))


def test_cash_balance_delta_add() -> None:
    result = CashBalanceDelta(min=-10.2, max=53.83, mean=16.36) + CashBalanceDelta(min=4.18, max=-12.03, mean=19.77)
    assert result.min == approx(-6.02)
    assert result.max == approx(41.8)
    assert result.mean == approx(36.13)

def test_cash_balance_delta_radd() -> None:
    result = FloatDistribution(min=0.4, max=112.34, mean=13.2) + CashBalanceDelta(min=1.3, max=3.6, mean=2.04)
    assert result.min == approx(1.7)
    assert result.max == approx(115.94)
    assert result.mean == approx(15.24)


# TODO: test generate_balance_updates()


def test_accumulate_endpoint_balances_no_updates() -> None:
    initial_balances = {
        CashEndpoint('My endpoint'): FloatDistribution(min=10, max=45.2, mean=20.3),
        CashEndpoint('endpoint2'): FloatDistribution.singular(86)
    }
    result = accumulate_endpoint_balances((), initial_balances)
    assert result == {}

def test_accumulate_endpoint_balances_unsorted() -> None:
    updates = (
        CashBalanceUpdate(date(2023, 1, 15), CashEndpoint('foo'), CashBalanceDelta(), None),
        CashBalanceUpdate(date(2023, 1, 14), CashEndpoint('bar'), CashBalanceDelta(), None),
    )
    with raises(ValueError):
        accumulate_endpoint_balances(updates)

def test_accumulate_endpoint_balances() -> None:
    endpoint1 = CashEndpoint('endpoint 1')
    endpoint2 = CashEndpoint('endpoint2')
    updates = (
        CashBalanceUpdate(date(2023, 1, 16), endpoint1, CashBalanceDelta(min=2.4), None),
        CashBalanceUpdate(date(2023, 1, 17), endpoint1, CashBalanceDelta(mean=12.5), None),
        CashBalanceUpdate(date(2023, 1, 20), endpoint2, CashBalanceDelta(mean=-6.4, max=1), None),
        CashBalanceUpdate(date(2023, 1, 23), endpoint1, CashBalanceDelta(max=3), None),
        CashBalanceUpdate(date(2023, 1, 23), endpoint2, CashBalanceDelta(max=-1.3), None),
        CashBalanceUpdate(date(2023, 1, 23), endpoint1, CashBalanceDelta(mean=6, max=-1.1), None),
    )
    initial_balances = {
        endpoint1: FloatDistribution(min=1.6, max=45.2, mean=20.3),
        endpoint2: FloatDistribution(min=78, max=88, mean=86)
    }
    result = accumulate_endpoint_balances(updates, initial_balances)
    assert tuple(result.keys()) == (endpoint1, endpoint2)
    endpoint1_result = result[endpoint1]
    assert len(endpoint1_result) == 3
    assert endpoint1_result[0].date == date(2023, 1, 16)
    assert endpoint1_result[0].amount.approx_eq(FloatDistribution(min=4, max=45.2, mean=20.3))
    assert endpoint1_result[1].date == date(2023, 1, 17)
    assert endpoint1_result[1].amount.approx_eq(FloatDistribution(min=4, max=45.2, mean=32.8))
    assert endpoint1_result[2].date == date(2023, 1, 23)
    assert endpoint1_result[2].amount.approx_eq(FloatDistribution(min=4, max=47.1, mean=38.8))
    endpoint2_result = result[endpoint2]
    assert len(endpoint2_result) == 2
    assert endpoint2_result[0].date == date(2023, 1, 20)
    assert endpoint2_result[0].amount.approx_eq(FloatDistribution(min=78, max=89, mean=79.6))
    assert endpoint2_result[1].date == date(2023, 1, 23)
    assert endpoint2_result[1].amount.approx_eq(FloatDistribution(min=78, max=87.7, mean=79.6))


# TODO: test simulate_cash_balances()


def test_summarise_total_cash_flow_empty_range() -> None:
    cash_flow = ScheduledCashFlow(
        'test', CashSource('test source'), CashSink('test sink'),
        FloatDistribution.singular(10),
        Once(date(2023, 1, 1)))
    result = summarise_total_cash_flow(cash_flow, DateRange.empty())
    assert result == FloatDistribution(min=0, max=0, mean=0)

def test_summarise_cash_flow_no_occurrences() -> None:
    cash_flow = ScheduledCashFlow(
        'test', CashSource('test source'), CashSink('test sink'),
        FloatDistribution.singular(10),
        Once(date(2023, 1, 1)))
    result = summarise_total_cash_flow(
        cash_flow,
        DateRange.half_open(date(2000, 1, 1), date(2023, 1, 1)))
    assert result == FloatDistribution(min=0, max=0, mean=0)

def test_summarise_cash_flow_certain_occurrence() -> None:
    cash_flow = ScheduledCashFlow(
        'test', CashSource('test source'), CashSink('test sink'),
        FloatDistribution(min=10, max=34, mean=24.9),
        Once(date(2023, 1, 1)))
    result = summarise_total_cash_flow(
        cash_flow,
        DateRange.half_open(date(2022, 1, 1), date(2024, 1, 1)))
    assert result.approx_eq(FloatDistribution(min=10, max=34, mean=24.9))

def test_summarise_cash_flow_uncertain_occurrence() -> None:
    schedule = Once(DateDistribution.from_probabilities({
        date(2023, 1, 1): 0.2,
        date(2023, 1, 2): 0.1,
        date(2023, 1, 3): 0.1,
        date(2023, 1, 4): 0.1,
        date(2023, 1, 5): 0.1,
        date(2023, 1, 6): 0.1,
        date(2023, 1, 7): 0.1,
        date(2023, 1, 8): 0.1,
        date(2023, 1, 9): 0.05,
        date(2023, 1, 10): 0.05
    }))
    cash_flow = ScheduledCashFlow(
        'test', CashSource('test source'), CashSink('test sink'),
        FloatDistribution(min=10, max=100, mean=20),
        schedule)
    result = summarise_total_cash_flow(
        cash_flow,
        DateRange.inclusive(date(2023, 1, 6), date(2023, 2, 1)))
    assert result.approx_eq(FloatDistribution(min=0, max=100, mean=8))

def test_summarise_cash_flow_multiple_events() -> None:
    schedule = Monthly(SimpleDayOfMonthSchedule((
        DayOfMonthDistribution.from_probabilities({1: 0.3, 14: 0.1, 23: 0.6}),
        DayOfMonthDistribution.from_probabilities({3: 0.1, 4: 0.1, 5: 0.1})
    )))
    cash_flow = ScheduledCashFlow(
        'test', CashSource('test source'), CashSink('test sink'),
        FloatDistribution(min=10, max=100, mean=20),
        schedule)
    result = summarise_total_cash_flow(
        cash_flow,
        DateRange.inclusive(date(2023, 1, 20), date(2023, 3, 4)))
    # 2023/1: event1 P=0.6, event2 P=0
    # 2023/2: event1 P=1, event2 P=0.3
    # 2023/3: event1 P=0.3, event2 P=0.2
    assert result.approx_eq(FloatDistribution(min=10, max=500, mean=48))

def test_summarise_cash_flow_sanity_check() -> None:
    cash_flow1 = ScheduledCashFlow(
        'test', CashSource('test source'), CashSink('test sink'),
        FloatDistribution(min=10, max=100, mean=20),
        Monthly(15, period=1))
    cash_flow2 = ScheduledCashFlow(
        'test', CashSource('test source'), CashSink('test sink'),
        FloatDistribution(min=30, max=300, mean=60),
        Monthly(15, period=3, range=DateRange.beginning_at(date(2023, 1, 1))))
    result1 = summarise_total_cash_flow(
        cash_flow1,
        DateRange.inclusive(date(2023, 1, 1), date(2025, 1, 1)))
    result2 = summarise_total_cash_flow(
        cash_flow2,
        DateRange.inclusive(date(2023, 1, 1), date(2025, 1, 1)))
    assert result1.approx_eq(result2)


def test_cash_flow_log_lt() -> None:
    # Note only date and bound_type are significant.

    assert (CashFlowLog(date(2023, 1, 1), -1, False, None, None, None)
          < CashFlowLog(date(2024, 2, 2), -1, False, None, None, None))
    assert (CashFlowLog(date(2023, 1, 1), 0, False, None, None, None)
          < CashFlowLog(date(2024, 2, 2), 0, False, None, None, None))
    assert (CashFlowLog(date(2023, 1, 1), 1, False, None, None, None)
          < CashFlowLog(date(2024, 2, 2), 1, False, None, None, None))
    assert (CashFlowLog(date(2023, 1, 1), -1, False, None, None, None)
          < CashFlowLog(date(2023, 1, 1), 0, False, None, None, None))
    assert (CashFlowLog(date(2023, 1, 1), -1, False, None, None, None)
          < CashFlowLog(date(2023, 1, 1), 1, False, None, None, None))
    assert (CashFlowLog(date(2023, 1, 1), 0, False, None, None, None)
          < CashFlowLog(date(2023, 1, 1), 1, False, None, None, None))

    assert not (CashFlowLog(date(2023, 2, 2), -1, False, None, None, None)
              < CashFlowLog(date(2022, 1, 1), -1, False, None, None, None))
    assert not (CashFlowLog(date(2023, 2, 2), 0, False, None, None, None)
              < CashFlowLog(date(2022, 1, 1), 0, False, None, None, None))
    assert not (CashFlowLog(date(2023, 2, 2), 1, False, None, None, None)
              < CashFlowLog(date(2022, 1, 1), 1, False, None, None, None))
    assert not (CashFlowLog(date(2023, 2, 2), 0, False, None, None, None)
              < CashFlowLog(date(2023, 1, 1), -1, False, None, None, None))
    assert not (CashFlowLog(date(2023, 1, 1), 1, False, None, None, None)
              < CashFlowLog(date(2023, 1, 1), -1, False, None, None, None))
    assert not (CashFlowLog(date(2023, 1, 1), 1, False, None, None, None)
              < CashFlowLog(date(2023, 1, 1), 0, False, None, None, None))

    assert not (CashFlowLog(date(2023, 1, 1), -1, False, None, None, None)
              < CashFlowLog(date(2023, 1, 1), -1, False, None, None, None))
    assert not (CashFlowLog(date(2023, 1, 1), 0, False, None, None, None)
              < CashFlowLog(date(2023, 1, 1), 0, False, None, None, None))
    assert not (CashFlowLog(date(2023, 1, 1), 1, False, None, None, None)
              < CashFlowLog(date(2023, 1, 1), 1, False, None, None, None))

def test_cash_flow_log_str_lower_bound_exact() -> None:
    c = CashFlowLog(
        date(2023, 1, 26), -1, True,
        FloatDistribution(min=1.234, mean=2.345, max=3.456),
        CashEndpoint('test endpoint1'), CashEndpoint('test endpoint2'))
    assert str(c) == '2023-01-26 vv | $[1.23, (2.35), 3.46] from "test endpoint1" to "test endpoint2"'

def test_cash_flow_log_str_lower_bound_inexact() -> None:
    c = CashFlowLog(
        date(2023, 1, 26), -1, False,
        FloatDistribution(min=1.234, mean=2.345, max=3.456),
        CashEndpoint('test endpoint1'), CashEndpoint('test endpoint2'))
    assert str(c) == '2023-01-26 ~v | $[1.23, (2.35), 3.46] from "test endpoint1" to "test endpoint2"'

def test_cash_flow_log_str_upper_bound_exact() -> None:
    c = CashFlowLog(
        date(2023, 1, 26), 1, True,
        FloatDistribution(min=1.234, mean=2.345, max=3.456),
        CashEndpoint('test endpoint1'), CashEndpoint('test endpoint2'))
    assert str(c) == '2023-01-26 ^^ | $[1.23, (2.35), 3.46] from "test endpoint1" to "test endpoint2"'

def test_cash_flow_log_str_upper_bound_inexact() -> None:
    c = CashFlowLog(
        date(2023, 1, 26), 1, False,
        FloatDistribution(min=1.234, mean=2.345, max=3.456),
        CashEndpoint('test endpoint1'), CashEndpoint('test endpoint2'))
    assert str(c) == '2023-01-26 ~^ | $[1.23, (2.35), 3.46] from "test endpoint1" to "test endpoint2"'

def test_cash_flow_log_str_both_bounds_exact() -> None:
    c = CashFlowLog(
        date(2023, 1, 26), 0, True,
        FloatDistribution(min=1.234, mean=2.345, max=3.456),
        CashEndpoint('test endpoint1'), CashEndpoint('test endpoint2'))
    assert str(c) == '2023-01-26 == | $[1.23, (2.35), 3.46] from "test endpoint1" to "test endpoint2"'

def test_cash_flow_log_str_both_bounds_inexact() -> None:
    c = CashFlowLog(
        date(2023, 1, 26), 0, False,
        FloatDistribution(min=1.234, mean=2.345, max=3.456),
        CashEndpoint('test endpoint1'), CashEndpoint('test endpoint2'))
    assert str(c) == '2023-01-26 ~~ | $[1.23, (2.35), 3.46] from "test endpoint1" to "test endpoint2"'

def test_generate_cash_flow_logs_empty_range() -> None:
    cash_flow = ScheduledCashFlow(
        'test', CashSource('test source'), CashSink('test sink'),
        FloatDistribution.singular(10),
        Once(date(2023, 1, 1)))
    result = tuple(generate_cash_flow_logs(cash_flow, DateRange.empty()))
    assert result == ()

def test_generate_cash_flow_logs_no_events() -> None:
    cash_flow = ScheduledCashFlow(
        'test', CashSource('test source'), CashSink('test sink'),
        FloatDistribution.singular(10),
        Once(date(2023, 1, 1)))
    result = tuple(generate_cash_flow_logs(cash_flow, DateRange.inclusive(date(2020, 1, 1), date(2022, 1, 1))))
    assert result == ()

def test_generate_cash_flow_logs_certain_event() -> None:
    source = CashSource('test source')
    sink = CashSink('test sink')
    amount = FloatDistribution(min=10, mean=15, max=20)
    cash_flow = ScheduledCashFlow('test', source, sink, amount, Monthly(10))
    result = tuple(generate_cash_flow_logs(cash_flow, DateRange.inclusive(date(2022, 1, 1), date(2022, 6, 1))))
    assert result == (
        CashFlowLog(date(2022, 1, 10), 0, True, amount, source, sink),
        CashFlowLog(date(2022, 2, 10), 0, True, amount, source, sink),
        CashFlowLog(date(2022, 3, 10), 0, True, amount, source, sink),
        CashFlowLog(date(2022, 4, 10), 0, True, amount, source, sink),
        CashFlowLog(date(2022, 5, 10), 0, True, amount, source, sink))

def test_generate_cash_flow_logs_uncertain_event() -> None:
    source = CashSource('test source')
    sink = CashSink('test sink')
    amount = FloatDistribution(min=10, mean=15, max=20)
    schedule = Monthly(SimpleDayOfMonthSchedule((
        DayOfMonthDistribution.from_probabilities({10: 0.1, 13: 0.3, 23: 0.2}),
        DayOfMonthDistribution.from_weights({3: 2, 16: 1, 18: 3})
    )))
    cash_flow = ScheduledCashFlow('test', source, sink, amount, schedule)
    result = tuple(generate_cash_flow_logs(cash_flow, DateRange.inclusive(date(2022, 1, 15), date(2022, 3, 15))))
    assert result == (
        # 2022/1
        CashFlowLog(date(2022, 1, 16), -1, False, amount, source, sink),    # Event 2
        CashFlowLog(date(2022, 1, 18), 1, True, amount, source, sink),      # Event 2
        CashFlowLog(date(2022, 1, 23), 0, False, amount, source, sink),     # Event 1
        # 2022/2
        CashFlowLog(date(2022, 2, 3), -1, True, amount, source, sink),      # Event 2
        CashFlowLog(date(2022, 2, 10), -1, True, amount, source, sink),     # Event 1
        CashFlowLog(date(2022, 2, 18), 1, True, amount, source, sink),      # Event 2
        CashFlowLog(date(2022, 2, 23), 1, False, amount, source, sink),     # Event 1
        # 2022/3
        CashFlowLog(date(2022, 3, 3), 0, False, amount, source, sink),      # Event 2
        CashFlowLog(date(2022, 3, 10), -1, True , amount, source, sink),    # Event 1
        CashFlowLog(date(2022, 3, 13), 1, False, amount, source, sink))     # Event 1
