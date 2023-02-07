from datetime import date
from random import betavariate, randint, random

from pytest import approx, raises

from cashflow.cash_flow import (
    CashBalanceDelta, CashBalanceRecord, CashBalanceUpdate, CashEndpoint, CashFlowLog, CashSink, CashSource,
    ScheduledCashFlow, accumulate_endpoint_balances, generate_balance_updates, generate_cash_flow_logs,
    simulate_cash_balances, summarise_total_cash_flow)
from cashflow.date_time import DateRange, DayOfMonthNumeral
from cashflow.probability import FloatDistribution
from cashflow.schedule import DateDistribution, DayOfMonthDistribution, Monthly, Once, SimpleDayOfMonthSchedule, Weekly

from .helpers import approx_floats


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


def test_generate_balance_updates_empty_range() -> None:
    cash_flow = ScheduledCashFlow(
        'schedule', CashSource('source'), CashSink('sink'),
        FloatDistribution.singular(2), Weekly(3))
    result = tuple(generate_balance_updates(cash_flow, DateRange.empty()))
    assert result == ()

def test_generate_balance_updates_no_occurrences() -> None:
    cash_flow = ScheduledCashFlow(
        'schedule', CashSource('source'), CashSink('sink'),
        FloatDistribution.singular(2), Once(date(2023, 4, 10)))
    result = tuple(generate_balance_updates(cash_flow, DateRange.up_to(date(2023, 3, 1))))
    assert result == ()

def test_generate_balance_updates_certain_occurrences() -> None:
    source = CashSource('source')
    sink = CashSink('sink')
    cash_flow = ScheduledCashFlow(
        'schedule', source, sink,
        FloatDistribution(min=10, mean=11, max=25),
        Monthly(day=6))
    date_range = DateRange.inclusive(date(2023, 6, 6), date(2023, 8, 6))
    result = tuple(generate_balance_updates(cash_flow, date_range))
    assert result == (
        CashBalanceUpdate(date(2023, 6, 6), source, CashBalanceDelta(min=-25), cash_flow),
        CashBalanceUpdate(date(2023, 6, 6), sink, CashBalanceDelta(max=25), cash_flow),
        CashBalanceUpdate(date(2023, 6, 7), source, CashBalanceDelta(mean=-11), cash_flow),
        CashBalanceUpdate(date(2023, 6, 7), sink, CashBalanceDelta(mean=11), cash_flow),
        CashBalanceUpdate(date(2023, 6, 7), source, CashBalanceDelta(max=-10), cash_flow),
        CashBalanceUpdate(date(2023, 6, 7), sink, CashBalanceDelta(min=10), cash_flow),

        CashBalanceUpdate(date(2023, 7, 6), source, CashBalanceDelta(min=-25), cash_flow),
        CashBalanceUpdate(date(2023, 7, 6), sink, CashBalanceDelta(max=25), cash_flow),
        CashBalanceUpdate(date(2023, 7, 7), source, CashBalanceDelta(mean=-11), cash_flow),
        CashBalanceUpdate(date(2023, 7, 7), sink, CashBalanceDelta(mean=11), cash_flow),
        CashBalanceUpdate(date(2023, 7, 7), source, CashBalanceDelta(max=-10), cash_flow),
        CashBalanceUpdate(date(2023, 7, 7), sink, CashBalanceDelta(min=10), cash_flow),

        CashBalanceUpdate(date(2023, 8, 6), source, CashBalanceDelta(min=-25), cash_flow),
        CashBalanceUpdate(date(2023, 8, 6), sink, CashBalanceDelta(max=25), cash_flow),
        CashBalanceUpdate(date(2023, 8, 7), source, CashBalanceDelta(mean=-11), cash_flow),
        CashBalanceUpdate(date(2023, 8, 7), sink, CashBalanceDelta(mean=11), cash_flow),
        CashBalanceUpdate(date(2023, 8, 7), source, CashBalanceDelta(max=-10), cash_flow),
        CashBalanceUpdate(date(2023, 8, 7), sink, CashBalanceDelta(min=10), cash_flow)
    )

def test_generate_balance_updates_effectively_certain_occurrences() -> None:
    source = CashSource('source')
    sink = CashSink('sink')
    schedule = Monthly(day=SimpleDayOfMonthSchedule((DayOfMonthDistribution.from_probabilities({
        6: 0.99999999
    }),)))
    cash_flow = ScheduledCashFlow(
        'schedule', source, sink,
        FloatDistribution(min=10, mean=11, max=25),
        schedule)
    date_range = DateRange.inclusive(date(2023, 6, 6), date(2023, 8, 6))
    result = tuple(generate_balance_updates(cash_flow, date_range))
    assert result == (
        CashBalanceUpdate(date(2023, 6, 6), source, CashBalanceDelta(min=-25), cash_flow),
        CashBalanceUpdate(date(2023, 6, 6), sink, CashBalanceDelta(max=25), cash_flow),
        CashBalanceUpdate(date(2023, 6, 7), source, CashBalanceDelta(mean=-11), cash_flow),
        CashBalanceUpdate(date(2023, 6, 7), sink, CashBalanceDelta(mean=11), cash_flow),
        CashBalanceUpdate(date(2023, 6, 7), source, CashBalanceDelta(max=-10), cash_flow),
        CashBalanceUpdate(date(2023, 6, 7), sink, CashBalanceDelta(min=10), cash_flow),

        CashBalanceUpdate(date(2023, 7, 6), source, CashBalanceDelta(min=-25), cash_flow),
        CashBalanceUpdate(date(2023, 7, 6), sink, CashBalanceDelta(max=25), cash_flow),
        CashBalanceUpdate(date(2023, 7, 7), source, CashBalanceDelta(mean=-11), cash_flow),
        CashBalanceUpdate(date(2023, 7, 7), sink, CashBalanceDelta(mean=11), cash_flow),
        CashBalanceUpdate(date(2023, 7, 7), source, CashBalanceDelta(max=-10), cash_flow),
        CashBalanceUpdate(date(2023, 7, 7), sink, CashBalanceDelta(min=10), cash_flow),

        CashBalanceUpdate(date(2023, 8, 6), source, CashBalanceDelta(min=-25), cash_flow),
        CashBalanceUpdate(date(2023, 8, 6), sink, CashBalanceDelta(max=25), cash_flow),
        CashBalanceUpdate(date(2023, 8, 7), source, CashBalanceDelta(mean=-11), cash_flow),
        CashBalanceUpdate(date(2023, 8, 7), sink, CashBalanceDelta(mean=11), cash_flow),
        CashBalanceUpdate(date(2023, 8, 7), source, CashBalanceDelta(max=-10), cash_flow),
        CashBalanceUpdate(date(2023, 8, 7), sink, CashBalanceDelta(min=10), cash_flow)
    )

def test_generate_balance_updates_uncertain_occurrences() -> None:
    source = CashSource('source')
    sink = CashSink('sink')
    schedule = Monthly(day=SimpleDayOfMonthSchedule((DayOfMonthDistribution.from_probabilities({
        6: 0.5,
        10: 0.2
    }),)))
    cash_flow = ScheduledCashFlow(
        'schedule', source, sink,
        FloatDistribution(min=10, mean=11, max=25),
        schedule)
    date_range = DateRange.inclusive(date(2023, 6, 6), date(2023, 8, 6))
    result = tuple(generate_balance_updates(cash_flow, date_range))
    assert result == approx_floats((
        CashBalanceUpdate(date(2023, 6, 6), source, CashBalanceDelta(min=-25), cash_flow),
        CashBalanceUpdate(date(2023, 6, 6), sink, CashBalanceDelta(max=25), cash_flow),
        CashBalanceUpdate(date(2023, 6, 7), source, CashBalanceDelta(mean=-5.5), cash_flow),
        CashBalanceUpdate(date(2023, 6, 7), sink, CashBalanceDelta(mean=5.5), cash_flow),
        CashBalanceUpdate(date(2023, 6, 11), source, CashBalanceDelta(mean=-2.2), cash_flow),
        CashBalanceUpdate(date(2023, 6, 11), sink, CashBalanceDelta(mean=2.2), cash_flow),
        # No max update because events are not certain.

        CashBalanceUpdate(date(2023, 7, 6), source, CashBalanceDelta(min=-25), cash_flow),
        CashBalanceUpdate(date(2023, 7, 6), sink, CashBalanceDelta(max=25), cash_flow),
        CashBalanceUpdate(date(2023, 7, 7), source, CashBalanceDelta(mean=-5.5), cash_flow),
        CashBalanceUpdate(date(2023, 7, 7), sink, CashBalanceDelta(mean=5.5), cash_flow),
        CashBalanceUpdate(date(2023, 7, 11), source, CashBalanceDelta(mean=-2.2), cash_flow),
        CashBalanceUpdate(date(2023, 7, 11), sink, CashBalanceDelta(mean=2.2), cash_flow),
        # No max update because events are not certain.

        CashBalanceUpdate(date(2023, 8, 6), source, CashBalanceDelta(min=-25), cash_flow),
        CashBalanceUpdate(date(2023, 8, 6), sink, CashBalanceDelta(max=25), cash_flow),
        CashBalanceUpdate(date(2023, 8, 7), source, CashBalanceDelta(mean=-5.5), cash_flow),
        CashBalanceUpdate(date(2023, 8, 7), sink, CashBalanceDelta(mean=5.5), cash_flow)
        # Last 2 mean updates are outside range.
    ))

def test_generate_balance_updates_likely_before_range() -> None:
    source = CashSource('source')
    sink = CashSink('sink')
    schedule = Monthly(day=SimpleDayOfMonthSchedule((DayOfMonthDistribution.uniformly_of(1, 2, 3),)))
    cash_flow = ScheduledCashFlow(
        'schedule', source, sink,
        FloatDistribution.singular(12.33),
        schedule)
    date_range = DateRange.half_open(date(2023, 1, 3), date(2023, 2, 1))
    updates = tuple(generate_balance_updates(cash_flow, date_range))
    assert updates == approx_floats((
        CashBalanceUpdate(date(2023, 1, 3), source, CashBalanceDelta(min=-12.33), cash_flow),
        CashBalanceUpdate(date(2023, 1, 3), sink, CashBalanceDelta(max=12.33), cash_flow),
        CashBalanceUpdate(date(2023, 1, 4), source, CashBalanceDelta(mean=-4.11), cash_flow),
        CashBalanceUpdate(date(2023, 1, 4), sink, CashBalanceDelta(mean=4.11), cash_flow),
        CashBalanceUpdate(date(2023, 1, 4), source, CashBalanceDelta(max=-4.11), cash_flow),
        CashBalanceUpdate(date(2023, 1, 4), sink, CashBalanceDelta(min=4.11), cash_flow)
    ))


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
    assert result == approx_floats({
        endpoint1: [
            CashBalanceRecord(date(2023, 1, 16), FloatDistribution(min=4, max=45.2, mean=20.3)),
            CashBalanceRecord(date(2023, 1, 17), FloatDistribution(min=4, max=45.2, mean=32.8)),
            CashBalanceRecord(date(2023, 1, 23), FloatDistribution(min=4, max=47.1, mean=38.8))
        ],
        endpoint2: [
            CashBalanceRecord(date(2023, 1, 20), FloatDistribution(min=78, max=89, mean=79.6)),
            CashBalanceRecord(date(2023, 1, 23), FloatDistribution(min=78, max=87.7, mean=79.6))
        ]
    })


def test_simulate_cash_balances_empty_range_no_initial_balances() -> None:
    result = simulate_cash_balances((), DateRange.empty(), {})
    assert len(result) == 0

def test_simulate_cash_balances_empty_range_with_initial_balances() -> None:
    initial_balances = {CashEndpoint('foo'): FloatDistribution.singular(4.5)}
    result = simulate_cash_balances((), DateRange.empty(), initial_balances)
    assert len(result) == 0

def test_simulate_cash_balances_no_occurrences_no_initial_balances() -> None:
    cash_flows = (ScheduledCashFlow('foo', CashSource('bar'), CashSink('qux'),
        FloatDistribution.singular(3), Once(date(2022, 8, 12))),)
    result = simulate_cash_balances(cash_flows, DateRange.beginning_at(date(2023, 1, 1)))
    assert len(result) == 0

def test_simulate_cash_balances_no_occurrences_with_initial_balances() -> None:
    source = CashSource('bar')
    sink = CashSink('qux')
    cash_flows = (ScheduledCashFlow('foo', source, sink,
        FloatDistribution.singular(3), Once(date(2022, 8, 12))),)
    date_range = DateRange.inclusive(date(2023, 1, 1), date(2024, 1, 1))
    initial_balances = {source: FloatDistribution.singular(15), sink: FloatDistribution.singular(6)}
    result = simulate_cash_balances(cash_flows, date_range, initial_balances)
    assert result == {
        source: [
            CashBalanceRecord(date(2023, 1, 1), FloatDistribution.singular(15)),
            CashBalanceRecord(date(2024, 1, 2), FloatDistribution.singular(15)),
        ],
        sink: [
            CashBalanceRecord(date(2023, 1, 1), FloatDistribution.singular(6)),
            CashBalanceRecord(date(2024, 1, 2), FloatDistribution.singular(6)),
        ]
    }

def test_simulate_cash_balances_some_occurrences() -> None:
    source1 = CashSource('source1')
    sink = CashSink('sink')
    source2 = CashSource('source2')
    cash_flows = (
        ScheduledCashFlow('flow1', source1, sink, FloatDistribution(min=10, mean=20, max=40), Monthly(day=2)),
        ScheduledCashFlow('flow2', source2, sink, FloatDistribution(min=2, mean=4, max=5),
            Monthly(day=SimpleDayOfMonthSchedule((DayOfMonthDistribution.from_probabilities({3: 0.25, 10: 0.5}),))))
    )
    date_range = DateRange.inclusive(date(2023, 1, 3), date(2023, 4, 2))
    initial_balances = {source1: FloatDistribution.singular(10), source2: FloatDistribution.singular(-10)}
    result = simulate_cash_balances(cash_flows, date_range, initial_balances)
    assert result == {
        source1: [
            # 2023/1
            CashBalanceRecord(date(2023, 1, 3), FloatDistribution.singular(10)),
            # 2023/2
            CashBalanceRecord(date(2023, 2, 2), FloatDistribution(min=-30, mean=10, max=10)),
            CashBalanceRecord(date(2023, 2, 3), FloatDistribution(min=-30, mean=-10, max=0)),
            # 2023/3
            CashBalanceRecord(date(2023, 3, 2), FloatDistribution(min=-70, mean=-10, max=0)),
            CashBalanceRecord(date(2023, 3, 3), FloatDistribution(min=-70, mean=-30, max=-10)),
            # 2023/4
            CashBalanceRecord(date(2023, 4, 2), FloatDistribution(min=-110, mean=-30, max=-10)),
            CashBalanceRecord(date(2023, 4, 3), FloatDistribution(min=-110, mean=-50, max=-20))
        ],
        source2: [
            # 2023/1
            CashBalanceRecord(date(2023, 1, 3), FloatDistribution(min=-15, mean=-10, max=-10)),
            CashBalanceRecord(date(2023, 1, 4), FloatDistribution(min=-15, mean=-11, max=-10)),
            CashBalanceRecord(date(2023, 1, 11), FloatDistribution(min=-15, mean=-13, max=-10)),
            # 2023/2
            CashBalanceRecord(date(2023, 2, 3), FloatDistribution(min=-20, mean=-13, max=-10)),
            CashBalanceRecord(date(2023, 2, 4), FloatDistribution(min=-20, mean=-14, max=-10)),
            CashBalanceRecord(date(2023, 2, 11), FloatDistribution(min=-20, mean=-16, max=-10)),
            # 2023/3
            CashBalanceRecord(date(2023, 3, 3), FloatDistribution(min=-25, mean=-16, max=-10)),
            CashBalanceRecord(date(2023, 3, 4), FloatDistribution(min=-25, mean=-17, max=-10)),
            CashBalanceRecord(date(2023, 3, 11), FloatDistribution(min=-25, mean=-19, max=-10)),
            # 2023/4
            CashBalanceRecord(date(2023, 4, 3), FloatDistribution(min=-25, mean=-19, max=-10))
        ],
        sink: [
            # 2023/1
            CashBalanceRecord(date(2023, 1, 3), FloatDistribution(min=0, mean=0, max=5)),
            CashBalanceRecord(date(2023, 1, 4), FloatDistribution(min=0, mean=1, max=5)),
            CashBalanceRecord(date(2023, 1, 11), FloatDistribution(min=0, mean=3, max=5)),
            # 2023/2
            CashBalanceRecord(date(2023, 2, 2), FloatDistribution(min=0, mean=3, max=45)),
            CashBalanceRecord(date(2023, 2, 3), FloatDistribution(min=10, mean=23, max=50)),
            CashBalanceRecord(date(2023, 2, 4), FloatDistribution(min=10, mean=24, max=50)),
            CashBalanceRecord(date(2023, 2, 11), FloatDistribution(min=10, mean=26, max=50)),
            # 2023/3
            CashBalanceRecord(date(2023, 3, 2), FloatDistribution(min=10, mean=26, max=90)),
            CashBalanceRecord(date(2023, 3, 3), FloatDistribution(min=20, mean=46, max=95)),
            CashBalanceRecord(date(2023, 3, 4), FloatDistribution(min=20, mean=47, max=95)),
            CashBalanceRecord(date(2023, 3, 11), FloatDistribution(min=20, mean=49, max=95)),
            # 2023/4
            CashBalanceRecord(date(2023, 4, 2), FloatDistribution(min=20, mean=49, max=135)),
            CashBalanceRecord(date(2023, 4, 3), FloatDistribution(min=30, mean=69, max=135)),
        ]
    }

def test_simulate_cash_balances_fuzz() -> None:
    source = CashSource('source')
    sink = CashSink('sink')
    for _ in range(5000):
        probabilities: dict[DayOfMonthNumeral, float] = {}
        equal_probabilities = random() <= 0.5
        distribution_certain = random() <= 0.5
        probability = betavariate(0.5, 0.5)
        probability_sum = 0
        for _ in range(randint(1, 31)):
            probability = probability if equal_probabilities else betavariate(0.5, 0.5)
            if probability > 0:
                if not distribution_certain and probability_sum + probability >= 0.999:
                    break
                probabilities[randint(1, 31)] = probability
                probability_sum += probability
        if distribution_certain:
            distribution = DayOfMonthDistribution.from_weights(probabilities)
        else:
            distribution = DayOfMonthDistribution.from_probabilities(probabilities)
        schedule = Monthly(day=SimpleDayOfMonthSchedule((distribution,)))

        exact_amount = random() <= 0.5
        min_amount = betavariate(0.5, 0.5) * 1000
        mean_amount = min_amount if exact_amount else min_amount + betavariate(0.5, 0.5) * 1000
        max_amount = min_amount if exact_amount else mean_amount + betavariate(0.5, 0.5) * 1000
        amount = FloatDistribution(min=min_amount, mean=mean_amount, max=max_amount)

        cash_flow = ScheduledCashFlow('schedule', source, sink, amount, schedule)

        start_day = randint(1, 31)
        end_day = randint(start_day, 31)
        date_range = DateRange.inclusive(date(2023, 1, start_day), date(2023, 1, end_day))

        simulate_cash_balances((cash_flow,), date_range)


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
    assert result == approx_floats(FloatDistribution(min=10, max=34, mean=24.9))

def test_summarise_cash_flow_effectively_certain_occurrence() -> None:
    schedule = Once(DateDistribution.from_probabilities({
        date(2023, 1, 1): 0.99999999999
    }))
    cash_flow = ScheduledCashFlow(
        'test', CashSource('test source'), CashSink('test sink'),
        FloatDistribution(min=10, max=34, mean=24.9),
        schedule)
    result = summarise_total_cash_flow(
        cash_flow,
        DateRange.half_open(date(2022, 1, 1), date(2024, 1, 1)))
    assert result == approx_floats(FloatDistribution(min=10, max=34, mean=24.9))

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
    assert result == approx_floats(FloatDistribution(min=0, max=100, mean=8))

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
    assert result == approx_floats(FloatDistribution(min=10, max=500, mean=48))

def test_summarise_cash_flow_sanity_check() -> None:
    cash_flow1 = ScheduledCashFlow(
        'test', CashSource('test source'), CashSink('test sink'),
        FloatDistribution(min=10, max=100, mean=20),
        Monthly(day=15, period=1))
    cash_flow2 = ScheduledCashFlow(
        'test', CashSource('test source'), CashSink('test sink'),
        FloatDistribution(min=30, max=300, mean=60),
        Monthly(day=15, period=3, range=DateRange.beginning_at(date(2023, 1, 1))))
    result1 = summarise_total_cash_flow(
        cash_flow1,
        DateRange.inclusive(date(2023, 1, 1), date(2025, 1, 1)))
    result2 = summarise_total_cash_flow(
        cash_flow2,
        DateRange.inclusive(date(2023, 1, 1), date(2025, 1, 1)))
    assert result1 == approx_floats(result2)


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
    cash_flow = ScheduledCashFlow('test', source, sink, amount, Monthly(day=10))
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
