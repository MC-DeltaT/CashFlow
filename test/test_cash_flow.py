from datetime import date

from pytest import approx, raises

from cashflow.cash_flow import CashBalanceDelta, CashBalanceUpdate, CashEndpoint, accumulate_endpoint_balances
from cashflow.probability import FloatDistribution


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


# TODO
