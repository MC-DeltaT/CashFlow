from cashflow.cash_flow import CashBalance, CashBalanceDelta

from pytest import approx


def test_cash_balance_delta_add() -> None:
    result = CashBalanceDelta(min=-10.2, max=53.83, mean=16.36) + CashBalanceDelta(min=4.18, max=-12.03, mean=19.77)
    assert result.min == approx(-6.02)
    assert result.max == approx(41.8)
    assert result.mean == approx(36.13)


def test_cash_balance_exactly() -> None:
    result = CashBalance.exactly(36.809)
    assert result.min == 36.809
    assert result.max == 36.809
    assert result.mean == 36.809

def test_cash_balance_add() -> None:
    result = CashBalance(min=-10.2, max=53.83, mean=16.36) + CashBalanceDelta(min=-4.18, max=-12.03, mean=19.77)
    assert result.min == approx(-14.38)
    assert result.max == approx(41.8)
    assert result.mean == approx(36.13)


# TODO
