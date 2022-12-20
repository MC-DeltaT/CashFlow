from pytest import raises

from cashflow.probability import FloatDistribution, effectively_certain


def test_effectively_certain_within_tolerance() -> None:
    assert effectively_certain(0.999, tolerance=1e-3)

def test_effectively_certain_outside_tolerance() -> None:
    assert not effectively_certain(0.999, tolerance=1e-4)

def test_effectively_certain_exact() -> None:
    assert effectively_certain(1, tolerance=0)

def test_effectively_certain_negative_tolerance() -> None:
    with raises(ValueError):
        effectively_certain(1, tolerance=-1e-9)


# TODO: test DiscreteDistribution


def test_float_distribution_construct_invalid1() -> None:
    with raises(ValueError):
        FloatDistribution(min=10, max=9, mean=15)

def test_float_distribution_construct_invalid2() -> None:
    with raises(ValueError):
        FloatDistribution(min=10, max=13, mean=15)

def test_float_distribution_construct_invalid3() -> None:
    with raises(ValueError):
        FloatDistribution(min=15, max=13, mean=10)

def test_float_distribution_construct_invalid4() -> None:
    with raises(ValueError):
        FloatDistribution(min=15, max=16, mean=10)

def test_float_distribution_construct_invalid5() -> None:
    with raises(ValueError):
        FloatDistribution(min=15, max=13, mean=10)

def test_float_distribution_singular() -> None:
    d = FloatDistribution.singular(23.456)
    assert d.min == 23.456
    assert d.max == 23.456
    assert d.mean == 23.456

def test_float_distribution_uniformly_in() -> None:
    d = FloatDistribution.uniformly_in(-0.25, 2)
    assert d.min == -0.25
    assert d.max == 2
    assert d.mean == 0.875

def test_float_distribution_uniformly_around() -> None:
    d = FloatDistribution.uniformly_around(10, 4)
    assert d.min == 6
    assert d.max == 14
    assert d.mean == 10

def test_float_distribution_to_str_singular() -> None:
    d = FloatDistribution.singular(123.456789)
    assert d.to_str(4) == '123.4568'

def test_float_distribution_to_str_nonsingular() -> None:
    d = FloatDistribution(min=-1.25, mean=3.4444444444, max=5.654)
    assert d.to_str(3) == '[-1.250, (3.444), 5.654]'

def test_float_distribution_neg() -> None:
    d = FloatDistribution(min=-1, max=3, mean=0.5)
    assert -d == FloatDistribution(min=-3, max=1, mean=-0.5)

def test_float_distribution_add_scalar() -> None:
    d = FloatDistribution(min=-1, max=2, mean=0.7)
    assert d + 17 == FloatDistribution(min=16, max=19, mean=17.7)

def test_float_distribution_add_distribution() -> None:
    d1 = FloatDistribution(min=-100, max=200, mean=123)
    d2 = FloatDistribution(min=-1, max=330, mean=-0.5)
    assert d1 + d2 == FloatDistribution(min=-101, max=530, mean=122.5)

def test_float_distribution_radd_scalar() -> None:
    d = FloatDistribution(min=-1, max=2, mean=0.7)
    assert 17 + d == FloatDistribution(min=16, max=19, mean=17.7)

def test_float_distribution_mul_scalar_positive() -> None:
    d = FloatDistribution(min=-10, max=11, mean=3.1)
    assert d * 1.5 == FloatDistribution(min=-15, max=16.5, mean=4.65)

def test_float_distribution_mul_scalar_negative() -> None:
    d = FloatDistribution(min=-10, max=11, mean=3.1)
    assert d * -1.5 == FloatDistribution(min=-16.5, max=15, mean=-4.65)

def test_float_distribution_rmul_scalar() -> None:
    d = FloatDistribution(min=-10, max=11, mean=3.1)
    assert 1.5 * d == FloatDistribution(min=-15, max=16.5, mean=4.65)
