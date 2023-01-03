from pytest import approx, raises

from cashflow.probability import DiscreteDistribution, DiscreteOutcome, FloatDistribution, effectively_certain


def test_effectively_certain_within_tolerance() -> None:
    assert effectively_certain(0.999, tolerance=1e-3)

def test_effectively_certain_outside_tolerance() -> None:
    assert not effectively_certain(0.999, tolerance=1e-4)

def test_effectively_certain_exact() -> None:
    assert effectively_certain(1, tolerance=0)

def test_effectively_certain_negative_tolerance() -> None:
    with raises(ValueError):
        effectively_certain(1, tolerance=-1e-9)


def test_discrete_outcome_construct_invalid_probability() -> None:
    with raises(ValueError):
        DiscreteOutcome(1, -0.000000001, 0.3)
    with raises(ValueError):
        DiscreteOutcome(1, 1.00000001, 1)

def test_discrete_outcome_construct_invalid_cumulative() -> None:
    with raises(ValueError):
        DiscreteOutcome(1, 0.5, 1.0000001)
    with raises(ValueError):
        DiscreteOutcome(1, 0.4, -0.5)

def test_discrete_distribution_construct_invalid_order() -> None:
    with raises(ValueError):
        DiscreteDistribution([
            DiscreteOutcome(1, 0.4, 0.4),
            DiscreteOutcome(2, 0.4, 0.8),
            DiscreteOutcome(1.5, 0.1, 1)
        ])

def test_discrete_distribution_construct_invalid_cumulative() -> None:
    with raises(ValueError):
        DiscreteDistribution([
            DiscreteOutcome(1, 0.4, 0.4),
            DiscreteOutcome(2, 0.3, 0.5),
            DiscreteOutcome(3, 0.1, 0.8)
        ])

def test_discrete_distribution_construct_invalid_sum() -> None:
    with raises(ValueError):
        DiscreteDistribution([
            DiscreteOutcome(1, 0.4, 0.4),
            DiscreteOutcome(2, 0.6, 1),
            DiscreteOutcome(3, 0.1, 1.1),
            DiscreteOutcome(4, 0.1, 1.2)
        ])

def test_discrete_distribution_from_weights_valid() -> None:
    d = DiscreteDistribution.from_weights({1: 1, 2: 3, 5: 2})
    assert [o.value for o in d.outcomes] == [1, 2, 5]
    assert [o.probability for o in d.outcomes] == approx([1/6, 3/6, 2/6])
    assert [o.cumulative_probability for o in d.outcomes] == approx([1/6, 4/6, 6/6])
    assert d.outcomes[-1].cumulative_probability == 1

def test_discrete_distribution_from_weights_negative_weights() -> None:
    with raises(ValueError):
        DiscreteDistribution.from_weights({1: 2, 3: 4, 5: -1})

def test_discrete_distribution_from_probabilities_sum_not_1() -> None:
    d = DiscreteDistribution.from_probabilities({1: 1/7, 2: 3/7, 5: 2/7})
    assert [o.value for o in d.outcomes] == [1, 2, 5]
    assert [o.probability for o in d.outcomes] == approx([1/7, 3/7, 2/7])
    assert [o.cumulative_probability for o in d.outcomes] == approx([1/7, 4/7, 6/7])

def test_discrete_distribution_from_probabilities_sum_1() -> None:
    d = DiscreteDistribution.from_probabilities({1: 1/7, 2: 3/7, 6: 1/7, 5: 2/7})
    assert [o.value for o in d.outcomes] == [1, 2, 5, 6]
    assert [o.probability for o in d.outcomes] == approx([1/7, 3/7, 2/7, 1/7])
    assert [o.cumulative_probability for o in d.outcomes] == approx([1/7, 4/7, 6/7, 7/7])
    assert d.outcomes[-1].cumulative_probability == 1

def test_discrete_distribution_from_probabilities_negative_probability() -> None:
    with raises(ValueError):
        DiscreteDistribution.from_probabilities({1: 0.2, 3: 0.4, 5: -0.0002})

def test_discrete_distribution_singular() -> None:
    d = DiscreteDistribution.singular(7.8)
    expected = DiscreteDistribution((DiscreteOutcome(7.8, 1, 1),))
    assert d == expected

def test_discrete_distribution_uniformly_in_nonempty() -> None:
    d = DiscreteDistribution.uniformly_in((5, 1, 2))
    assert [o.value for o in d.outcomes] == [1, 2, 5]
    assert [o.probability for o in d.outcomes] == approx([1/3, 1/3, 1/3])
    assert [o.cumulative_probability for o in d.outcomes] == approx([1/3, 2/3, 3/3])
    assert d.outcomes[-1].cumulative_probability == 1

def test_discrete_distribution_uniformly_in_empty() -> None:
    d = DiscreteDistribution.uniformly_in(())
    assert not d.outcomes

def test_discrete_distribution_uniformly_of_nonempty() -> None:
    d = DiscreteDistribution.uniformly_of(5, 1, 2)
    assert [o.value for o in d.outcomes] == [1, 2, 5]
    assert [o.probability for o in d.outcomes] == approx([1/3, 1/3, 1/3])
    assert [o.cumulative_probability for o in d.outcomes] == approx([1/3, 2/3, 3/3])
    assert d.outcomes[-1].cumulative_probability == 1

def test_discrete_distribution_uniformly_of_empty() -> None:
    d = DiscreteDistribution.uniformly_of()
    assert not d.outcomes

def test_discrete_distribution_null() -> None:
    d = DiscreteDistribution.null()
    assert not d.outcomes

def test_discrete_distribution_iterate() -> None:
    d = DiscreteDistribution.uniformly_in(range(-10, 20))
    outcomes = tuple(d.iterate(-4, 2))
    assert [o.value for o in outcomes] == list(range(-4, 2))
    assert [o.probability for o in outcomes] == approx([1/30] * 6)
    assert [o.cumulative_probability for o in outcomes] == approx([7/30, 8/30, 9/30, 10/30, 11/30, 12/30])

def test_discrete_distribution_probability_in() -> None:
    d = DiscreteDistribution((
        DiscreteOutcome(1, 0.3, 0.3),
        DiscreteOutcome(2, 0.15, 0.45),
        DiscreteOutcome(3, 0.04, 0.49),
        DiscreteOutcome(4, 0.2, 0.69),
        DiscreteOutcome(5, 0.2, 0.89)
    ))
    assert d.probability_in(2, 5) == approx(0.15 + 0.04 + 0.2)
    assert d.probability_in(2, 2) == 0

def test_discrete_distribution_possible_in() -> None:
    d = DiscreteDistribution.uniformly_in(range(-5, 23))
    assert not d.possible_in(-20, -5)
    assert d.possible_in(-10, -4)
    assert d.possible_in(2, 7)
    assert d.possible_in(14, 28)
    assert not d.possible_in(23, 50)
    assert not d.possible_in(5, 5)

def test_discrete_distribution_certain_in_nonsingular() -> None:
    d = DiscreteDistribution.from_weights({1: 1, 2: 2, 3: 1, 4: 4, 5: 3})
    assert not d.certain_in(-3, 5)
    assert d.certain_in(1, 6)
    assert not d.certain_in(2, 4)
    assert d.certain_in(-10, 20)
    assert not d.certain_in(2, 6)
    assert not d.certain_in(20, 30)
    assert not d.certain_in(5, 6)
    assert not d.certain_in(5, 5)

def test_discrete_distribution_certain_in_singular() -> None:
    d = DiscreteDistribution.singular(6)
    assert d.certain_in(6, 7)
    assert d.certain_in(0, 10)
    assert not d.certain_in(6, 6)

def test_discrete_distribution_certain_in_sum_not_1() -> None:
    d = DiscreteDistribution((
        DiscreteOutcome(1, 0.3, 0.3),
        DiscreteOutcome(2, 0.15, 0.45),
        DiscreteOutcome(3, 0.04, 0.49),
        DiscreteOutcome(4, 0.2, 0.69),
        DiscreteOutcome(5, 0.2, 0.89)
    ))
    assert not d.certain_in(0, 4)
    assert not d.certain_in(1, 6)
    assert not d.certain_in(10, 23)

def test_discrete_distribution_has_possible_outcomes_true() -> None:
    d = DiscreteDistribution.singular(4)
    assert d.has_possible_outcomes

def test_discrete_distribution_has_possible_outcomes_false() -> None:
    d = DiscreteDistribution(())
    assert not d.has_possible_outcomes

def test_discrete_distribution_lower_bound_inclusive() -> None:
    d = DiscreteDistribution.uniformly_in(range(-10, 10, 2))
    assert d.lower_bound_inclusive(-100).value == -10
    assert d.lower_bound_inclusive(-10).value == -10
    assert d.lower_bound_inclusive(5).value == 6
    assert d.lower_bound_inclusive(6).value == 6
    assert d.lower_bound_inclusive(8).value == 8
    assert d.lower_bound_inclusive(9) is None
    assert d.lower_bound_inclusive(100) is None

def test_discrete_distribution_upper_bound_inclusive() -> None:
    d = DiscreteDistribution.uniformly_in(range(-10, 10, 2))
    assert d.upper_bound_inclusive(-100) is None
    assert d.upper_bound_inclusive(-10).value == -10
    assert d.upper_bound_inclusive(5).value == 4
    assert d.upper_bound_inclusive(6).value == 6
    assert d.upper_bound_inclusive(8).value == 8
    assert d.upper_bound_inclusive(9).value == 8
    assert d.upper_bound_inclusive(100).value == 8

def test_discrete_distribution_drop() -> None:
    d = DiscreteDistribution((
        DiscreteOutcome(1, 0.3, 0.3),
        DiscreteOutcome(2, 0.15, 0.45),
        DiscreteOutcome(3, 0.04, 0.49),
        DiscreteOutcome(4, 0.2, 0.69),
        DiscreteOutcome(5, 0.2, 0.89)
    ))
    result = d.drop(lambda v: v % 2 == 0)
    assert [o.value for o in result.outcomes] == [1, 3, 5]
    assert [o.probability for o in result.outcomes] == approx([0.3, 0.04, 0.2])
    assert [o.cumulative_probability for o in result.outcomes] == approx([0.3, 0.34, 0.54])

def test_discrete_distribution_map_values_bijection() -> None:
    d = DiscreteDistribution.from_weights({1: 1, 2: 4, 4: 3})
    result = d.map_values(lambda v: v + 1)
    assert [o.value for o in result.outcomes] == [2, 3, 5]
    assert [o.probability for o in result.outcomes] == approx([1/8, 4/8, 3/8])
    assert [o.cumulative_probability for o in result.outcomes] == approx([1/8, 5/8, 8/8])

def test_discrete_distribution_map_values_not_bijection() -> None:
    d = DiscreteDistribution.uniformly_in(range(20))
    result = d.map_values(lambda v: v if v % 6 == 0 else v // 3)
    assert [o.value for o in result.outcomes] == [0, 1, 2, 3, 4, 5, 6, 12, 18]
    assert [o.probability for o in result.outcomes] == approx([3/20, 3/20, 2/20, 3/20, 2/20, 3/20, 2/20, 1/20, 1/20])
    assert [o.cumulative_probability for o in result.outcomes] == \
        approx([3/20, 6/20, 8/20, 11/20, 13/20, 16/20, 18/20, 19/20, 20/20])

# TODO: test approx_eq


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
    # Note: not use approx() here, all three values should be exact.
    assert d.min == 23.456
    assert d.max == 23.456
    assert d.mean == 23.456

def test_float_distribution_uniformly_in() -> None:
    d = FloatDistribution.uniformly_in(-0.25, 2)
    assert d.min == approx(-0.25)
    assert d.max == approx(2)
    assert d.mean == approx(0.875)

def test_float_distribution_uniformly_around() -> None:
    d = FloatDistribution.uniformly_around(10, 4)
    assert d.min == approx(6)
    assert d.max == approx(14)
    assert d.mean == approx(10)

def test_float_distribution_to_str_singular() -> None:
    d = FloatDistribution.singular(123.456789)
    assert d.to_str(4) == '123.4568'

def test_float_distribution_to_str_nonsingular() -> None:
    d = FloatDistribution(min=-1.25, mean=3.4444444444, max=5.654)
    assert d.to_str(3) == '[-1.250, (3.444), 5.654]'

# TODO: test approx_eq

def test_float_distribution_neg() -> None:
    d = FloatDistribution(min=-1, max=3, mean=0.5)
    assert -d == FloatDistribution(min=-3, max=1, mean=-0.5)

def test_float_distribution_add_scalar() -> None:
    d = FloatDistribution(min=-1, max=2, mean=0.7)
    # TODO: use approx
    assert d + 17 == FloatDistribution(min=16, max=19, mean=17.7)

def test_float_distribution_add_distribution() -> None:
    d1 = FloatDistribution(min=-100, max=200, mean=123)
    d2 = FloatDistribution(min=-1, max=330, mean=-0.5)
    # TODO: use approx
    assert d1 + d2 == FloatDistribution(min=-101, max=530, mean=122.5)

def test_float_distribution_radd_scalar() -> None:
    d = FloatDistribution(min=-1, max=2, mean=0.7)
    # TODO: use approx
    assert 17 + d == FloatDistribution(min=16, max=19, mean=17.7)

def test_float_distribution_mul_scalar_positive() -> None:
    d = FloatDistribution(min=-10, max=11, mean=3.1)
    # TODO: use approx
    assert d * 1.5 == FloatDistribution(min=-15, max=16.5, mean=4.65)

def test_float_distribution_mul_scalar_negative() -> None:
    d = FloatDistribution(min=-10, max=11, mean=3.1)
    # TODO: use approx
    assert d * -1.5 == FloatDistribution(min=-16.5, max=15, mean=-4.65)

def test_float_distribution_rmul_scalar() -> None:
    d = FloatDistribution(min=-10, max=11, mean=3.1)
    # TODO: use approx
    assert 1.5 * d == FloatDistribution(min=-15, max=16.5, mean=4.65)
