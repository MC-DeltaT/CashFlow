from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from math import isclose
from typing import Callable, Generic, TypeVar, Union

from .utility import Ordered


__all__ = [
    'DEFAULT_CERTAINTY_TOLERANCE',
    'DiscreteDistribution',
    'DiscreteOutcome',
    'effectively_certain',
    'FloatDistribution'
]


TOrdered = TypeVar('TOrdered', bound=Ordered)
TOrdered2 = TypeVar('TOrdered2', bound=Ordered)


DEFAULT_CERTAINTY_TOLERANCE = 1e-6
"""Default tolerance when deciding if a probability is "certain"."""


def effectively_certain(probability: float, /, *, tolerance: float = DEFAULT_CERTAINTY_TOLERANCE) -> bool:
    """Checks if a probability is near enough to 1 to be considered "certain" for practical purposes.
        Often a probability won't be exactly 1 due to floating point inaccuracy."""

    if tolerance < 0:
        raise ValueError('tolerance must be >= 0')
    else:
        return probability >= 1 - tolerance


@dataclass(frozen=True)
class DiscreteOutcome(Generic[TOrdered]):
    value: TOrdered
    probability: float      # The unconditional probability that the outcome is equal to `value`.
    cumulative_probability: float   # The probability that the outcome is less than or equal to `value`.

    def __post_init__(self) -> None:
        if not 0 < self.probability <= 1:
            raise ValueError('probability must be in the range (0, 1]')
        elif not self.probability <= self.cumulative_probability <= 1:
            raise ValueError('cumulative_probability must be in the range [probability, 1]')


@dataclass(frozen=True)
class DiscreteDistribution(Generic[TOrdered]):
    """Describes the probabilities of a set of discrete outcomes.
    
        This class can represent an entire probability distribution (i.e. probability sums to 1) or a subset of a
        distribution (i.e. probability sums to less than 1)."""

    outcomes: tuple[DiscreteOutcome[TOrdered], ...]     # Sorted in ascending order.

    def __init__(self, outcomes: Iterable[DiscreteOutcome[TOrdered]]) -> None:
        """Construct the distribution from information about each outcome.
        
            Outcomes need not perfectly describe a complete probability distribution, but the following must be true:
                - Outcomes must have strictly increasing values (i.e. in ascending order, no duplicates).
                - The sum of probabilities of all outcomes must be <= 1.
                - Cumulative probabilities must be consistent."""

        outcomes = tuple(outcomes)
        if not all(outcomes[i].value < outcomes[i + 1].value for i in range(len(outcomes) - 1)):
            raise ValueError('outcomes must have strictly increasing values')
        if not all(outcomes[i].cumulative_probability + outcomes[i + 1].probability
                    <= outcomes[i + 1].cumulative_probability
                   for i in range(len(outcomes) - 1)):
            raise ValueError('Invalid cumulative probabilities')
        if sum(outcome.probability for outcome in outcomes) > 1:
            raise ValueError('Sum of probabilities of all outcomes must be in <= 1')
        super().__setattr__('outcomes', outcomes)

    @classmethod
    def from_weights(cls, value_weights: Mapping[TOrdered, float], /):
        """Creates a distribution from a mapping of values to likelihood weights.
            The probability of each outcome is formed by normalising the weights to sum to 1."""

        total_weight = sum(value_weights.values())
        value_probabilities = {value: weight / total_weight for value, weight in value_weights.items()}
        return cls._from_probabilities(value_probabilities)

    @classmethod
    def from_probabilities(cls, value_probabilities: Mapping[TOrdered, float], /):
        """Creates a distribution from a mapping from values to occurrence probabilities."""

        return cls._from_probabilities(value_probabilities)

    @classmethod
    def singular(cls, value: TOrdered, /):
        """Creates a distribution with a single value with 100% probability."""

        return cls.uniformly_of(value)

    @classmethod
    def uniformly_in(cls, values: Iterable[TOrdered], /):
        """Creates a distribution where each of `values` has an equal probability of occurring.
            The total probability will sum to 1."""

        return cls.from_weights({value: 1 for value in values})

    @classmethod
    def uniformly_of(cls, *values: TOrdered):
        """Creates a distribution where each of `values` has an equal probability of occurring.
            The total probability will sum to 1."""

        return cls.uniformly_in(values)

    @classmethod
    def null(cls):
        """Creates a distribution where all outcomes have zero probability."""

        return cls(())

    def iterate(self, inclusive_lower_bound: TOrdered, exclusive_upper_bound: TOrdered) \
            -> Iterable[DiscreteOutcome[TOrdered]]:
        """Iterates outcomes within the interval [`inclusive_lower_bound`, `exclusive_upper_bound`) that have nonzero
            probability, in ascending order."""

        return (outcome for outcome in self.outcomes if inclusive_lower_bound <= outcome.value < exclusive_upper_bound)

    def probability_in(self, inclusive_lower_bound: TOrdered, exclusive_upper_bound: TOrdered) -> float:
        """Returns the sum of probability of all outcomes in the interval
            [`inclusive_lower_bound`, `exclusive_upper_bound`)."""

        return sum(outcome.probability for outcome in self.iterate(inclusive_lower_bound, exclusive_upper_bound))

    def possible_in(self, inclusive_lower_bound: TOrdered, exclusive_upper_bound: TOrdered) -> bool:
        """Checks if there are any outcomes with nonzero probability within the interval
            [`inclusive_lower_bound`, `exclusive_upper_bound`)."""

        return any(True for _ in self.iterate(inclusive_lower_bound, exclusive_upper_bound))

    def certain_in(self, inclusive_lower_bound: TOrdered, exclusive_upper_bound: TOrdered,
            tolerance: float = DEFAULT_CERTAINTY_TOLERANCE) -> bool:
        """Checks if the distribution is guaranteed to take on a value within the interval
            [`inclusive_lower_bound`, `exclusive_upper_bound`)."""

        # The max probability is 1, so if the event is certain within the interval, then the interval must span the
        # entire distribution and the sum of all probabilities must be 1.
        return (len(self.outcomes) > 0
            and self.outcomes[0].value >= inclusive_lower_bound and self.outcomes[-1].value < exclusive_upper_bound
            and effectively_certain(self.outcomes[-1].cumulative_probability, tolerance=tolerance))

    @property
    def has_possible_outcomes(self) -> bool:
        return len(self.outcomes) > 0

    def lower_bound_inclusive(self, value: TOrdered, /) -> DiscreteOutcome[TOrdered] | None:
        """Returns the outcome with the lowest value >= `value`and nonzero probability, or `None` if there is no such
            outcome."""

        idx = bisect_left(self.outcomes, value, key=lambda outcome: outcome.value)
        if idx < len(self.outcomes):
            return self.outcomes[idx]
        else:
            return None

    def upper_bound_inclusive(self, value: TOrdered, /) -> DiscreteOutcome[TOrdered] | None:
        """Returns the outcome with the highest value <= `value` and nonzero probability, or `None` if there is no such
            outcome."""

        idx = bisect_right(self.outcomes, value, key=lambda outcome: outcome.value)
        if idx > 0:
            return self.outcomes[idx - 1]
        else:
            return None

    def subset(self, func: Callable[[TOrdered], bool], /, adjust_cumulative: bool):
        """Creates a new distribution where outcomes for which `func` returns false have 0 probability (i.e. removed).
            
            The occurrence probability of each remaining outcome is unchanged.
            If `adjust_cumulative` is true, the cumulative probabilities are updated to reflect the removed outcomes."""

        filtered_outcomes = (outcome for outcome in self.outcomes if func(outcome.value))
        if adjust_cumulative:
            value_probabilities = {outcome.value: outcome.probability for outcome in filtered_outcomes}
            # Removing outcomes is guaranteed to reduce the cumulative probability to below 1, so no need to clamp.
            # If no outcomes are removed then this operation is a no-op, so also no need to clamp.
            return self._from_probabilities(value_probabilities, clamp_cumulative_down=False, clamp_cumulative_up=False)
        else:
            return type(self)(filtered_outcomes)

    def map_values(self, func: Callable[[TOrdered], TOrdered2], /):
        """Creates a new distribution with outcome values mapped by `func`.

            The mapping need not be bijective. If multiple values are mapped to the same new value, the occurrence
            probabilities will be summed. However, the result must still be a valid distribution (e.g. total probability
            cannot exceed 1)."""

        value_probabilities: defaultdict[TOrdered2, float] = defaultdict(float)
        for outcome in self.outcomes:
            value_probabilities[func(outcome.value)] += outcome.probability
        return DiscreteDistribution[TOrdered2].from_probabilities(value_probabilities)

    def approx_eq(self, other: 'DiscreteDistribution[TOrdered]', /, *, rel_tol: float = 1e-6, abs_tol: float = 0) \
            -> bool:
        """Checks if two distributions have the same outcomes and probabilities, tolerating some floating point
            inaccuracy when comparing probabilities."""

        def outcome_approx_eq(outcome1: DiscreteOutcome[TOrdered], outcome2: DiscreteOutcome[TOrdered]) -> bool:
            # The outcome values must be exactly equal, because the distribution is designed to be discrete - we don't
            # care for floating point values.
            return (outcome1.value == outcome2.value
                and isclose(outcome1.probability, outcome2.probability, rel_tol=rel_tol, abs_tol=abs_tol)
                and isclose(outcome1.cumulative_probability, outcome2.cumulative_probability,
                    rel_tol=rel_tol, abs_tol=abs_tol))

        if len(self.outcomes) != len(other.outcomes):
            return False
        else:
            return all(outcome_approx_eq(o1, o2) for o1, o2 in zip(self.outcomes, other.outcomes))

    def _find_outcome(self, value: TOrdered, /) -> DiscreteOutcome[TOrdered] | None:
        """Returns the outcome with the given value and nonzero probability, or `None` if there is no such outcome."""

        idx = bisect_left(self.outcomes, value, key=lambda outcome: outcome.value)
        if idx < len(self.outcomes) and self.outcomes[idx].value == value:
            return self.outcomes[idx]
        else:
            return None

    _CUMULATIVE_PROBABILITY_CLAMP = 1e-9
    """If the difference between a cumulative probability and 1 is less than this value, then the probability may be
        clamped to 1 in some circumstances to correct for floating point inaccuracy."""

    @classmethod
    def _from_probabilities(cls, value_probabilities: Mapping[TOrdered, float], /,
            clamp_cumulative_down: float = _CUMULATIVE_PROBABILITY_CLAMP,
            clamp_cumulative_up: float = _CUMULATIVE_PROBABILITY_CLAMP):
        outcomes: list[DiscreteOutcome[TOrdered]] = []
        sorted_values = sorted(value_probabilities.keys())
        cumulative_probability = 0
        for value in sorted_values:
            probability = value_probabilities[value]
            if probability <= 0:
                raise ValueError('Probabilities must be > 0')
            new_cumulative_probability = cumulative_probability + probability
            # The cumulative probability may exceed 1 slightly due to floating point inaccuracy, which we can correct.
            if clamp_cumulative_down and new_cumulative_probability > 1:
                # Only do the correction the first time and if the error is small.
                if cumulative_probability < 1 and (diff := new_cumulative_probability - 1) < clamp_cumulative_down:
                    new_cumulative_probability = 1
                    if outcomes:
                        # Also need to reduce the outcome's probability to have a consistent distribution.
                        probability -= diff
                else:
                    # Otherwise we assume the caller has provided invalid probabilities (e.g. total >> 1).
                    raise ValueError('Sum of probabilities must not exceed 1')
            outcomes.append(DiscreteOutcome(value, probability, new_cumulative_probability))
            cumulative_probability = new_cumulative_probability
        # Due to floating point inaccuracy, the final cumulative probability may be slightly less than 1 even if it
        # should add up to 1, which we can correct for.
        if clamp_cumulative_up and outcomes:
            diff = 1 - outcomes[-1].cumulative_probability
            if 0 < diff < clamp_cumulative_up:
                # Also need to increase the last outcome's probability to have a consistent distribution.
                outcomes[-1] = replace(outcomes[-1],
                    probability=outcomes[-1].probability + diff, cumulative_probability=1)
        return cls(outcomes)


@dataclass(frozen=True, kw_only=True)
class FloatDistribution:
    """A basic probability distribution on the real numbers."""

    min: float
    max: float
    mean: float

    def __post_init__(self) -> None:
        if not self.min <= self.mean <= self.max:
            raise ValueError('min <= mean <= max must be true')

    @classmethod
    def singular(cls, value: float, /):
        """Creates a distribution with a single value with 100% probability."""

        return cls(min=value, max=value, mean=value)

    @classmethod
    def uniformly_in(cls, lower_bound: float, upper_bound: float):
        """Creates a uniform distribution on the interval [`lower_bound`, `upper_bound`]."""

        return cls(min=lower_bound, max=upper_bound, mean=(lower_bound + upper_bound) / 2)

    @classmethod
    def uniformly_around(cls, centre: float, radius: float):
        """Creates a uniform distribution on the interval [`centre` - `radius`, `centre` + `radius`]"""

        radius = abs(radius)
        return cls(min=centre - radius, max=centre + radius, mean=centre)

    def to_str(self, decimals: int = 2) -> str:
        if decimals < 0:
            raise ValueError('decimals must be nonnegative')

        def float_to_str(value: float) -> str:
            return format(round(value, decimals), f'.{decimals}f')

        if abs(self.min - self.max) < 10 ** -decimals:
            return f'{float_to_str(self.min)}'
        else:
            return f'[{float_to_str(self.min)}, ({float_to_str(self.mean)}), {float_to_str(self.max)}]'

    def approx_eq(self, other: 'FloatDistribution', /, *, rel_tol: float = 1e-6, abs_tol: float = 0) -> bool:
        """Checks if two distributions have equal min, mean, and max, with tolerance for floating point inaccuracy."""

        return (isclose(self.min, other.min, rel_tol=rel_tol, abs_tol=abs_tol)
            and isclose(self.max, other.max, rel_tol=rel_tol, abs_tol=abs_tol)
            and isclose(self.mean, other.mean, rel_tol=rel_tol, abs_tol=abs_tol))

    def __neg__(self):
        return type(self)(min=-self.max, max=-self.min, mean=-self.mean)

    def __add__(self, other: Union[float, 'FloatDistribution'], /):
        if isinstance(other, FloatDistribution):
            # Is this legit maths?
            return type(self)(min=self.min + other.min, max=self.max + other.max, mean=self.mean + other.mean)
        else:
            return type(self)(min=self.min + other, max=self.max + other, mean=self.mean + other)

    def __radd__(self, other: Union[float, 'FloatDistribution'], /):
        return self + other

    def __mul__(self, other: float, /):
        if other >= 0:
            return type(self)(min=self.min * other, max=self.max * other, mean=self.mean * other)
        else:
            # If multiplier is negative, need to swap min and max.
            return type(self)(min=self.max * other, max=self.min * other, mean=self.mean * other)

    def __rmul__(self, other: float, /):
        return self * other
