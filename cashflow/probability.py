from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Callable, Generic, TypeVar, Union

from .utility import Ordered


__all__ = [
    'clamp_certain',
    'DEFAULT_CERTAINTY_TOLERANCE',
    'DiscreteDistribution',
    'DiscreteOutcome',
    'effectively_certain',
    'FloatDistribution'
]


T_Ordered = TypeVar('T_Ordered', bound=Ordered)
T_Ordered2 = TypeVar('T_Ordered2', bound=Ordered)


DEFAULT_CERTAINTY_TOLERANCE = 1e-6
"""Default tolerance when deciding if a probability is "certain"."""


def effectively_certain(probability: float, /, *, tolerance: float = DEFAULT_CERTAINTY_TOLERANCE) -> bool:
    """Checks if a probability is near enough to 1 to be considered "certain" for practical purposes.
        Often a probability won't be exactly 1 due to floating point inaccuracy."""

    if tolerance < 0:
        raise ValueError('tolerance must be >= 0')
    else:
        return probability >= 1 - tolerance


def clamp_certain(probability: float, /, *, tolerance: float = DEFAULT_CERTAINTY_TOLERANCE) -> float:
    """If `probability` is very near to 1, then returns 1. Else returns `probability`."""

    if effectively_certain(probability, tolerance=tolerance):
        return 1
    else:
        return probability


@dataclass(frozen=True)
class DiscreteOutcome(Generic[T_Ordered]):
    value: T_Ordered
    probability: float      # The unconditional probability that the outcome is equal to `value`.

    def __post_init__(self) -> None:
        if not 0 < self.probability <= 1:
            raise ValueError('probability must be in the range (0, 1]')


@dataclass(frozen=True)
class DiscreteDistribution(Generic[T_Ordered]):
    """Describes the probabilities of a set of discrete outcomes.

        This class can represent an entire probability distribution (i.e. probability sums to 1) or a subset of a
        distribution (i.e. probability sums to less than 1)."""

    outcomes: tuple[DiscreteOutcome[T_Ordered], ...]     # Sorted in ascending order.

    def __init__(self, outcomes: Iterable[DiscreteOutcome[T_Ordered]]) -> None:
        """Construct the distribution from information about each outcome.

            Outcomes need not perfectly describe a complete probability distribution, but the following must be true:
                - Outcomes must have strictly increasing values (i.e. in ascending order, no duplicates).
                - The sum of probabilities of all outcomes must be <= 1."""

        outcomes = tuple(outcomes)
        if not all(outcomes[i].value < outcomes[i + 1].value for i in range(len(outcomes) - 1)):
            raise ValueError('outcomes must have strictly increasing values')
        if sum(outcome.probability for outcome in outcomes) > 1:
            raise ValueError('Sum of probabilities of all outcomes must be in <= 1')
        super().__setattr__('outcomes', outcomes)

    @classmethod
    def from_weights(cls, value_weights: Mapping[T_Ordered, float], /):
        """Creates a distribution from a mapping of values to likelihood weights.
            The probability of each outcome is formed by normalising the weights to sum to 1."""

        total_weight = sum(value_weights.values())
        value_probabilities = {value: weight / total_weight for value, weight in value_weights.items()}
        return cls._from_probabilities(value_probabilities)

    @classmethod
    def from_probabilities(cls, value_probabilities: Mapping[T_Ordered, float], /):
        """Creates a distribution from a mapping from values to occurrence probabilities.

            If the sum of probabilities is very near 1, then the probabilities will be adjusted so that the sum is
            exactly 1."""

        return cls._from_probabilities(value_probabilities)

    @classmethod
    def singular(cls, value: T_Ordered, /):
        """Creates a distribution with a single value with 100% probability."""

        return cls.uniformly_of(value)

    @classmethod
    def uniformly_in(cls, values: Iterable[T_Ordered], /):
        """Creates a distribution where each of `values` has an equal probability of occurring.
            The total probability will sum to 1."""

        return cls.from_weights({value: 1 for value in values})

    @classmethod
    def uniformly_of(cls, *values: T_Ordered):
        """Creates a distribution where each of `values` has an equal probability of occurring.
            The total probability will sum to 1."""

        return cls.uniformly_in(values)

    @classmethod
    def null(cls):
        """Creates a distribution with no possible outcomes."""

        return cls(())

    @property
    def has_possible_outcomes(self) -> bool:
        return len(self.outcomes) > 0

    def probability_in(self, inclusive_lower_bound: T_Ordered, exclusive_upper_bound: T_Ordered) -> float:
        """Returns the sum of probability of all outcomes in the interval
            [`inclusive_lower_bound`, `exclusive_upper_bound`)."""

        return sum(outcome.probability for outcome in self.iterate(inclusive_lower_bound, exclusive_upper_bound))

    def cumulative_probability(self, value: T_Ordered, /) -> float:
        """Computes the total probability of outcomes with value <= `value`."""

        cumulative = sum(outcome.probability for outcome in self.outcomes if outcome.value <= value)
        # Sum may exceed 1 slightly due to floating point error.
        return min(cumulative, 1)

    def lower_bound_inclusive(self, value: T_Ordered, /) -> DiscreteOutcome[T_Ordered] | None:
        """Returns the outcome with the lowest value >= `value`and nonzero probability, or `None` if there is no such
            outcome."""

        idx = bisect_left(self.outcomes, value, key=lambda outcome: outcome.value)
        if idx < len(self.outcomes):
            return self.outcomes[idx]
        else:
            return None

    def upper_bound_inclusive(self, value: T_Ordered, /) -> DiscreteOutcome[T_Ordered] | None:
        """Returns the outcome with the highest value <= `value` and nonzero probability, or `None` if there is no such
            outcome."""

        idx = bisect_right(self.outcomes, value, key=lambda outcome: outcome.value)
        if idx > 0:
            return self.outcomes[idx - 1]
        else:
            return None

    def iterate(self, inclusive_lower_bound: T_Ordered, exclusive_upper_bound: T_Ordered) \
            -> Iterable[DiscreteOutcome[T_Ordered]]:
        """Iterates outcomes within the interval [`inclusive_lower_bound`, `exclusive_upper_bound`) that have nonzero
            probability, in ascending order."""

        return (outcome for outcome in self.outcomes if inclusive_lower_bound <= outcome.value < exclusive_upper_bound)

    def subset(self, func: Callable[[T_Ordered], bool], /):
        """Creates a new distribution where outcomes for which `func` returns false have 0 probability (i.e. removed).

            The occurrence probability of each remaining outcome is unchanged."""

        filtered_outcomes = (outcome for outcome in self.outcomes if func(outcome.value))
        return type(self)(filtered_outcomes)

    def map_values(self, func: Callable[[T_Ordered], T_Ordered2], /):
        """Creates a new distribution with outcome values mapped by `func`.

            The mapping need not be bijective. If multiple values are mapped to the same new value, the occurrence
            probabilities will be summed. However, the result must still be a valid distribution (e.g. total probability
            cannot exceed 1)."""

        value_probabilities: defaultdict[T_Ordered2, float] = defaultdict(float)
        for outcome in self.outcomes:
            value_probabilities[func(outcome.value)] += outcome.probability
        return DiscreteDistribution[T_Ordered2].from_probabilities(value_probabilities)

    _CUMULATIVE_PROBABILITY_CLAMP = 1e-9
    """If the difference between a cumulative probability and 1 is less than this value, then the probability may be
        clamped to 1 in some circumstances to correct for floating point inaccuracy."""

    @classmethod
    def _from_probabilities(cls, value_probabilities: Mapping[T_Ordered, float], /,
            clamp_cumulative_down: float = _CUMULATIVE_PROBABILITY_CLAMP,
            clamp_cumulative_up: float = _CUMULATIVE_PROBABILITY_CLAMP):
        outcomes: list[DiscreteOutcome[T_Ordered]] = []
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
            outcomes.append(DiscreteOutcome(value, probability))
            cumulative_probability = new_cumulative_probability
        # Due to floating point inaccuracy, the final cumulative probability may be slightly less than 1 even if it
        # should add up to 1, which we can correct for.
        if clamp_cumulative_up and outcomes:
            diff = 1 - cumulative_probability
            if 0 < diff < clamp_cumulative_up:
                # Also need to increase the last outcome's probability to have a consistent distribution.
                outcomes[-1] = replace(outcomes[-1], probability=outcomes[-1].probability + diff)
        return cls(outcomes)


@dataclass(frozen=True, kw_only=True)
class FloatDistribution:
    """A basic probability distribution on the real numbers."""

    min: float
    mean: float
    max: float

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

        if radius < 0:
            raise ValueError('radius must be nonnegative')
        return cls(min=centre - radius, max=centre + radius, mean=centre)

    @classmethod
    def from_inexact(cls, *, min: float, mean: float, max: float, tolerance: float = 1e-9):
        """Creates a distribution where `min`, `mean`, and `max` are fixed up to ensure `min` <= `mean` <= `max`.

            Useful for when constructing a distribution from the results of floating point arithmetic."""

        if mean < min and min - mean < tolerance:
            mean = min
        if max < mean and mean - max < tolerance:
            max = mean
        return cls(min=min, mean=mean, max=max)

    def to_str(self, decimals: int = 2) -> str:
        if decimals < 0:
            raise ValueError('decimals must be nonnegative')

        def float_to_str(value: float) -> str:
            return format(round(value, decimals), f'.{decimals}f')

        if abs(self.min - self.max) < 10 ** -decimals:
            return f'{float_to_str(self.min)}'
        else:
            return f'[{float_to_str(self.min)}, ({float_to_str(self.mean)}), {float_to_str(self.max)}]'

    def __neg__(self):
        return type(self)(min=-self.max, max=-self.min, mean=-self.mean)

    def __add__(self, other: Union[float, 'FloatDistribution'], /) -> 'FloatDistribution':
        if isinstance(other, (float, int)):
            return type(self)(min=self.min + other, max=self.max + other, mean=self.mean + other)
        elif isinstance(other, FloatDistribution):
            # Is this legit maths?
            return type(self)(min=self.min + other.min, max=self.max + other.max, mean=self.mean + other.mean)
        else:
            return NotImplemented

    def __radd__(self, other: Union[float, 'FloatDistribution'], /):
        return self + other

    def __mul__(self, other: float, /) -> 'FloatDistribution':
        if isinstance(other, (float, int)):
            if other >= 0:
                return type(self)(min=self.min * other, max=self.max * other, mean=self.mean * other)
            else:
                # If multiplier is negative, need to swap min and max.
                return type(self)(min=self.max * other, max=self.min * other, mean=self.mean * other)
        else:
            return NotImplemented

    def __rmul__(self, other: float, /):
        return self * other
