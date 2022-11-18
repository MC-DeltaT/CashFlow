"""Probability-related functionality."""


from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Callable, DefaultDict, Generic, Iterable, Iterator, Mapping, Protocol, Sequence, TypeVar


__all__ = [
    'DEFAULT_CERTAINTY_TOLERANCE',
    'DiscreteDistribution',
    'DiscreteInterval',
    'DiscreteOutcome',
    'effectively_certain',
    'ExplicitDiscreteDistribution',
    'FloatDistribution'
]


TOrdered = TypeVar('TOrdered', bound='Ordered')
TOrdered2 = TypeVar('TOrdered2', bound='Ordered')
TOrdered_co = TypeVar('TOrdered_co', bound='Ordered', covariant=True)


class Ordered(Protocol):
    """A type with a total ordering."""

    @abstractmethod
    def __eq__(self, other: Any, /) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __lt__(self: TOrdered, other: TOrdered, /) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __le__(self: TOrdered, other: TOrdered, /) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError()


class DiscreteInterval(Protocol[TOrdered_co]):
    @abstractmethod
    def __contains__(self, value: Any, /) -> bool:
        """Checks if `value` is contained within the interval."""

        raise NotImplementedError()

    @abstractmethod
    def __iter__(self) -> Iterator[TOrdered_co]:
        """Iterates all values contained within the interval, in ascending order."""

        raise NotImplementedError()


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


class DiscreteDistribution(Generic[TOrdered], ABC):
    """A generic probability distribution of discrete outcomes.
        The sum of probabilities of outcomes within the distribution is <= 1."""

    @staticmethod
    def singular(value: TOrdered, /) -> 'ExplicitDiscreteDistribution[TOrdered]':
        """Creates a distribution with a single value with 100% probability."""

        return DiscreteDistribution.uniformly_of(value)

    @staticmethod
    def uniformly_in(values: Iterable[TOrdered], /) -> 'ExplicitDiscreteDistribution[TOrdered]':
        """Creates a distribution where each of `values` has an equal probability of occurring.
            The total probability will sum to 1."""

        return ExplicitDiscreteDistribution.from_weights({value: 1 for value in values})

    @staticmethod
    def uniformly_of(*values: TOrdered) -> 'ExplicitDiscreteDistribution[TOrdered]':
        """Creates a distribution where each of `values` has an equal probability of occurring.
            The total probability will sum to 1."""

        return ExplicitDiscreteDistribution.uniformly_in(values)

    @abstractmethod
    def iterate(self, interval: DiscreteInterval[TOrdered], /) -> Iterable[DiscreteOutcome[TOrdered]]:
        """Iterates outcomes within `interval` that have nonzero probability, in ascending order."""

        raise NotImplementedError()

    @abstractmethod
    def drop(self, func: Callable[[TOrdered], bool], /) -> 'DiscreteDistribution[TOrdered]':
        """Creates a new distribution with outcomes for which `func` returns true are removed.
            The occurrence probability of each remaining outcome is unchanged, but the cumulative probabilities are
            updated."""

        raise NotImplementedError()

    @abstractmethod
    def map_values(self, func: Callable[[TOrdered], TOrdered2]) -> 'DiscreteDistribution[TOrdered2]':
        """Creates a new distribution with outcome values mapped by `func`.
            The mapping need not be bijective. If multiple values are mapped to the same new value, the occurrence
            probabilities will be summed. However, the result must still be a valid distribution (e.g. total probability
            cannot exceed 1)."""

        raise NotImplementedError()

    def could_occur_in(self, interval: DiscreteInterval[TOrdered], /) -> bool:
        """Checks if there is any outcomes with nonzero probability within `interval`."""

        return any(True for _ in self.iterate(interval))

    @abstractmethod
    def __add__(self, other: 'DiscreteDistribution[TOrdered]', /) -> 'DiscreteDistribution[TOrdered]':
        """Creates a new distribution with outcomes merged from both distributions.
            The probabilities of the outcomes are adjusted accordingly."""

        raise NotImplementedError()

    @abstractmethod
    def __mul__(self, other: 'DiscreteDistribution[TOrdered]', /) -> 'DiscreteDistribution[TOrdered]':
        """Creates a new distribution where each outcome's probability is multiplied by the probability from the other
            distribution."""

        raise NotImplementedError()


@dataclass(frozen=True, eq=False)
class ExplicitDiscreteDistribution(DiscreteDistribution[TOrdered]):
    """A discrete distribution given by explicit outcome data."""

    outcomes: Sequence[DiscreteOutcome[TOrdered]]       # Sorted in ascending order.

    def __post_init__(self) -> None:
        if not all(self.outcomes[i].value < self.outcomes[i + 1].value for i in range(len(self.outcomes) - 1)):
            raise ValueError('outcomes must have strictly increasing values')
        if not all(self.outcomes[i].cumulative_probability + self.outcomes[i].probability
                    <= self.outcomes[i + 1].cumulative_probability
                   for i in range(len(self.outcomes) - 1)):
            raise ValueError('outcomes must have monotonically increasing cumulative probabilities')

    @classmethod
    def from_weights(cls, value_weights: Mapping[TOrdered, float], /):
        """Creates a distribution from a mapping from values to likelihood weights.
            The probability of each outcome is formed by normalising the weights to sum to 1."""

        total_weight = sum(value_weights.values())
        value_probabilities = {value: weight / total_weight for value, weight in value_weights.items()}
        return cls._from_probabilities(value_probabilities)

    @classmethod
    def from_probabilities(cls, value_probabilities: Mapping[TOrdered, float], /):
        """Creates a distribution from a mapping from values to occurrence probabilities."""

        return cls._from_probabilities(value_probabilities)

    def iterate(self, interval: DiscreteInterval[TOrdered], /) -> Iterable[DiscreteOutcome[TOrdered]]:
        return (outcome for outcome in self.outcomes if outcome.value in interval)

    def map_values(self, func: Callable[[TOrdered], TOrdered2]):
        value_probabilities: DefaultDict[TOrdered2, float] = defaultdict(float)
        for outcome in self.outcomes:
            value_probabilities[func(outcome.value)] += outcome.probability
        return ExplicitDiscreteDistribution[TOrdered2].from_probabilities(value_probabilities)

    def drop(self, func: Callable[[TOrdered], bool], /):
        value_probabilities = {
            outcome.value: outcome.probability for outcome in self.outcomes if not func(outcome.value)}
        # Removing outcomes is guaranteed to reduce the cumulative probability to below 1, so no need to clamp.
        # If no outcomes are removed then this operation is a no-op, so also no need to clamp.
        return self._from_probabilities(value_probabilities, clamp_cumulative_down=False, clamp_cumulative_up=False)

    def __add__(self, other: 'DiscreteDistribution[TOrdered]', /):
        if not isinstance(other, ExplicitDiscreteDistribution):
            return NotImplemented
        value_probabilities = defaultdict(float, {outcome.value: outcome.probability for outcome in self.outcomes})
        for outcome in other.outcomes:
            value_probabilities[outcome.value] += outcome.probability
        for value in value_probabilities:
            value_probabilities[value] /= 2
        return self._from_probabilities(value_probabilities)

    def __mul__(self, other: 'DiscreteDistribution[TOrdered]', /):
        if not isinstance(other, ExplicitDiscreteDistribution):
            return NotImplemented
        value_probabilities = defaultdict(lambda: 1, {outcome.value: outcome.probability for outcome in self.outcomes})
        for outcome in other.outcomes:
            value_probabilities[outcome.value] *= outcome.probability
        return self._from_probabilities(value_probabilities)

    _CUMULATIVE_PROBABILITY_CLAMP = 1e-9
    """If the difference between a cumulative probability and 1 is less than this value, then the probability may be
        clamped to 1 in some circumstances."""

    @classmethod
    def _from_probabilities(cls, value_probabilities: Mapping[TOrdered, float], /,
            clamp_cumulative_down: float = _CUMULATIVE_PROBABILITY_CLAMP,
            clamp_cumulative_up: float = _CUMULATIVE_PROBABILITY_CLAMP):
        outcomes: list[DiscreteOutcome[TOrdered]] = []
        sorted_values = sorted(value_probabilities.keys())
        cumulative_probability = 0
        for value in sorted_values:
            probability = value_probabilities[value]
            new_cumulative_probability = cumulative_probability + probability
            # The cumulative probability may exceed 1 slightly due to floating point errors, which we can correct for.
            if clamp_cumulative_down and new_cumulative_probability > 1:
                # Only do the correction the first time and if the error is small.
                if cumulative_probability < 1 and (diff := new_cumulative_probability - 1) < clamp_cumulative_down:
                    new_cumulative_probability = 1
                    if outcomes:
                        # Also need to reduce the previous outcome's probability to have a consistent distribution.
                        outcomes[-1] = replace(outcomes[-1], probability=outcomes[-1].probability - diff)
                else:
                    # Otherwise we assume the caller has provided invalid probabilities (e.g. total >> 1).
                    raise ValueError('Sum of probabilities must not exceed 1')
            outcomes.append(DiscreteOutcome(value, probability, new_cumulative_probability))
            cumulative_probability = new_cumulative_probability
        # Due to floating point errors, the final cumulative probability may be slightly less than 1 even if it should
        # add up to 1, which we can correct for.
        if clamp_cumulative_up and outcomes:
            diff = 1 - outcomes[-1].cumulative_probability
            if 0 < diff < clamp_cumulative_up:
                # Also need to increase the last outcome's probability to have a consistent distribution.
                outcomes[-1] = replace(outcomes[-1],
                    probability=outcomes[-1].probability + diff, cumulative_probability=1)
        return cls(outcomes)


@dataclass(frozen=True, kw_only=True)
class FloatDistribution:
    """A probability distribution on the real numbers."""

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

    def __neg__(self):
        return type(self)(min=-self.max, max=-self.min, mean=-self.mean)

    def __mul__(self, other: float, /):
        if other >= 0:
            return type(self)(min=self.min * other, max=self.max * other, mean=self.mean * other)
        else:
            # If multiplier is negative, need to swap min and max.
            return type(self)(min=self.max * other, max=self.min * other, mean=self.mean * other)

    def __rmul__(self, other: float, /):
        return self * other


DEFAULT_CERTAINTY_TOLERANCE = 1e-6
"""Default tolerance when deciding if a probability is "certain"."""


def effectively_certain(probability: float, /, tolerance: float = DEFAULT_CERTAINTY_TOLERANCE) -> bool:
    """Checks if a probability is near enough to 1 to be considered "certain" for practical purposes.
        Often a probability won't be exactly 1 due to floating point errors."""

    if tolerance < 0:
        raise ValueError('tolerance must be >= 0')
    else:
        return probability >= 1 - tolerance
