import abc
import time
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

import numpy as np  # type: ignore
from pydantic import BaseModel  # type: ignore

K = TypeVar("K")

MerchItem = Tuple[str, int]


class IntSampler(metaclass=abc.ABCMeta):
    """Interface for an integer sampler."""

    def __init__(self, lower: int, upper: int) -> None:
        self.lower = lower
        self.upper = upper

    @abc.abstractmethod
    def gen(self) -> int:
        """
        Generate an integer between `lower` and `upper` according to some distribution.
        """
        ...


class UniformIntSampler(IntSampler):

    def gen(self) -> int:
        """Generate integer according to uniform distribution."""
        return np.random.randint(self.lower, self.upper)


class NormalIntSampler(IntSampler):

    def __init__(
            self,
            lower: int,
            upper: int,
            mean: Optional[float] = None,
            std: Optional[float] = None
    ) -> None:
        super().__init__(lower, upper)
        if not mean:
            self.mean = (self.lower + self.upper) / 2
        else:
            self.mean = mean

        if not std:
            self.std = (self.upper - self.lower) / 6  # This std seems to be sensible
        else:
            self.std = std

    def gen(self) -> int:
        """Generate integer according to Gaussian distribution."""
        return np.random.normal(self.mean, self.std)


class ImmutableModel(BaseModel):
    """Immutable pydantic model, that provides hash function."""

    class Config:
        frozen = True


class Trip(ImmutableModel):
    from_city: str
    to_city: str

    merch: Tuple[MerchItem, ...]


class Route(ImmutableModel):
    id: int
    route: Tuple[Trip, ...]


@dataclass
class RouteGenerator:

    # Probability of starting in/moving to a certain city
    city_dist: Dict[str, float]

    # Probability of taking at least 1 element of some merch on a trip
    merch_dist: Dict[str, float]

    # Range within the maximum length of unique merchandise items falls.
    merch_len_sampler: IntSampler

    # Range of amount of merchandise we will take for each element
    merch_sampler_map: Mapping[str, IntSampler]

    # Range of amount of trips each route will contain
    route_len_sampler: IntSampler

    def gen_city(
            self,
            city_prior_dist: Optional[Dict[str, float]] = None
    ) -> str:
        """Generate a random city according to some distribution."""
        if not city_prior_dist:
            city_prior_dist = self.city_dist

        return sample_item(city_prior_dist)

    def gen_merch_items(self, merch_len: int) -> Tuple[MerchItem, ...]:
        """Gen some amount of merch items."""

        merch_names = sample_items(self.merch_dist, merch_len)

        # Make sure all merch item names are unique
        assert len(merch_names) == len(set(merch_names))

        # Generate a random amount of merch items for each merch item name
        merch_items = [
            (merch_name, self.merch_sampler_map[merch_name].gen())
            for merch_name in merch_names
        ]
        return tuple(merch_items)

    def gen_trip(self, from_city: Optional[str] = None) -> Trip:
        """Generate a trip."""

        if not from_city:
            from_city = self.gen_city()

        new_dist = adjust(self.city_dist, from_city)

        to_city = self.gen_city(new_dist)

        merch_len = self.merch_len_sampler.gen()

        merch = self.gen_merch_items(merch_len)

        return Trip(from_city=from_city, to_city=to_city, merch=merch)

    def gen_trips(self, n: int, unique: bool = False) -> Tuple[Trip, ...]:

        # TODO: Optionally make sure no two trips with same start <-> end can exist
        from_city = None
        trips = []

        for _ in range(n):
            trip = self.gen_trip(from_city)
            from_city = trip.to_city
            trips.append(trip)

        return tuple(trips)

    def gen_route(self, id: int) -> Route:
        route_len = self.route_len_sampler.gen()
        route = self.gen_trips(route_len)

        return Route(id=id, route=route)


def get_uni_dist_cat(keys: Set[Any]) -> Dict[Any, float]:
    return {key: 1/len(keys) for key in keys}


def sample_items(dist: Dict[K, float], k: int) -> List[K]:
    """Sample keys from a dictionary according to the weights in values."""
    return np.random.choice(list(dist.keys()), size=k, replace=False, p=list(dist.values()))


def sample_item(dist: Dict[K, float]) -> K:
    """Sample a key from a dictionary according to the weights in values."""
    return sample_items(dist, 1)[0]


def adjust(dist: Dict[K, float], to_remove: K) -> Dict[K, float]:
    """Adjust a given distribution after removing an element."""
    p_remove = dist[to_remove]

    total_p = sum(dist.values())
    new_p = total_p - p_remove

    remaining_elements = {
        k: v * total_p / new_p for k,
        v in dist.items() if k != to_remove
    }

    return remaining_elements


if __name__ == "__main__":

    merch_opts = {"Apples", "Pears", "Bananas"}
    cities = {"Amsterdam", "Utrecht", "Delft"}

    merch_uni_dist = get_uni_dist_cat(merch_opts)
    city_uni_dist = get_uni_dist_cat(cities)

    merch_len_sampler = UniformIntSampler(1, 3)

    merch_sampler_map = {
        merch_name: UniformIntSampler(1, 100)
        for merch_name in merch_opts
    }

    route_len_sampler = UniformIntSampler(1, 5)

    generator = RouteGenerator(
        city_uni_dist,
        merch_uni_dist,
        merch_len_sampler,
        merch_sampler_map,
        route_len_sampler
    )

    N_ROUTES = 100

    start = time.time()
    routes = [generator.gen_route(i) for i in range(N_ROUTES)]
    end = time.time()
    print(routes)
    print(f"Generated {N_ROUTES} in {end - start} seconds.")
    # print(routes)
