import abc
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
    Union,
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
            self.std = (self.upper - self.lower) / \
                6  # This std seems to be sensible
        else:
            self.std = std

    def gen(self) -> int:
        """Generate integer according to Gaussian distribution."""
        return int(np.random.normal(self.mean, self.std))


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

    def __iter__(self):
        for trip in self.route:
            yield trip

class NoisedRoute(Route):
    parent_route_id: int


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

    def gen_merch_items(
            self,
            merch_len: int,
            merch_dist: Optional[Dict[str, float]] = None
    ) -> Tuple[MerchItem, ...]:
        """Gen some amount of merch items."""

        if merch_dist is None:
            merch_dist = self.merch_dist

        merch_names = sample_items(merch_dist, merch_len)

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
    return np.random.choice(
        list(dist.keys()),
        size=k,
        replace=False,
        p=list(dist.values())
    )


def sample_item(dist: Dict[K, float]) -> K:
    """Sample a key from a dictionary according to the weights in values."""
    return sample_items(dist, 1)[0]


def adjust(dist: Dict[K, float], to_remove: Union[K,  Set[K]]) -> Dict[K, float]:
    """Adjust a given distribution after removing an element."""
    if not isinstance(to_remove, set):
        p_remove = dist[to_remove]
        to_remove = set([to_remove])
    else:
        p_remove = sum(dist[key] for key in to_remove)

    total_p = sum(dist.values())
    new_p = total_p - p_remove

    remaining_elements = {
        k: v * total_p / new_p for k,
        v in dist.items() if k not in to_remove
    }

    return remaining_elements


def noise_route(
        route: Route,
        id: int,
        data_gen: RouteGenerator,
        merch_item_noise_map: Dict[str, IntSampler],
        merch_len_noiser: IntSampler
) -> Route:
    # TODO: How do we add / remove routes?
    # TODO: Should we change from / to city with some probability?
    noised_trips = []
    for trip in route:
        noised_merch = []

        # Noise merch amount
        for merch_name, amount in trip.merch:
            merch_item_noise = merch_item_noise_map[merch_name].gen()
            noised_amount = abs(amount + merch_item_noise)
            noised_merch.append((merch_name, noised_amount))

        # Noise merch length
        merch_len_noise = merch_len_noiser.gen()
        noised_merch_len = abs(merch_len_noise + len(trip.merch))

        # If we have a shorter merch length, sample that amount of items from the merch
        if noised_merch_len < len(trip.merch):
            merch_to_keep_indices = np.random.choice(
                np.arange(0, len(trip.merch) - 1),
                size=noised_merch_len,
                replace=False
            )
            noised_merch = [noised_merch[idx]
                            for idx in merch_to_keep_indices]

        # If we have more items, generate some new once, excluding elements currently
        # in merch.
        elif noised_merch_len > len(trip.merch):
            n_new_merch = noised_merch_len - len(trip.merch)
            merch_names = {merch_name for merch_name, _ in trip.merch}
            adjusted_dist = adjust(data_gen.merch_dist, merch_names)
            if adjusted_dist:
                new_merch = data_gen.gen_merch_items(n_new_merch, adjusted_dist)
                noised_merch += list(new_merch)
            else:
                noised_merch.append(trip.merch)

        # Create new trip
        noised_trip = Trip(
            from_city=trip.from_city,
            to_city=trip.to_city,
            merch=tuple(noised_merch)
        )
        # print(trip)
        # print(noised_trip)
        noised_trips.append(noised_trip)

    return NoisedRoute(
        id=id,
        route=noised_trips,
        parent_route_id=route.id
    )

# %%
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

    N_ROUTES = 3
    routes = [generator.gen_route(i) for i in range(N_ROUTES)]
    merch_item_noise_map = {
        merch_name: NormalIntSampler(-5, 5) for merch_name in merch_opts
    }
    merch_len_noiser = NormalIntSampler(-3, 3)
    route_to_noise = routes[0]
    noised_route = noise_route(route_to_noise, 42, generator,
                merch_item_noise_map, merch_len_noiser)

    print("Original")
    print(route_to_noise)
    print("Noised")
    print(noised_route)
