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

    def __len__(self):
        return len(self.route)


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
        route_len_noiser: IntSampler,
        merch_item_noise_map: Dict[str, IntSampler],
        merch_len_noiser: IntSampler
) -> Route:
    # TODO: How do we add / remove routes?
    # TODO: Should we change from / to city with some probability?
    new_route_len = len(route) + route_len_noiser.gen()

    if new_route_len <= 0:
        return NoisedRoute(id=id, route=[], parent_route_id=route.id)

    extra_trips = []

    if new_route_len <= len(route):
        trips_to_noise = sample_items(
            get_uni_dist_cat(route.route), new_route_len)
    else:
        trips_to_noise = list(route.route)
        n_routes_to_gen = new_route_len - len(planned_routes)
        prev_trip_city = trips_to_noise[-1].from_city
        for i in range(n_routes_to_gen):
            trip = data_gen.gen_trip(prev_trip_city)
            extra_trips.append(trip)
            prev_trip_city = extra_trips[i].from_city

    noised_trips = []

    for trip in trips_to_noise:
        noised_merch = []

        # Noise merch amount
        for merch_name, amount in trip.merch:
            merch_item_noise = merch_item_noise_map[merch_name].gen()
            noised_amount = amount + merch_item_noise
            if noised_amount > 0:
                noised_merch.append((merch_name, noised_amount))

        # Noise merch length
        merch_len_noise = merch_len_noiser.gen()
        noised_merch_len = merch_len_noise + len(trip.merch)
        if noised_merch_len <= 0:
            break

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

                # Make sure that we can never sample more items than we have
                if n_new_merch > len(adjusted_dist):
                    n_new_merch = len(adjusted_dist)

                new_merch = data_gen.gen_merch_items(
                    n_new_merch, adjusted_dist)
                noised_merch += list(new_merch)
            else:
                noised_merch.append(trip.merch)

        # Create new trip
        noised_trip = Trip(
            from_city=trip.from_city,
            to_city=trip.to_city,
            merch=tuple(noised_merch)
        )
        noised_trips.append(noised_trip)

    noised_trips += extra_trips

    if not noised_trips:
        raise Exception(
            "Noised trips such that no trips are left, please adjust parameters "
            "so this cannot happen."
        )
    return NoisedRoute(
        id=id,
        route=noised_trips,
        parent_route_id=route.id
    )


# %%
if __name__ == "__main__":

    start = time.time()
    # Options for planned routes
    N_PLANNED_ROUTES = 1_000

    merch_items = {"Apples", "Pears", "Bananas",
                   "Kiwis", "Oranges", "Mandarins"}
    cities = {"Amsterdam", "Utrecht", "Delft"}

    merch_uni_dist = get_uni_dist_cat(merch_items)
    city_uni_dist = get_uni_dist_cat(cities)

    merch_len_sampler = UniformIntSampler(3, 5)
    merch_sampler_map = {
        merch_name: UniformIntSampler(50, 100)
        for merch_name in merch_items
    }
    route_len_sampler = UniformIntSampler(3, 5)
    generator = RouteGenerator(
        city_uni_dist,
        merch_uni_dist,
        merch_len_sampler,
        merch_sampler_map,
        route_len_sampler
    )
    planned_routes = [generator.gen_route(i) for i in range(N_PLANNED_ROUTES)]

    # Options for planned routes
    N_ACTUAL_ROUTES = 10_000

    # How much each merch quantity will be noised
    merch_item_noise_map = {
        merch_name: NormalIntSampler(-50, 50) for merch_name in merch_items
    }

    # How big each trip merch length will be
    merch_len_noiser = NormalIntSampler(-2, 2)
    route_len_sampler = NormalIntSampler(-2, 2)

    actual_routes = []
    planned_routes_uni_dist = get_uni_dist_cat(planned_routes)
    for i in range(N_ACTUAL_ROUTES):
        picked_actual_route = sample_item(planned_routes_uni_dist)
        actual_route = noise_route(
            picked_actual_route,
            i,
            generator,
            route_len_sampler,
            merch_item_noise_map,
            merch_len_noiser
        )
        actual_routes.append(actual_route)

    print(planned_routes)
    print(actual_routes)
    end = time.time()
    print(f"{end - start} seconds elapsed.")
