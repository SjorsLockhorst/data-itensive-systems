import random
import copy
from typing import Any, Dict, Tuple, List, Optional, TypeVar

from dataclasses import dataclass
from pydantic import BaseModel

K = TypeVar("K")

MerchItem = Tuple[str, int]


def sample_item(dist: Dict[K, float]) -> K:
    return random.choices(
        list(dist.keys()),
        list(dist.values())
    )[0]


class Trip(BaseModel):

    from_city: str
    to_city: str

    merch: List[MerchItem]


class Route(BaseModel):
    id: int
    route: List[Trip]


@dataclass
class RouteGenerator:

    merch_prior_prob: Dict[str, float]
    city_prior_prob: Dict[str, float]

    merch_len_range: Tuple[int, int]
    merch_items_n_range: Tuple[int, int]

    route_len_range: Tuple[int, int]

    def gen_city(
            self,
            city_prior_dist: Optional[Dict[str, float]] = None
    ) -> str:
        if not city_prior_dist:
            city_prior_dist = self.city_prior_prob

        return sample_item(city_prior_dist)

    def gen_merch_item(
        self,
    ) -> MerchItem:
        merch_name = sample_item(self.merch_prior_prob)
        merch_items_n = random.randint(*self.merch_items_n_range)
        return (merch_name, merch_items_n)

    def gen_trip(self, from_city: Optional[str] = None) -> Trip:
        if not from_city:
            from_city = self.gen_city()

        other_city_prior_prob = copy.deepcopy(self.city_prior_prob)
        del other_city_prior_prob[from_city]
        to_city = self.gen_city(other_city_prior_prob)

        merch_len = random.randint(*self.merch_len_range)
        merch = [self.gen_merch_item() for _ in range(merch_len)]

        return Trip(from_city=from_city, to_city=to_city, merch=merch)

    def gen_trips(self, n: int) -> List[Trip]:
        from_city = None
        trips = []

        for _ in range(n):
            trip = self.gen_trip(from_city)
            from_city = trip.to_city
            trips.append(trip)

        return trips


    def gen_route(self, id: int) -> List[Trip]:
        route_len = random.randint(*self.route_len_range)
        route = self.gen_trips(route_len)
        return Route(id=id, route=route)



def gen_uniform_dist(keys: List[Any]) -> Dict[Any, float]:
    return {key: 1/len(keys) for key in keys}


if __name__ == "__main__":
    products = {"Apples", "Pears", "Bananas"}
    cities = {"Amsterdam", "Utrecht", "Delft"}

    merch_uni_dist = gen_uniform_dist(products)
    city_uni_dist = gen_uniform_dist(cities)

    MERCH_LEN_RANGE = (1, 3)
    MERCH_ITEMS_N_RANGE = 100, 300

    ROUTE_LEN_RANGE = (1, 5)

    generator = RouteGenerator(
        merch_uni_dist,
        city_uni_dist,
        MERCH_LEN_RANGE,
        MERCH_ITEMS_N_RANGE,
        ROUTE_LEN_RANGE
    )

    N_ROUTES = 10

    routes = [generator.gen_route(i) for i in range(N_ROUTES)]
    print(routes)
