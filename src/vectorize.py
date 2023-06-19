from itertools import product
from typing import Final

import yaml
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf

from data_gen import CONFIG_PATH

with open(CONFIG_PATH, encoding="utf-8") as f:
    config = yaml.safe_load(f)

_merch_items = config["merch_items"]
_cities = config["cities"]

MERCH_ITEM_MIN: Final = config["merch_sampler_map"]["low"]
MERCH_ITEM_MAX: Final = config["merch_sampler_map"]["high"]

_combinations = list(product(_cities, _cities, _merch_items))
# Make sure from_city and to_city are not the same
_combinations = [
    (from_city, to_city, merch)
    for from_city, to_city, merch in _combinations
    if from_city != to_city
]
VECTOR_SIZE: Final = len(_combinations)
# Generate a hashmap from combination to index
VECTOR_MAP: Final = {
    combination: index for index, combination in enumerate(_combinations)
}


@udf(VectorUDT())
def create_vectors(routes):
    vector_elements = []
    for route in routes:
        for merch_name, merch_weight in route.merch.items():
            # Normalize the weight
            normalized_weight = (merch_weight - MERCH_ITEM_MIN) / (
                MERCH_ITEM_MAX - MERCH_ITEM_MIN
            )
            vector_elements.append(
                (
                    VECTOR_MAP[(route.from_city, route.to_city, merch_name)],
                    normalized_weight,
                )
            )

    return Vectors.sparse(VECTOR_SIZE, vector_elements)


# Flatten the merchandise column to get product and its weight
def vectorize_routes(routes_df):
    return routes_df.withColumn("route_vector", create_vectors("route"))
