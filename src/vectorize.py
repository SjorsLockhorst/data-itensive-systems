from itertools import product
import math
from typing import Final

import yaml
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf

from data_gen import CONFIG_PATH

with open(CONFIG_PATH, encoding="utf-8") as f:
    config = yaml.safe_load(f)

_merch_items = config["merch_items"]
_cities = config["cities"]

MERCH_ITEM_MIN: Final = min(
    config["merch_sampler_map"]["low"] - config["merch_item_noise_map"]["low"], 0
)
MERCH_ITEM_MAX: Final = (
    config["merch_sampler_map"]["high"] + config["merch_item_noise_map"]["high"]
)

NORMALIZATION_RECIPROCAL: Final = 1 / (MERCH_ITEM_MAX - MERCH_ITEM_MIN)
VEC_MAX_DIST = math.sqrt(len(_cities) * (len(_cities) - 1) * len(_merch_items))

_combinations = list(product(_cities, _cities, _merch_items))
_combinations = [
    (from_city, to_city, merch)
    for from_city, to_city, merch in _combinations
    if from_city != to_city
]
VECTOR_SIZE: Final = len(_combinations)
VECTOR_MAP: Final = {
    combination: index for index, combination in enumerate(_combinations)
}


@udf(VectorUDT())
def create_vectors(routes):
    vector_elements = {}

    for route in routes:
        for merch_name, merch_weight in route.merch.items():
            normalized_weight = (
                merch_weight - MERCH_ITEM_MIN
            ) * NORMALIZATION_RECIPROCAL
            vector_index = VECTOR_MAP[(route.from_city, route.to_city, merch_name)]
            if vector_index in vector_elements:
                vector_elements[vector_index] += normalized_weight + NORMALIZATION_RECIPROCAL
            else:
                vector_elements[vector_index] = normalized_weight + NORMALIZATION_RECIPROCAL

    return Vectors.sparse(VECTOR_SIZE, vector_elements)


def vectorize_routes(routes_df):
    return routes_df.withColumn("route_vector", create_vectors("route"))
