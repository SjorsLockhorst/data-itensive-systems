import math
import os
from itertools import product
from typing import Final

import yaml
from data_gen import EXP_DIR
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf


def get_vec_max_dist(idx):
    with open(os.path.join(EXP_DIR, f"{idx}.yaml"), encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _merch_items = config["merch_items"]
    _cities = config["cities"]
    return math.sqrt(len(_cities) * (len(_cities) - 1) * len(_merch_items))

VEC_MAX_DIST = get_vec_max_dist(1)


def get_vec_params(idx):
    with open(os.path.join(EXP_DIR, f"{idx}.yaml"), encoding="utf-8") as f:
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
    return VECTOR_SIZE, VECTOR_MAP, NORMALIZATION_RECIPROCAL, MERCH_ITEM_MIN


def create_vectors(df, vec_size, vec_map, norm_reciprocal, merch_item_min):


    @udf(VectorUDT())
    def _create_vectors(routes):
        vector_elements = {}

        for route in routes:
            for merch_name, merch_weight in route.merch.items():
                normalized_weight = (
                    merch_weight - merch_item_min
                ) * norm_reciprocal
                vector_index = vec_map[(route.from_city, route.to_city, merch_name)]
                if vector_index in vector_elements:
                    vector_elements[vector_index] += normalized_weight + norm_reciprocal
                else:
                    vector_elements[vector_index] = normalized_weight + norm_reciprocal

        return Vectors.sparse(vec_size, vector_elements)

    return df.withColumn("route_vector", _create_vectors(df["route"]))


def vectorize_routes(routes_df, idx):

    vec_params = get_vec_params(idx)
    return create_vectors(routes_df, *vec_params)
