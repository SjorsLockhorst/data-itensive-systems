import os
from typing import Final

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType,
)
from vectorize import vectorize_routes

DIR_PATH: Final = os.path.dirname(os.path.realpath(__file__))
DATA_DIR: Final = os.path.join(DIR_PATH, "..", "data")

spark = (
    SparkSession.builder.appName("Pay Routes")
    .config("spark.driver.memory", "10G")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

schema = StructType(
    [
        StructField(
            "route",
            ArrayType(
                StructType(
                    [
                        StructField("from_city", StringType(), nullable=True),
                        StructField("to_city", StringType(), nullable=True),
                        StructField(
                            "merch", MapType(StringType(), IntegerType()), nullable=True
                        ),
                    ]
                )
            ),
            nullable=True,
        ),
        StructField("uuid", StringType(), nullable=True),
        StructField("original_route_uuid", StringType(), nullable=True),
    ]
)


def load_json_to_spark(file_name):
    return spark.read.json(file_name, schema=schema)


def load_and_vectorize(idx=0):

    planned_routes_path = os.path.join(DATA_DIR, f"planned_routes_{idx}.json")
    actual_routes_path = os.path.join(DATA_DIR, f"actual_routes_{idx}.json")
    planned_routes_df = load_json_to_spark(planned_routes_path)
    actual_routes_df = load_json_to_spark(actual_routes_path)
    planned_df = vectorize_routes(planned_routes_df, idx)
    actual_df = vectorize_routes(actual_routes_df, idx)
    return (planned_df, actual_df)
