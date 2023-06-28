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

spark = (
    SparkSession.builder.appName("Pay Routes")
    .config("spark.driver.memory", "10G")
    .getOrCreate()
)

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
    planned_routes_df = load_json_to_spark(f"planned_routes_{idx}.json")
    actual_routes_df = load_json_to_spark(f"actual_routes_{idx}.json")
    planned_df = vectorize_routes(planned_routes_df)
    actual_df = vectorize_routes(actual_routes_df)
    return (planned_df, actual_df)
