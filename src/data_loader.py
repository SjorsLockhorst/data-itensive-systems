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

spark = SparkSession.builder.appName("Pay Routes").getOrCreate()

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
    ]
)


def load_json_to_spark(file_name):
    return spark.read.json(file_name, schema=schema)


if __name__ == "__main__":
    planned_routes_df = load_json_to_spark("planned_routes.json")
    actual_routes_df = load_json_to_spark("actual_routes.json")
    df = vectorize_routes(planned_routes_df)
    df.show()
