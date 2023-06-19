from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Pay Routes").getOrCreate()


def load_json_to_spark(file_name):
    df = spark.read.json(file_name)
    return df


if __name__ == "__main__":
    planned_routes_df = load_json_to_spark("planned_routes.json")
    actual_routes_df = load_json_to_spark("actual_routes.json")
