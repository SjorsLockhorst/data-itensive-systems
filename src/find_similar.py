from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.sql import functions as F


def find_similar(planned, actual):
    brp = BucketedRandomProjectionLSH(
        inputCol="route_vector", outputCol="hashes", numHashTables=2, bucketLength=100
    )
    model = brp.fit(planned)

    transformed_planned_df = model.transform(planned)
    transformed_actual_df = model.transform(actual)

    result = model.approxSimilarityJoin(
        transformed_actual_df,
        transformed_planned_df,
        threshold=float("inf"),
        distCol="EuclideanDistance",
    )
    print(f"{result.count()} size after approxSimilarityJoin")

    # Get the nearest neighbor from the transformed_planned_df
    nearest_neighbors = result.groupBy("datasetA").agg(
        F.min("EuclideanDistance").alias("EuclideanDistance")
    )

    # Join back to get the full row information from planned_df
    similar_df = nearest_neighbors.join(result, ["datasetA", "EuclideanDistance"])

    return similar_df.select(
        F.col("datasetA.uuid").alias("actual_route_uuid"),
        F.col("datasetA.original_route_uuid").alias("original_route_uuid"),
        F.col("datasetB.uuid").alias("predicted_original_route_uuid"),
        F.col("EuclideanDistance"),
    )


def evaluate_accuracy(similar_df, actual_df):
    print("Measuring accuracy")
    count = similar_df.filter(
        similar_df.original_route_uuid == similar_df.predicted_original_route_uuid
    ).count()

    print(f"Number of rows where 'original_route_uuid' equals to 'uuid': {count}")
    print(f"Accuracy: {count / actual_df.count() * 100:.2f}%")
