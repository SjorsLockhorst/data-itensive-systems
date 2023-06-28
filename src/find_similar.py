from time import perf_counter as time

from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.sql import functions as F


def find_similar(planned, actual):
    brp = BucketedRandomProjectionLSH(
        inputCol="route_vector", outputCol="hashes", numHashTables=2, bucketLength=100
    )
    fit_start = time()
    model = brp.fit(planned)
    print(f"Fitting LSH took {time() - fit_start:.4f}s")

    transform_start = time()
    transformed_planned_df = model.transform(planned)
    transformed_actual_df = model.transform(actual)
    print(f"Transforming data took {time() - transform_start:.4f}s")

    compare_start = time()
    result = model.approxSimilarityJoin(
        transformed_actual_df,
        transformed_planned_df,
        threshold=float("inf"),
        distCol="EuclideanDistance",
    )
    #  print(f"Did {result.count()} comparisons")
    print(f"Comparing data {time() - compare_start:.4f}s")

    # Get the nearest neighbor from the transformed_planned_df
    min_start = time()
    nearest_neighbors = result.groupBy("datasetA").agg(
        F.min("EuclideanDistance").alias("EuclideanDistance")
    )
    print(f"Found distance minima in {time() - min_start:.4f}s")

    join_start = time()
    # Join back to get the full row information from planned_df
    similar_df = nearest_neighbors.join(result, ["datasetA", "EuclideanDistance"])
    print(f"Joined back based on distance minima {time() - join_start:.4f}s")
    return similar_df


def evaluate_accuracy(similar_df, actual_df):
    count = similar_df.filter(
        similar_df.datasetA.original_route_uuid == similar_df.datasetB.uuid
    ).count()

    print(f"Number of rows where 'original_route_uuid' equals to 'uuid': {count}")
    print(f"Accuracy: {count / actual_df.count() * 100:.2f}%")
