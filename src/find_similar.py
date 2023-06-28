from time import perf_counter as time
from typing import Final

from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.sql import functions as F

from cost import calc_payment
from data_loader import load_and_vectorize


EUCLIDIAN_THRESHOLD: Final = 14


def run(idx, evaluate_semantic_cost=False):
    global_start = time()

    vec_start = time()
    planned, actual = load_and_vectorize(idx=idx)
    print(f"Loading and vectorising took {time() - vec_start:.4f}s.\n")

    planned_vecs = planned.drop("route")
    actual_vecs = actual.drop("route")

    similar_start = time()
    similar_df = find_similar(planned_vecs, actual_vecs)
    print(f"Finding similar routes in total took {time() - similar_start:.4f}s.\n")

    evaluate_accuracy(similar_df, actual_vecs)
    euclidian_cost_df = calculate_payment(similar_df=similar_df)
    if evaluate_semantic_cost:
        semantic_cost_df = calculate_payment_semantically(similar_df, actual, planned)
        evaluate_accuracy_cost(
            euclidian_cost_df=euclidian_cost_df, semantic_cost_df=semantic_cost_df
        )

    print(f"{time() - global_start:.4f}s elapsed in total.")


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


def calculate_payment(similar_df):
    print("Calculating cost function based on norm. euclidean distance")

    euclid_cost_start = time()

    max_dist = similar_df.agg(F.max(F.col("EuclideanDistance"))).collect()[0][0]
    min_dist = 0

    cost_df = similar_df.withColumn(
        "NormEuclideanDistance",
        (F.col("EuclideanDistance") - min_dist)
        / max_dist,  # * (lower + (upper - lower))
    )
    cost_df = cost_df.withColumn(
        "euclidian_payment", (1 - F.col("NormEuclideanDistance")) * 1000
    )

    print(f"Found cost based on euclidian similiarity in {time() - euclid_cost_start}s")
    return cost_df


def calculate_payment_semantically(similar_df, actual, planned):
    print("Calculate payment semantically")
    joined_preds = (
        similar_df.select(
            [
                F.col("datasetA.uuid").alias("actual_uuid"),
                F.col("datasetB.uuid").alias("planned_uuid"),
            ]
        )
        .join(
            actual.select(
                [
                    F.col("route").alias("actual_route"),
                    F.col("uuid").alias("actual_uuid"),
                ]
            ),
            "actual_uuid",
        )
        .join(
            planned.select(
                [
                    F.col("route").alias("planned_route"),
                    F.col("uuid").alias("planned_uuid"),
                ]
            ),
            "planned_uuid",
        )
    )
    preds_with_payment = joined_preds.withColumn(
        "semantic_payment", calc_payment("planned_route", "actual_route")
    )
    return preds_with_payment


def evaluate_accuracy_cost(euclidian_cost_df, semantic_cost_df):
    preds_with_payment = euclidian_cost_df.join(
        semantic_cost_df,
        euclidian_cost_df.datasetA.uuid == semantic_cost_df.actual_uuid,
    )
    print(
        preds_with_payment.select(["euclidian_payment", "semantic_payment"])
        .summary()
        .show()
    )
    print(preds_with_payment.describe().show())
    print(preds_with_payment.stat.corr("semantic_payment", "euclidian_payment"))


if __name__ == "__main__":
    run(idx=1)
