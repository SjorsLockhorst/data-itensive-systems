from time import perf_counter as time

from pyspark.sql import functions as F

from cost import calc_payment


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
