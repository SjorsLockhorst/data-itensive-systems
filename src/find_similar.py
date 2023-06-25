from time import time
import math

from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf

from data_loader import load_and_vectorize

global_start = time()

vec_start = time()
planned, actual = load_and_vectorize()

# planned = planned.cache()
# actual = actual.cache()

# actual = actual.withColumnRenamed("route", "actual_route")
# planned = planned.drop("original_route_uuid")
# planned = planned.withColumnsRenamed({"uuid": "original_route_uuid", "route": "planned_route"})
#
# joined = actual.join(planned, "original_route_uuid")
# joined = joined.select(["planned_route", "actual_route"]).cache()
# print(joined.head())

print(f"Loading and vectorising took {time() - vec_start}s.")
planned_vecs = planned.drop("route").cache()
actual_vecs = actual.drop("route").cache()

fit_start = time()
brp = BucketedRandomProjectionLSH(
    inputCol="route_vector", outputCol="hashes", numHashTables=2, bucketLength=100
)
model = brp.fit(planned_vecs)
print(f"Fitting LSH took {time() - fit_start}")

transform_start = time()

transformed_planned_df = model.transform(planned_vecs)
transformed_actual_df = model.transform(actual_vecs)
# print(transformed_planned_df.head().hashes[0])
print(f"Transforming data took {time() - transform_start}")

compare_start = time()
result = model.approxSimilarityJoin(
    transformed_actual_df,
    transformed_planned_df,
    threshold=20,
    distCol="EuclideanDistance",
).cache()
print(f"Did {result.count()} comparisons")
print(f"Comparing data {time() - compare_start}")

# Get the nearest neighbor from the transformed_planned_df
min_start = time()
nearest_neighbors = result.groupBy("datasetA").agg(
    F.min("EuclideanDistance").alias("EuclideanDistance")
).cache()
print(f"Found minima in {time() - min_start}s")

join_start = time()

# Join back to get the full row information from planned_df
final_results = nearest_neighbors.join(
    F.broadcast(result),
    ["datasetA", "EuclideanDistance"]
).cache()
print(f"Joined in {time() - join_start}s")

# Evaluate the accuracy
count = final_results.filter(
    final_results.datasetA.original_route_uuid == final_results.datasetB.uuid
).count()


print(f"Number of rows where 'original_route_uuid' equals to 'uuid': {count}")
print(f"Accuracy: {count / actual_vecs.count()}")

print("Calculating cost function based on norm. euclidean distance")


euclid_cost_start = time()

max_dist = final_results.agg(
    F.max(F.col("EuclideanDistance"))).collect()[0][0]
min_dist = final_results.agg(
    F.min(F.col("EuclideanDistance"))).collect()[0][0]

NORMALISATION_RECIPROCAL = 1 / (max_dist - min_dist)

final_results = final_results\
    .withColumn(
        "NormEuclideanDistance",
        (F.col("EuclideanDistance") - min_dist) * NORMALISATION_RECIPROCAL
    )
final_results = final_results\
    .withColumn(
        "Similarity",
        1 - F.col("NormEuclideanDistance")
    )

final_results = final_results\
    .withColumn(
        "EuclidPayment",
        F.col("Similarity") * 1000
    ).cache()

print(
    f"Found cost based on euclidian similiarity in {time() - euclid_cost_start}s")

print(final_results
      .select(
          ["EuclideanDistance", "NormEuclideanDistance", "EuclidPayment"]
      )
      .describe()
      .show())

print(final_results.count())
joined_preds = final_results.select([
    F.col("datasetA.uuid").alias("actual_uuid"),
    F.col("datasetB.uuid").alias("pred_planned_uuid"),
    F.col("datasetA.original_route_uuid").alias("gold_label_planned_uuid"),
    F.col("EuclideanDistance"),
    F.col("NormEuclideanDistance"),
    F.col("Similarity"),
    F.col("EuclidPayment")
])\
    .join(
    actual.select([
        F.col("route").alias("actual_route"),
        F.col("uuid").alias("actual_uuid"),
    ]),
    "actual_uuid")\
    .join(
    planned.select([
        F.col("route").alias("planned_route"),
        F.col("uuid").alias("pred_planned_uuid")
    ]),
    "pred_planned_uuid"
)\
    .cache()


@udf(FloatType())
def calc_payment(planned, actual):

    expected_quantities = {}
    actual_quantities = {}

    quantity_to_move = 0
    # Initialize quantities in warehouses based on expected route
    for trip in planned:
        from_city = trip["from_city"]
        to_city = trip["to_city"]
        merchandise = trip["merch"]

        if to_city not in expected_quantities:
            expected_quantities[from_city] = merchandise.copy()
            quantity_to_move += sum(merchandise.values())
        else:
            for item, expected_quantity in merchandise.items():
                expected_quantities[to_city][item] = expected_quantities[to_city].get(
                    item, 0) + expected_quantity
                quantity_to_move += expected_quantity
                
    cost_per_unit = 1000 / quantity_to_move * 0.1
    # cost_per_unit = 0.1

    # Update quantities in warehouses based on actual route
    for trip in actual:
        from_city = trip["from_city"]
        to_city = trip["to_city"]
        merchandise = trip["merch"]

        if to_city not in actual_quantities:
            actual_quantities[to_city] = merchandise.copy()
        else:
            for item, expected_quantity in merchandise.items():
                actual_quantities[to_city][item] = actual_quantities[to_city].get(
                    item, 0) + expected_quantity

    # Calculate the cost function based on differences in quantities
    payment = 1000
    total_missed_quantity = 0
    for city in actual_quantities:
        if city in expected_quantities:
            for item, expected_quantity in expected_quantities[city].items():
                if item in actual_quantities[city]:
                    actual_quantity = actual_quantities[city][item]
                    missed_quantity = abs(actual_quantity - expected_quantity)
                    total_missed_quantity += missed_quantity
                    payment -=  missed_quantity * cost_per_unit
                else:
                    payment -= expected_quantity * cost_per_unit
                    total_missed_quantity += expected_quantity

        else:
            payment -= sum(actual_quantities[city].values()) * cost_per_unit

    # deviation = total_missed_quantity / quantity_to_move
    #
    # # Define the sigmoid function parameters
    # a = 10  # Amplification factor
    # b = 0.5  # Dampening factor
    #
    # # Calculate the weight using the sigmoid function
    # weight = 1 / (1 + math.exp(-a * (deviation - b)))
    # print(payment)
    # print(deviation)
    # print(weight)
    #
    # payment *= weight
    payment = max(payment, 0)
    # Calculate the payment ensuring it falls between 0 and 1000
    return payment


# PAYMENT_PER_UNIT_DEVIANCE = 0.01
# udf_payment = udf(lambda x, y: calc_payment(
#     PAYMENT_PER_UNIT_DEVIANCE, x, y), FloatType())

print("Arrived at payment")
preds_with_payment = joined_preds.withColumn(
    "Payment", calc_payment("planned_route", "actual_route")).cache()
# print(preds_with_payment.select(["EuclidPayment", "Payment"]).summary().show())
print(preds_with_payment.describe().show())
print(preds_with_payment.stat.corr("Payment", "EuclidPayment"))
print(preds_with_payment.select(["Payment", "EuclidPayment"]).head(10))
# print(preds_with_payment.head(10))


print(f"{time() - global_start}s elapsed in total.")
