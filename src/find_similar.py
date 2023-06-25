from time import time

from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.sql import functions as F

from data_loader import load_and_vectorize

global_start = time()

vec_start = time()
planned, actual = load_and_vectorize()
print(f"Loading and vectorising took {time() - vec_start}s.")
planned = planned.drop("route").cache()
actual = actual.drop("route").cache()

fit_start = time()
brp = BucketedRandomProjectionLSH(
    inputCol="route_vector", outputCol="hashes", numHashTables=2, bucketLength=100
)
model = brp.fit(planned)
print(f"Fitting LSH took {time() - fit_start}")

transform_start = time()

transformed_planned_df = model.transform(planned)
transformed_actual_df = model.transform(actual)
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
print(f"Accuracy: {count / actual.count()}")

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
        "Payment",
        F.col("Similarity") * 1000
    ).cache()

print(
    f"Found cost based on euclidian similiarity in {time() - euclid_cost_start}s")

print(final_results
      .select(
          ["EuclideanDistance", "NormEuclideanDistance", "Payment"]
      )
      .describe()
      .show())


print(f"{time() - global_start}s elapsed in total.")
