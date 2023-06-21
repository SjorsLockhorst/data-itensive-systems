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
    inputCol="route_vector", outputCol="hashes", numHashTables=10, bucketLength=100
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
    threshold=6,
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
final_joined_result = nearest_neighbors.join(
    F.broadcast(result),
    ["datasetA", "EuclideanDistance"]
).cache()
print(f"Joined in {time() - join_start}s")

# Evaluate the accuracy
count = final_joined_result.filter(
    final_joined_result.datasetA.original_route_uuid == final_joined_result.datasetB.uuid
).count()
average_dist = final_joined_result.agg(F.mean(F.col("EuclideanDistance"))).collect()[0][0]
std_dist = final_joined_result.agg(F.stddev(F.col("EuclideanDistance"))).collect()[0][0]

print(f"Number of rows where 'original_route_uuid' equals to 'uuid': {count}")
print(f"Accuracy: {count / actual.count()}")
print(f"EuclideanDistance: M={average_dist:.2f}; SD={std_dist:.2f}")
print(f"{time() - global_start}s elapsed in total.")

