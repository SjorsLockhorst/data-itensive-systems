from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.sql import functions as F

from data_loader import load_and_vectorize

planned, actual = load_and_vectorize()
planned = planned.drop("route")
actual = actual.drop("route")

brp = BucketedRandomProjectionLSH(
    inputCol="route_vector", outputCol="hashes", bucketLength=3, numHashTables=1
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


# Get the nearest neighbor from the transformed_planned_df
nearest_neighbors = result.groupBy("datasetA").agg(
    F.min("EuclideanDistance").alias("EuclideanDistance")
)
# Join back to get the full row information from planned_df
nearest_neighbors = nearest_neighbors.join(result, ["datasetA", "EuclideanDistance"])

# Evaluate the accuracy
count = nearest_neighbors.filter(
    nearest_neighbors.datasetA.original_route_uuid == nearest_neighbors.datasetB.uuid
).count()
print(f"Number of rows where 'original_route_uuid' equals to 'uuid': {count}")
print(f"Accuracy: {count / actual.count()}")
