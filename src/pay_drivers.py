from typing import Final

import pyspark.sql.functions as F
import numpy as np


RETRIEVE_AMOUNT: Final = 500_000
DEPOSIT_FEE_PER_REMAINDER: Final = 0.01


def pay_drivers(cost_df, print_buckets=False):
    # Rename the column for ease of use
    cost_df = cost_df.select(
        F.col("actual_route_uuid").alias("uuid"),
        F.col("euclidian_payment").alias("value"),
    )

    cost_df = cost_df.sort(cost_df["value"].desc())

    rows = cost_df.collect()
    buckets = []
    current_sizes = []
    for row in rows:
        for i, bucket in enumerate(buckets):
            # If this value fits into this bucket, add it
            if current_sizes[i] + row.value <= RETRIEVE_AMOUNT:
                bucket.append((row.uuid, row.value))
                current_sizes[i] += row.value
                break
        else:
            # No existing bucket can accommodate the value, so create a new one
            buckets.append([(row.uuid, row.value)])
            current_sizes.append(row.value)

    # Calculate the remaining space in each bucket
    remainders = [RETRIEVE_AMOUNT - size for size in current_sizes]

    if print_buckets:
        for i, bucket in enumerate(buckets):
            print(f"Bucket {i+1}:")
            for uuid, value in bucket:
                print(f"  UUID: {uuid}, Value: {value}")
                print(f"  Remainder: {remainders[i]}")

    total_remaining = sum(remainders) * DEPOSIT_FEE_PER_REMAINDER
    avg_remaining = np.mean(remainders) * DEPOSIT_FEE_PER_REMAINDER
    total_withdrew = len(buckets)

    # Print the total remaining space across all buckets
    print(f"Total remainder fee: {sum(remainders) * DEPOSIT_FEE_PER_REMAINDER:.2f} EUR")
    print(f"Average remainder fee: {np.mean(remainders) * DEPOSIT_FEE_PER_REMAINDER:.2f} EUR")
    print(f"Total retrievals: {total_withdrew}")

    return total_remaining, avg_remaining, total_withdrew
