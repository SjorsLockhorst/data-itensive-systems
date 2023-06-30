import os
from time import perf_counter as time
from typing import Final

import yaml
import pandas as pd
import pyspark.sql.functions as F

from cost_calculations import (
    calculate_payment,
    calculate_payment_semantically,
    evaluate_accuracy_cost,
)
from data_gen import generate_dataset
from data_loader import load_and_vectorize
from find_similar import evaluate_accuracy, find_similar
from pay_drivers import pay_drivers

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
EXPERIMENTS_DIR = os.path.join(DIR_PATH, "..", "experiments")
experiments = sorted([os.path.join(EXPERIMENTS_DIR, experiment_file)
               for experiment_file in os.listdir(EXPERIMENTS_DIR)])
DATA_DIR: Final = os.path.join(DIR_PATH, "..", "data")


def run(idx, measure_accuracy=False, evaluate_semantic_cost=False, threshold=20):
    global_start = time()

    planned, actual = load_and_vectorize(idx=idx)

    planned_vecs = planned.drop("route")
    actual_vecs = actual.drop("route")

    similar_df, comparisons = find_similar(planned_vecs, actual_vecs, threshold=threshold)
    similar_df.cache()

    acc = None

    if measure_accuracy:
        acc = evaluate_accuracy(similar_df, actual_vecs)

    euclid_payment_mean = None
    euclid_payment_stdev = None

    euclidian_cost_df = calculate_payment(similar_df=similar_df, norm_weight=1/threshold).cache()
    euclid_payment_mean = euclidian_cost_df.select(
        F.mean(euclidian_cost_df.euclidian_payment)).collect()[0][0]
    euclid_payment_stdev = euclidian_cost_df.select(
        F.stddev(euclidian_cost_df.euclidian_payment)).collect()[0][0]

    semantic_payment_mean = None
    semantic_payment_stdev = None

    corr = None

    if evaluate_semantic_cost:
        semantic_cost_df = calculate_payment_semantically(
            similar_df, actual, planned).cache()

        semantic_payment_mean = semantic_cost_df.select(
            F.mean(semantic_cost_df.semantic_payment)).collect()[0][0]
        semantic_payment_stdev = semantic_cost_df.select(
            F.stddev(semantic_cost_df.semantic_payment)).collect()[0][0]

        corr = evaluate_accuracy_cost(
            euclidian_cost_df=euclidian_cost_df, semantic_cost_df=semantic_cost_df
        )

    total_remaining, avg_remaining, total_withdrew = pay_drivers(
        cost_df=euclidian_cost_df)

    total_time = time() - global_start

    print(f"{total_time:.4f}s elapsed in total.")
    results = [
        total_time,
        comparisons,
        acc,
        euclid_payment_mean,
        euclid_payment_stdev,
        semantic_payment_mean,
        semantic_payment_stdev,
        corr,
        total_remaining,
        avg_remaining,
        total_withdrew
    ]
    return results


if __name__ == "__main__":
    INCLUDE_ANALYTICS = False
    FORCE_REGENERATE_DATA = False
    MEASURE_ACCURACY = True
    EVALUATE_SEMANTIC_COST = True
    THRESHOLD = 6.5

    result_list = []
    for experiment_path in experiments:
        with open(experiment_path, encoding="utf-8") as f:
            experiment_conf = yaml.safe_load(f)

        idx = int(os.path.splitext(os.path.split(experiment_path)[1])[0])

        print(idx)
        if idx != 3:
            continue

        planned_routes_path = os.path.join(
            DATA_DIR, f"planned_routes_{idx}.json")
        actual_routes_path = os.path.join(
            DATA_DIR, f"actual_routes_{idx}.json")

        if not os.path.exists(planned_routes_path) or not os.path.exists(
                actual_routes_path
        ) or FORCE_REGENERATE_DATA:
            generate_dataset(experiment_path, idx=idx)

        results = run(
                idx,
                measure_accuracy=MEASURE_ACCURACY,
                evaluate_semantic_cost=EVALUATE_SEMANTIC_COST,
                threshold=THRESHOLD
        )
        info = [idx, experiment_conf["n_planned_routes"],
                experiment_conf["n_actual_routes"]]
        all_results = info + results
        print(all_results)
        result_list.append(all_results)

    results_df = pd.DataFrame(
        result_list,
        columns=["idx",
            "n_planned",
            "n_actual",
            "runtime",
            "total_comparisons",
            "accuracy",
            "avg_euclid_payment",
            "std_euclid_payment",
            "avg_semantic_payment",
            "std_semantic_payment",
            "semantic_euclid_payment_corr",
            "total_remainder_fee",
            "avg_remainder_fee",
            "n_withdrawls",
        ],
        index="idx"
    )
    print(results_df)
