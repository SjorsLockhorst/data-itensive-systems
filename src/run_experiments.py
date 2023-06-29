import os
from time import perf_counter as time

import yaml

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
experiment_path = os.path.join(DIR_PATH, "experiments.yaml")


def run(idx, measure_accuracy=False, evaluate_semantic_cost=False):
    global_start = time()

    vec_start = time()
    planned, actual = load_and_vectorize(idx=idx)
    print(f"Loading and vectorising took {time() - vec_start:.4f}s.\n")

    planned_vecs = planned.drop("route")
    actual_vecs = actual.drop("route")

    similar_start = time()
    similar_df = find_similar(planned_vecs, actual_vecs)
    print(f"Finding similar routes in total took {time() - similar_start:.4f}s.\n")

    if measure_accuracy:
        evaluate_accuracy(similar_df, actual_vecs)

    euclidian_cost_df = calculate_payment(similar_df=similar_df)

    if evaluate_semantic_cost:
        semantic_cost_df = calculate_payment_semantically(similar_df, actual, planned)
        evaluate_accuracy_cost(
            euclidian_cost_df=euclidian_cost_df, semantic_cost_df=semantic_cost_df
        )

    pay_drivers(cost_df=euclidian_cost_df)
    print(f"{time() - global_start:.4f}s elapsed in total.")


if __name__ == "__main__":
    with open(experiment_path, encoding="utf-8") as f:
        experiment_conf = yaml.safe_load(f)

    for idx, config_path in experiment_conf["runs"].items():
        if not os.path.exists(f"planned_routes_{idx}.json") or not os.path.exists(
            f"actual_routes_{idx}.json"
        ):
            generate_dataset(config_path, idx=idx)
        run(idx)
