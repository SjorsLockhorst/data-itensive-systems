import os

import yaml

from data_gen import generate_dataset
from find_similar import run

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
experiment_path = os.path.join(DIR_PATH, "experiments.yaml")

if __name__ == "__main__":
    with open(experiment_path, encoding="utf-8") as f:
        experiment_conf = yaml.safe_load(f)

    for idx, config_path in enumerate(experiment_conf["runs"]):
        if not os.path.exists(f"planned_routes_{idx}.json") or not os.path.exists(f"actual_routes_{idx}.json"):
            generate_dataset(config_path, idx=idx)
        run(idx)
