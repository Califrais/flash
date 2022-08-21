# License: BSD 3 clause

"""
This script produces hyper-parameters optimization experiments (Table 1 and 3) from the WildWood's paper.

"""


import sys
import os
import subprocess
from time import time
from datetime import datetime
import logging
import pickle as pkl
import numpy as np
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_consistent_length, _check_sample_weight

sys.path.extend([".", ".."])
from lights.datasets import load_PBC_Seq

from experiment import (
    LightsExperiment,
)
from lifelines.utils import concordance_index as c_index_score

def get_train_sample_weights(X, y, sample_weight=None):
    check_consistent_length(X, y)
    sample_weight_ = _check_sample_weight(sample_weight, X, dtype=np.float32)

    return sample_weight_


def set_experiment(
    learner_name,
    max_hyperopt_eval,
    output_folder_path,
):
    experiment_setting = {
        "Lights": LightsExperiment(
            max_hyperopt_evals=max_hyperopt_eval,
            output_folder_path=output_folder_path,
        ),
    }
    return experiment_setting[learner_name]


def set_dataloader(dataset_name):
    loaders_mapping = {
        "PBC_Seq": load_PBC_Seq,
    }
    return loaders_mapping[dataset_name]


def run_hyperopt(
    dataset,
    learner_name,
    max_hyperopt_eval,
    results_dataset_path,
):
    col_data = []
    col_learner = []
    col_fit_time = []
    col_predict_time = []
    col_Cindex = []
    col_Cindex_train = []

    col_data.append(dataset.name)
    col_learner.append(learner_name)


    random_state_ = random_states["data_extract_random_state"]
    data_lights_train, Y_rep_train, data_competing_train, \
    data_lights_test, Y_rep_test, data_competing_test = \
        dataset.split(random_state=random_state_)
    data_lights_tr, Y_rep_tr, data_competing_tr, \
    data_lights_val, Y_rep_val, data_competing_val = \
        dataset.split(random_state=random_state_)

    data_lights_train_extract, data_competing_train_extract = \
        dataset.extract(data_lights_train, data_competing_train)
    data_lights_test_extract, data_competing_test_extract = \
        dataset.extract(data_lights_test, data_competing_test)
    data_lights_tr_extract, data_competing_tr_extract = \
        dataset.extract(data_lights_tr, data_competing_tr)
    data_lights_val_extract, data_competing_val_extract = \
        dataset.extract(data_lights_val, data_competing_val)

    exp = set_experiment(
        learner_name,
        max_hyperopt_eval,
        results_dataset_path,
    )

    print("Run train-val hyperopt exp...")
    if learner_name == "lights":
        data_tr = data_lights_tr_extract + (Y_rep_tr,)
        data_val = data_lights_val_extract + (Y_rep_val, )
        data_train = data_lights_train_extract + (Y_rep_train, )
        data_test = data_lights_test_extract + (Y_rep_train,)
    else:
        data_tr = data_competing_tr_extract
        data_val = data_competing_val_extract
        data_train = data_competing_train_extract
        data_test = data_competing_test_extract
    tuned_cv_result = exp.optimize_params(
        data_train=data_tr,
        data_val=data_val,
        max_evals=max_hyperopt_eval,
        verbose=True,
    )

    print("Run fitting with tuned params...")
    fit_time_list, predict_time_list = [], []

    (
        Cindex_list,
        Cindex_train_list,
        brier_score_list,
        brier_score_train_list,
        cumulative_dynamic_auc_list,
        cumulative_dynamic_auc_train_list,
        integrated_brier_score_list,
        integrated_brier_score_train_list,
    ) = ([], [], [], [], [], [], [], [])

    for fit_seed in fit_seeds:
        tic = time()
        model, _ = exp.fit(
            tuned_cv_result["params"],
            data_train,
        )
        toc = time()
        fit_time = toc - tic
        logging.info("Fitted %s in %.2f seconds" % (learner_name, fit_time))
        fit_time_list.append(fit_time)
        tic = time()
        Cindex = exp.score(model, data_test)
        Cindex_train = exp.predict(model, data_train)
        toc = time()
        predict_time = toc - tic
        predict_time_list.append(predict_time)
        logging.info("Predict %s in %.2f seconds" % (learner_name, predict_time))

        Cindex = max(Cindex, 1 - Cindex)
        Cindex_train = max(Cindex_train, 1 - Cindex_train)
        Cindex_list.append(Cindex)
        Cindex_train_list.append(Cindex_train)

    Cindex, Cindex_train = (
        np.mean(Cindex_list),
        np.mean(Cindex_train_list),
    )

    col_Cindex.append(Cindex)
    col_Cindex_train.append(Cindex_train)

    col_fit_time.append(np.mean(fit_time_list))
    col_predict_time.append(np.mean(predict_time_list))

    logging.info(
        "Cindex= %.2f"
        % (
            float(Cindex)
        )
    )

    results = pd.DataFrame(
        {
            "dataset": col_data,
            "learner": col_learner,
            "fit_time": col_fit_time,
            "predict_time": col_predict_time,
            "Cindex": col_Cindex,
            "Cindex_train": col_Cindex_train,
        }
    )

    return results


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learner_name",
        choices=[
            "Lights",
        ],
    )
    parser.add_argument(
        "--dataset_name",
        choices=[
            "PBC_Seq",
        ],
    )
    parser.add_argument("-n", "--hyperopt_evals", type=int, default=50)
    parser.add_argument("-o", "--output_folder_path", default=None)
    parser.add_argument("--random_state_seed", type=int, default=42)

    args = parser.parse_args()

    learner_name = args.learner_name
    loader = set_dataloader(args.dataset_name)
    max_hyperopt_eval = args.hyperopt_evals
    random_state_seed = args.random_state_seed

    if args.output_folder_path is None:
        if not os.path.exists("results"):
            os.mkdir("results")
        results_home_path = "results/"
    else:
        results_home_path = args.output_folder_path

    random_states = {
        "data_extract_random_state": random_state_seed,
        "train_val_split_random_state": 1 + random_state_seed,
        "expe_random_state": 2 + random_state_seed,
    }
    fit_seeds = [0, 1, 2, 3, 4]

    logging.info("=" * 128)
    dataset = loader()
    logging.info("Launching experiments for %s" % dataset.name)

    if not os.path.exists(results_home_path + dataset.name):
        os.mkdir(results_home_path + dataset.name)
    results_dataset_path = results_home_path + dataset.name + "/"

    results = run_hyperopt(
        dataset,
        learner_name,
        max_hyperopt_eval,
        results_dataset_path,
    )
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Get the commit number as a string
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    commit = commit.decode("utf-8").strip()

    filename = (
        "exp_hyperopt_"
        + "_"
        + str(learner_name)
        + "_"
        + str(max_hyperopt_eval)
        # + "_"
        # + now
        + ".pickle"
    )

    with open(results_dataset_path + filename, "wb") as f:
        pkl.dump(
            {
                "datetime": now,
                "commit": commit,
                "max_hyperopt_eval": max_hyperopt_eval,
                "results": results,
            },
            f,
        )

    logging.info("Saved results in file %s" % results_dataset_path + filename)
