# License: BSD 3 clause

"""
This module implements hyper-parameters optimization, with classes specific to each algorithm, using `hyperopt`,
used in experiments with hyper-parmeters optimization.

Inspired from
[CatBoost's experiments codes](https://github.com/catboost/benchmarks/blob/master/quality_benchmarks/experiment.py)
"""


from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
import numpy as np
import os
import time
from datetime import datetime
import pickle as pkl
import sys

sys.path.extend([".", ".."])

from lights.inference import prox_QNEM
from lifelines.utils import concordance_index as c_index_score

# TODO: to add regressors for every Experiment


class Experiment(object):
    """
    Base class, for hyper-parameters optimization experiments with hyperopt
    """

    def __init__(
        self,
        bst_name=None,
        hyperopt_evals=50,
        output_folder_path="./",
        verbose=True,
    ):
        self.bst_name = bst_name
        self.best_loss = np.inf
        self.hyperopt_evals, self.hyperopt_eval_num = hyperopt_evals, 0
        self.output_folder_path = os.path.join(output_folder_path, "")
        self.default_params, self.best_params = None, None
        self.verbose = verbose

        # to specify definitions in particular experiments
        self.title = None
        self.space = None
        self.trials = None

        self.metric = "c_index"

    def optimize_params(
        self,
        data_train=None,
        data_val=None,
        max_evals=None,
        verbose=True,
    ):
        max_evals = max_evals or self.hyperopt_evals
        self.trials = Trials()
        self.hyperopt_eval_num, self.best_loss = 0, np.inf

        _ = fmin(
            fn=lambda params: self.run(
                data_train,
                data_val,
                params,
                verbose=verbose,
            ),
            space=self.space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=self.trials,
        )

        self.best_params = self.trials.best_trial["result"]["params"]
        if self.verbose:
            now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            filename = (
                "best_params_results_"
                + str(self.bst_name)
                + "_"
                + str(self.hyperopt_evals)
                # + "_"
                # + now
                + ".pickle"
            )

            with open(self.output_folder_path + filename, "wb") as f:
                pkl.dump(
                    {
                        "datetime": now,
                        "max_hyperopt_eval": self.hyperopt_evals,
                        "result": self.trials.best_trial["result"],
                        "trials": self.trials,
                    },
                    f,
                )
        return self.trials.best_trial["result"]

    def run(
        self,
        data_train,
        data_val,
        params=None,
        verbose=False,
    ):
        params = params or self.default_params
        params = self.preprocess_params(params)
        start_time = time.time()
        bst = self.fit(params, data_train)
        fit_time = time.time() - start_time
        score = self.score(bst, data_val)
        evals_result = -max(score, 1 - score)
        results = {
            "loss": evals_result,
            "fit_time": fit_time,
            "status": STATUS_FAIL if np.isnan(evals_result) else STATUS_OK,
            "params": params.copy(),
        }

        self.best_loss = min(self.best_loss, results["loss"])
        self.hyperopt_eval_num += 1
        results.update(
            {"hyperopt_eval_num": self.hyperopt_eval_num, "best_loss": self.best_loss}
        )

        if verbose:
            print(
                "[{0}/{1}]\teval_time={2:.2f} sec\tcurrent_{3}={4:.6f}\tmin_{3}={5:.6f}".format(
                    self.hyperopt_eval_num,
                    self.hyperopt_evals,
                    fit_time,
                    self.metric,
                    results["loss"],
                    self.best_loss,
                )
            )
        return results

    def fit(
        self,
        params,
        data_train
    ):
        raise NotImplementedError("Method fit is not implemented.")

    def score(self, bst, data_test):
        raise NotImplementedError("Method score is not implemented.")

    def preprocess_params(self, params):
        raise NotImplementedError("Method preprocess_params is not implemented.")


class LightsExperiment(Experiment):
    """
    Experiment class for sklearn's RandomForestClassifier, for hyper-parameters optimization experiments with hyperopt
    """

    def __init__(
        self,
        max_hyperopt_evals=50,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            "lights",
            max_hyperopt_evals,
            output_folder_path,
        )

        # hard-coded params search space here
        #TODO: Update later
        zeta_xi_max, zeta_gamma_max = 1, 1
        self.space = {
            "l_pen_EN": hp.uniform('l_pen_EN', zeta_xi_max * 1e-4, zeta_xi_max),
            "l_pen_SGL":hp.uniform('l_pen_SGL', zeta_gamma_max * 1e-4, zeta_gamma_max),
        }
        # hard-coded default params here
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "lights"

    def preprocess_params(self, params):
        params_ = params.copy()
        return params_

    def fit(
        self,
        params,
        data_train,
    ):
        clf = prox_QNEM(**params)
        X_train, Y_train, T_train, delta_train, Y_rep_train = data_train
        clf.fit(X_train, Y_train, T_train, delta_train, Y_rep_train)
        return clf, None

    def score(self, bst, data_test):
        X_test, Y_test, T_test, delta_test, Y_rep_test = data_test
        preds = bst.score(X_test, Y_test, T_test, delta_test, Y_rep_test)
        return preds

