from statsmodels.duration.hazard_regression import PHReg
import pandas as pd
import numpy as np

def initialize_asso_params(self, X, T, delta):
    """Initialize the time-independent associated parameters and baseline
    Hazard by standard Cox model

    Parameters
    ----------
    X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
        The time-independent features matrix

    T : `np.ndarray`, shape=(n_samples,)
        Censored times of the event of interest

    delta : `np.ndarray`, shape=(n_samples,)
        Censoring indicator

    Returns
    -------
    gamma_0 : `np.ndarray`, shape=(n_time_indep_features,)
        The time-independent associated parameters

    baseline_hazard : `np.ndarray`, shape=(n_samples,)
        The baseline Hazard
    """
    n_time_indep_features = X.shape[1]
    X_columns = ['X' + str(j + 1) for j in range(X.shape[1])]
    data = pd.DataFrame(data=np.hstack((X, T.reshape(-1, 1))),
                        columns=X_columns + ['T'])
    cox = PHReg.from_formula("T ~ " + ' + '.join(X_columns), data,
                             status=delta, ties="breslow")
    rslt = cox.fit()

    gamma_0 = rslt.params
    baseline_hazard = rslt.baseline_cumulative_hazard_function[0](T)

    return gamma_0, baseline_hazard