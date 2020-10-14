from lifelines import CoxPHFitter
import pandas as pd
import numpy as np


def initialize_asso_params(X, T, delta):
    """Initialize the time-independent association parameters and baseline
    hazard using a standard Cox model

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
        The time-independent association parameters

    baseline_hazard : `np.ndarray`, shape=(n_samples,)
        The baseline hazard function evaluated at each censored time
    """
    n_time_indep_features = X.shape[1]
    X_columns = ['X' + str(j + 1) for j in range(n_time_indep_features)]
    survival_labels = ['T', 'delta']
    data = pd.DataFrame(data=np.hstack((X, T.reshape(-1, 1))),
                        columns=X_columns + survival_labels)

    cox = CoxPHFitter()
    cox.fit(data, duration_col='T', event_col='delta')

    gamma_0 = cox.params_.values
    baseline_hazard = cox.baseline_hazard_
    # since the baseline hazard returned is the hazard function when all
    # covariates are set to their mean
    baseline_hazard /= np.exp(gamma_0.dot(data[X_columns].mean()))

    return gamma_0, baseline_hazard
