from lifelines import CoxPHFitter
import pandas as pd
import numpy as np


def initialize_baseline_hazard(X, T, delta):
    """Initialize baseline hazard using a standard Cox model

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
    baseline_hazard : `np.ndarray`, shape=(n_samples,)
        The baseline hazard function evaluated at each censored time
    """
    n_time_indep_features = X.shape[1]
    X_columns = ['X' + str(j + 1) for j in range(n_time_indep_features)]
    survival_labels = ['T', 'delta']
    data = pd.DataFrame(data=np.hstack((X, T.reshape(-1, 1),
                                        delta.reshape(-1, 1))),
                        columns=X_columns + survival_labels)

    cox = CoxPHFitter()
    cox.fit(data, duration_col='T', event_col='delta')
    baseline_hazard = cox.baseline_hazard_

    return baseline_hazard
