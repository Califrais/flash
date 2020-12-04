# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

from datetime import datetime
from lights.base.history import History
from time import time
import numpy as np


class Learner:
    """The base class for a Solver.
    Not intended for end-users, but for development only.
    It should be sklearn-learn compliant

    Parameters
    ----------
    verbose : `bool`, default=True
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default=10
        Print history information when ``n_iter`` (iteration number) is
        a multiple of ``print_every``
    """

    def __init__(self, verbose=True, print_every=10):
        self.verbose = verbose
        self.print_every = print_every
        self.history = History()

    def _start_solve(self):
        # Reset history
        self.history.clear()
        self.time_start = Learner._get_now()
        self._numeric_time_start = time()

        if self.verbose:
            print("Launching the solver " + self.__class__.__name__ + "...")

    def _end_solve(self):
        self.time_end = self._get_now()
        t = time()
        self.time_elapsed = t - self._numeric_time_start

        if self.verbose:
            print("Done solving using " + self.__class__.__name__ + " in "
                  + "%.2e seconds" % self.time_elapsed)

    @staticmethod
    def _get_now():
        return str(datetime.now()).replace(" ", "_").replace(":", "-")

    def get_history(self, key=None):
        """Return history of the solver

        Parameters
        ----------
        key : `str`, default=None
            if None all history is returned as a dict
            if str then history of the required key is given

        Returns
        -------
        output : `dict` or `list`
            if key is None or key is not in history then output is
                dict containing history of all keys
            if key is not None and key is in history, then output is a list
            containing history for the given key
        """
        val = self.history.values.get(key, None)
        if val is None:
            return self.history.values
        else:
            return val

    def get_history_keys(self):
        """Return names of the elements stored in history

        Returns
        -------
        output : `list`
            list containing names of history keys
        """
        return self.history.values.keys()


def block_diag(l_arr):
    """Create a block diagonal matrix from provided list of arrays.

    Parameters
    ----------
    l_arr : list of arrays, up to 2-D
        Input arrays.

    Returns
    -------
    out : ndarray
        Array with input arrays on the diagonal.
    """
    shapes = np.array([a.shape for a in l_arr])
    out_dtype = np.find_common_type([arr.dtype for arr in l_arr], [])
    out = np.zeros(np.sum(shapes, axis=0), dtype=out_dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = l_arr[i]
        r += rr
        c += cc

    return out


def normalize(X):
    """Normalize X to have mean 0 and std 1

    Parameters
    ----------
    X : `np.ndarray`, shape=(n, d)
        A time-independent features matrix

    Returns
    -------
    X_norm : `np.ndarray`, shape=(n, d)
        The corresponding normilized matrix with mean 0 and std 1
    """
    mean = X.mean(0)
    std = X.std(0)
    X_norm = (X - mean) / std

    return X_norm


def from_ts_to_design_features(Y_il, fixed_effect_time_order):
    """Extracts the design features from a given longitudinal trajectory

    Parameters
    ----------
    Y_il : `pandas.Series`
        A longitudinal trajectory
    fixed_effect_time_order : `int`
        Order of fixed effect features

    Returns
    -------
    U_il : `np.array`
        The corresponding fixed-effect design features
    V_il : `np.array`
        The corresponding random-effect design features
    Y_il : `np.array`
        The corresponding outcomes
    n_il : `list`
        The corresponding number of measurements
    """
    times_il = Y_il.index.values
    y_il = Y_il.values.reshape(-1, 1)
    n_il = len(times_il)
    U_il = np.ones(n_il)
    for t in range(1, fixed_effect_time_order + 1):
        U_il = np.c_[U_il, times_il ** t]
    # linear time-varying features
    V_il = np.ones(n_il)
    V_il = np.c_[V_il, times_il]
    return U_il, V_il, y_il, n_il


def extract_features(Y, fixed_effect_time_order):
    """Extract the design features from longitudinal data

    Parameters
    ----------
    Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
        The longitudinal data. Each element of the dataframe is a
        pandas.Series
    fixed_effect_time_order : `int`
        Order of the higher time monomial considered for the representations of
        the time-varying features corresponding to the fixed effect. The
        dimension of the corresponding design matrix is then equal to
        fixed_effect_time_order + 1

    Returns
    -------
    U : `np.array`, shape=(N_total, q)
        Matrix that concatenates the fixed-effect design features of the
        longitudinal data of all subjects, with N_total the total number
        of longitudinal measurements for all subjects (N_total = sum(N))
    V : `np.array`, shape=(N_total, n_samples x r)
        Bloc-diagonal matrix with the random-effect design features of
        the longitudinal data of all subjects
    y : `np.array`, shape=(N_total,)
        Vector that concatenates all longitudinal outcomes for all
        subjects
    N : `list`, shape=(n_samples,)
        List with the number of longitudinal measurements for each subject

    U_L : `list` of L `np.array`
        Fixed-effect design features arranged by l-th order
    V_L : `list` of L `np.array`
        Random-effect design features arranged by l-th order
    y_L : `list` of L `np.array`
        Longitudinal outcomes arranged by l-th order
    N_L : `list` of L `list`
        Number of longitudinal measurements arranged by l-th order
    """
    n_samples, n_long_features = Y.shape
    U, V, y, N = [], [], [], []
    U_L = [[] for _ in range(n_long_features)]
    V_L = [[] for _ in range(n_long_features)]
    y_L = [[] for _ in range(n_long_features)]
    N_L = [[] for _ in range(n_long_features)]

    for i in range(n_samples):
        Y_i = Y.iloc[i]
        U_i, V_i, y_i, N_i = [], [], np.array([]), []
        for l in range(n_long_features):
            U_il, V_il, y_il, N_il = from_ts_to_design_features(
                Y_i[l], fixed_effect_time_order)

            U_i.append(U_il)
            V_i.append(V_il)
            y_i = np.append(y_i, y_il)
            N_i.append(N_il)

            U_L[l].append(U_il)
            V_L[l].append(V_il)
            y_L[l].append(y_il)
            N_L[l].append(N_il)

        U.append(block_diag(U_i))
        V.append(block_diag(V_i))
        y.append(y_i.reshape(-1, 1))
        N.append(N_i)

    for l in range(n_long_features):
        U_L[l] = np.concatenate(tuple(U_L[l]))
        V_L[l] = block_diag(V_L[l])
        y_L[l] = np.concatenate(tuple(y_L[l]))

    return (U, V, y, N), (U_L, V_L, y_L, N_L)


def logistic_grad(z):
    """Overflow proof computation of 1 / (1 + exp(-z)))
    """
    idx_pos = np.where(z >= 0.)
    idx_neg = np.where(z < 0.)
    res = np.empty(z.shape)
    res[idx_pos] = 1. / (1. + np.exp(-z[idx_pos]))
    res[idx_neg] = 1 - 1. / (1. + np.exp(z[idx_neg]))
    return res


def logistic_loss(z):
    """Overflow proof computation of log(1 + exp(-z))
    """
    idx_pos = np.where(z >= 0.)
    idx_neg = np.where(z < 0.)
    res = np.empty(z.shape)
    res[idx_pos] = np.log(1. + np.exp(-z[idx_pos]))
    z_neg = z[idx_neg]
    res[idx_neg] = -z_neg + np.log(1. + np.exp(z_neg))
    return res


def get_times_infos(T):
    """Get censored times indicators

    Parameters
    ----------
    T : `np.ndarray`, shape=(n_samples,)
        Censored times of the event of interest

    Returns
    -------
    T_u : `np.ndarray`, shape=(J,)
        The J unique censored times of the event of interest

    J : `int`
        The J unique censored times of the event of interest

    indicator_1 : `np.ndarray`, shape=(n_samples, J)
        The indicator matrix for comparing event times (T == T_u)

    indicator_2 : `np.ndarray`, shape=(n_samples, J)
        The indicator matrix for comparing event times (T <= T_u)
    """
    T_u = np.unique(T)
    J = T_u.shape[0]
    indicator_1 = T.reshape(-1, 1) == T_u
    indicator_2 = T.reshape(-1, 1) >= T_u
    return T_u, J, indicator_1, indicator_2


def get_vect_from_ext(v_ext):
    """Obtain the signed coefficient vector from its extension on positive
    and negative parts
    """
    dim = len(v_ext)
    if dim % 2 != 0:
        raise ValueError("``v_ext`` dimension cannot be odd, got %s" % dim)
    v = v_ext[:dim // 2] - v_ext[dim // 2:]
    return v

def get_ext_from_vec(v):
    """Obtain r from extension vector include positive and negative parts
    from the signed coefficient vecto
    """
    dim = len(v)
    v_ext = np.concatenate((v, -v)).reshape(-1, 1)
    v_ext[v_ext < 0] = 0
    return v_ext


def get_xi_from_xi_ext(xi_ext, fit_intercept):
    """Get the time-independent coefficient vector from its extension on
    positive and negative parts

    Parameters
    ----------
    xi_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
        The time-independent coefficient vector decomposed on positive and
        negative parts

    fit_intercept : `bool`
        If `True`, include an intercept in the model for the time independent
        features

    Returns
    -------
    xi_0 : `float`
        The intercept term

    xi : `np.ndarray`, shape=(n_time_indep_features,)
        The time-independent coefficient vector
    """
    dim = len(xi_ext) // 2
    xi = xi_ext[:dim] - xi_ext[dim:]
    if fit_intercept:
        xi_0 = xi[0]
        xi = xi[1:]
    else:
        xi_0 = 0
    return xi_0, xi


def clean_xi_ext(xi_ext, fit_intercept):
    """Removes potential intercept coefficients in the time-independent
    coefficient vector decomposed on positive and negative parts

    Parameters
    ----------
    xi_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
        The time-independent coefficient vector decomposed on positive and
        negative parts

    fit_intercept : `bool`
        If `True`, include an intercept in the model for the time independent
        features

    Returns
    -------
    xi_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
        The time-independent coefficient vector decomposed on positive and
        negative parts without potential intercept coefficients
    """
    if fit_intercept:
        n_time_indep_features = len(xi_ext) // 2
        xi_ext = np.delete(xi_ext, [0, n_time_indep_features + 1])
    return xi_ext
