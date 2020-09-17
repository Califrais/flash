# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

from datetime import datetime
from lights.history import History
from time import time
from scipy.linalg import block_diag
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
        key : str [optional, default=None]
            if None all history is returned as a dict
            if str then history of the required key is given

        Returns
        -------
        output : dict or list
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

    @staticmethod
    def extract_features(Y, fixed_effect_time_order):
        """Extract the design features from longitudinal data

        Parameters
        ----------
        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The longitudinal data. Each element of the dataframe is a
            pandas.Series
        fixed_effect_time_order : `int`
            Order of fixed effect features

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
            List with the number of the longitudinal measurements for each
            subject

        U_L : `list` of `np.array`
            The fixed-effect features of the simulated longitudinal data arranged by l-th order
        V_L : `list` of `np.array`
            The random-effect features of the simulated longitudinal data arranged by l-th order
        y_L : `list` of `np.array`
            The outcome of the simulated longitudinal data arranged by l-th order
        N_L : `list` of `list`
            The number samples of the simulated longitudinal data arranged by l-th order

        """

        def extract_specified_features(Y_il):
            """Extract the longitudinal data of subject i-th outcome l-th
            into features of the multivariate linear mixed model

            Parameters
            ----------
            Y_il : `pandas.Series`
                The simulated longitudinal data of l-th outcome of i-th subject

            Returns
            -------
            U_il : `np.array`
                The fixed-effect features for of l-th outcome of i-th subject
            Y_il : `np.array`
                The l-th outcome of i-th subject
            n_il : `list`
                The number samples of l-th outcome of i-th subject
            """
            times_il = Y_il.index.values
            y_il = Y_il.values
            N_il = len(times_il)
            U_il = np.ones(N_il)
            for t in range(fixed_effect_time_order):
                U_il = np.c_[U_il, times_il ** (t + 1)]
            return U_il, y_il, N_il

        n, L = Y.shape
        U, V, y, N = [], [], [], []
        U_L, V_L, y_L, N_L = [], [], [], []
        for i in range(n):
            Y_i = Y.iloc[i]
            L = len(Y_i)
            for l in range(L):
                U_il, y_il, N_il = extract_specified_features(Y_i[l])
                V_il = U_il

                if l == 0:
                    U_i = U_il
                    V_i = V_il
                    y_i = y_il
                    N_i = [N_il]

                else:
                    U_i = block_diag(U_i, U_il)
                    V_i = block_diag(V_i, V_il)
                    y_i = np.concatenate((y_i, y_il))

                    #TODO add the required .reshape(-1, 1)

                    N_i.append(N_il)

                if i == 0:
                    U_L.append(U_il)
                    V_L.append(V_il)
                    y_L.append(y_il)
                    N_L.append([N_il])
                else:
                    U_L[l] = np.concatenate((U_L[l], U_il))
                    V_L[l] = block_diag(V_L[l], V_il)
                    y_L[l] = np.concatenate((y_L[l], y_il))
                    N_L[l].append(N_il)

            if i == 0:
                V = V_i
            else:
                V = block_diag(V, V_i)
            U.append(U_i)
            y.append(y_i)
            N.append(N_i)

        return (U, V, y, N), (U_L, V_L, y_L, N_L)
