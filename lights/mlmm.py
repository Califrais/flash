# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

from lights.base import Learner
import numpy as np


class MLMM(Learner):
    """EM Algorithm for fitting a multivariate linear mixed model

    Parameters
    ----------
    tol : `float`, default=1e-5
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

    max_iter : `int`, default=100
        Maximum number of iterations of the solver

    verbose : `bool`, default=True
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default=10
        Print history information when ``n_iter`` (iteration number) is
        a multiple of ``print_every``
    """

    def __init__(self, max_iter=100, verbose=False, print_every=10, tol=1e-5):
        Learner.__init__(self, verbose=verbose, print_every=print_every)
        self.max_iter = max_iter
        self.verbose = verbose
        self.print_every = print_every
        self.tol = tol

        # Attributes that will be instantiated afterwards
        self.beta = None
        self.D = None
        self.phi = None

    @staticmethod
    def log_lik(Y):
        """Computes the log-likelihood of the multivariate linear mixed model

        Parameters
        ----------
        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The simulated longitudinal data. Each element of the dataframe is
            a pandas.Series

        Returns
        -------
        output : `float`
            The value of the log-likelihood
        """
        log_lik = 0
        # TODO: Van-Tuan
        return log_lik

    def fit(self, Y):
        """Fit the multivariate linear mixed model

        Parameters
        ----------
        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The simulated longitudinal data. Each element of the dataframe is
            a pandas.Series
        """
        verbose = self.verbose
        max_iter = self.max_iter
        print_every = self.print_every
        tol = self.tol
        self._start_solve()

        # We initialize parameters by fitting univariate linear mixed models
        ulmm = ULMM()
        ulmm.fit(Y)
        beta = ulmm.beta
        D = ulmm.D
        phi = ulmm.phi

        log_lik = self.log_lik(Y)
        obj = -log_lik
        rel_obj = 1.
        self.history.update(n_iter=0, obj=obj, rel_obj=rel_obj)
        if verbose:
            self.history.print_history()

        n_iter = 0
        for n_iter in range(max_iter):
            if n_iter % print_every == 0:
                self.history.update(n_iter=n_iter, obj=obj,
                                    rel_obj=rel_obj)
                if verbose:
                    self.history.print_history()
            # E-Step
            # TODO Van Tuan

            # M-Step
            # TODO Van Tuan

            prev_obj = obj
            log_lik = self.log_lik(Y)
            obj = -log_lik
            rel_obj = abs(obj - prev_obj) / abs(prev_obj)
            if (n_iter > max_iter) or (rel_obj < tol):
                break

        self.history.update(n_iter=n_iter + 1, obj=obj, rel_obj=rel_obj)
        if verbose:
            self.history.print_history()
        self._end_solve()
        self.beta = beta
        self.D = D
        self.phi = phi


class ULMM:
    """Fit univariate linear mixed models
    """

    def __init__(self, fixed_effect_time_order=5):
        self.fixed_effect_time_order = fixed_effect_time_order

        # Attributes that will be instantiated afterwards
        self.beta = None
        self.D = None
        self.phi = None

    def fit(self, Y):
        """Fit univariate linear mixed models

        Parameters
        ----------
        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The simulated longitudinal data. Each element of the dataframe is
            a pandas.Series
        """
        # TODO Van Tuan
        fixed_effect_time_order = self.fixed_effect_time_order
        n_long_features = Y.shape[1]
        q = (fixed_effect_time_order + 1) * n_long_features
        r = 2 * n_long_features  # all r_l=2
        beta = np.zeros(q)
        D = np.ones((r, r))
        phi = np.ones(n_long_features)

        self.beta = beta
        self.D = D
        self.phi = phi