# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

from lights.base import Learner
import numpy as np
import statsmodels.formula.api as smf
from scipy.linalg import block_diag
import pandas as pd


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

    def __init__(self, max_iter=100, verbose=True, print_every=10, tol=1e-5, fixed_effect_time_order=5):
        Learner.__init__(self, verbose=verbose, print_every=print_every)
        self.max_iter = max_iter
        self.verbose = verbose
        self.print_every = print_every
        self.tol = tol
        self.fixed_effect_time_order = fixed_effect_time_order

        # Attributes that will be instantiated afterwards
        self.beta = None
        self.D = None
        self.phi = None

    def log_lik(self, Y, Eb):
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

        n, L = Y.shape
        r = Eb[0].shape[0]

        (U, V, y, N), (U_L, V_L, y_L, N_L) = self.extract_features(Y, self.fixed_effect_time_order)
        Eb_c = np.concatenate(Eb).reshape(-1, 1)
        M = np.dot(np.concatenate(U), self.beta) + np.dot(V, Eb_c)
        diag = []
        for i in range(n):
            for l in range(L):
                N_il = N[i][l]
                diag += [1 / self.phi[l, 0]] * N_il
        S = np.diag(diag)
        log_det_S = np.log(diag).sum()

        y_c = np.concatenate(y).reshape(-1, 1)
        Eb_c = np.concatenate(Eb).reshape(-1, 1)
        log_lik = np.linalg.multi_dot([y_c.transpose(), S, M]) - 0.5 * np.linalg.multi_dot(
            [M.transpose(), S, M]) - 0.5 * np.linalg.multi_dot([y_c.transpose(), S, y_c]) \
                  - 0.5 * np.array(N).sum() * np.log(2 * np.pi) + 0.5 * log_det_S - 0.5 * n * r * np.log(
            2 * np.pi) - 0.5 * n * np.log(np.linalg.det(self.D)) \
                  - 0.5 * Eb_c.transpose().dot(np.kron(np.eye(n), np.linalg.inv(self.D))).dot(Eb_c)

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
        fixed_effect_time_order = self.fixed_effect_time_order
        random_effect_time_order = fixed_effect_time_order
        q_l = fixed_effect_time_order + 1
        r_l = random_effect_time_order + 1
        self._start_solve()

        # We initialize parameters by fitting univariate linear mixed models
        ulmm = ULMM(fixed_effect_time_order)
        ulmm.fit(Y)
        beta = ulmm.beta.reshape(-1, 1)
        D = ulmm.D
        phi = ulmm.phi.reshape(-1, 1)

        log_lik = 1.
        obj = -log_lik
        rel_obj = 1.
        self.history.update(n_iter=0, obj=obj, rel_obj=rel_obj)
        if verbose:
            self.history.print_history()

        # feature extraction
        (U, V, y, N), (U_L, V_L, y_L, N_L) = self.extract_features(Y, fixed_effect_time_order)
        n, L = Y.shape
        n_iter = 0
        for n_iter in range(max_iter):
            if n_iter % print_every == 0:
                self.history.update(n_iter=n_iter, obj=obj,
                                    rel_obj=rel_obj)
                if verbose:
                    self.history.print_history()

            # E-Step
            Eb, A = [], []
            Eb_L, A_L = [], []
            for i in range(n):
                U_i, y_i, N_i = U[i], y[i].reshape(-1, 1), N[i]
                V_i = U_i
                diag = []
                for l in range(L):
                    N_il = N_i[l]
                    diag += [1 / phi[l, 0]] * N_il

                Si = np.diag(diag)

                # MVN covariance matrix for [bi | yi]
                Dinv = np.linalg.inv(D)
                Ai = np.linalg.inv(V_i.transpose().dot(Si).dot(V_i) + Dinv)

                # MVN mean vector for [bi | yi]
                Eb.append(Ai.dot(V_i.transpose()).dot(Si).dot(y_i - U_i.dot(beta)))

                A.append(Ai)

                for l in range(L):
                    if i == 0:
                        Eb_L.append(Eb[i][r_l * l: r_l * (l + 1), 0])
                        A_L.append(Ai[r_l * l : r_l * (l + 1), r_l * l : r_l * (l + 1)])
                    else:
                        Eb_L[l] = np.concatenate((Eb_L[l].transpose(), Eb[i][r_l * l : r_l * (l + 1), 0].transpose())).transpose()
                        A_L[l] = block_diag(A_L[l], Ai[r_l * l : r_l * (l + 1), r_l * l : r_l * (l + 1)])


            # M-Step
            # Update beta
            U_beta = np.concatenate(U)
            V_beta = V
            y_beta = np.concatenate(y).reshape(-1, 1)
            mu_beta = np.concatenate(Eb).reshape(-1, 1)

            op1 = np.dot(U_beta.transpose(), U_beta)
            op2 = np.dot(U_beta.transpose(), (y_beta - np.dot(V_beta, mu_beta)))
            beta = np.linalg.inv(op1).dot(op2)

            # Update D
            mu_D = np.array(Eb).reshape(n, -1).transpose()
            D = (1/n)*(np.array(A).sum(axis=0) + np.dot(mu_D, mu_D.transpose()))

            # Update phi
            mu_L = Eb_L
            for l in range(L):
                N_l = np.array(N_L[l]).sum()
                y_l = y_L[l].reshape(-1, 1)
                U_l = U_L[l]
                V_l = V_L[l]
                beta_l = beta[2 * l: 2 * (l + 1)]
                A_l = A_L[l]
                mu_l = mu_L[l].reshape(-1, 1)

                residFixed = y_l - U_l.dot(beta_l)
                phi[l] = (1/N_l)*(np.dot(residFixed.transpose(), (residFixed - 2 * V_l.dot(mu_l)))\
                         + np.trace(V_l.transpose().dot(V_l).dot(A_l + np.dot(mu_l, mu_l.transpose()))))

            prev_obj = obj
            self.beta = beta
            self.D = D
            self.phi = phi
            log_lik = self.log_lik(Y, Eb)
            print("Likelihood", log_lik)
            obj = -log_lik
            rel_obj = abs(obj - prev_obj) / abs(prev_obj)
            if (n_iter > max_iter) or (rel_obj < tol):
                break

        self.history.update(n_iter=n_iter + 1, obj=obj, rel_obj=rel_obj)
        if verbose:
            self.history.print_history()
        self._end_solve()


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

        fixed_effect_time_order = self.fixed_effect_time_order
        random_effect_time_order = fixed_effect_time_order
        q_l = fixed_effect_time_order + 1
        r_l = random_effect_time_order + 1
        n, n_long_features = Y.shape
        q = q_l * n_long_features
        r = r_l * n_long_features
        beta = np.zeros(q)
        D = np.zeros((r, r))
        phi = np.ones(n_long_features)
        n, L = Y.shape
        for l in range(n_long_features):
            data = pd.DataFrame(columns=['U', 'V', 'Y', 'S'])
            s = 0
            for i in range(n):
                times_il = Y.iloc[i][l].index.values
                U = times_il
                for t in range(fixed_effect_time_order-1):
                    U = np.c_[U, times_il ** (t + 2)]
                V = U
                Y_il = Y.iloc[i][l].values
                n_il = len(times_il)
                data = data.append(pd.DataFrame(data = {'U': U, 'V': V, 'Y': Y_il, 'S': [s]*n_il}))
                s += 1
            md = smf.mixedlm("Y ~ U", data, groups=data["S"], re_formula="~V")
            mdf = md.fit()
            beta[q_l * l] = mdf.params["Intercept"]
            beta[q_l * l + 1 : q_l * (l + 1)] = mdf.params["U"]
            D[r_l * l : r_l * (l + 1), r_l * l : r_l * (l + 1)] = np.array([[mdf.params["Group Var"], mdf.params["Group x V Cov"]], [mdf.params["Group x V Cov"], mdf.params["V Var"]]])
            phi[l] = mdf.resid.values.var()

        self.beta = beta
        self.D = D
        self.phi = phi