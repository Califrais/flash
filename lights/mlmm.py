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
    max_iter : `int`, default=100
        Maximum number of iterations of the solver

    verbose : `bool`, default=True
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default=10
        Print history information when ``n_iter`` (iteration number) is
        a multiple of ``print_every``

    tol : `float`, default=1e-5
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

    fixed_effect_time_order : `int`, default=5
        Order of the higher time monomial considered for the representations of
        the time-varying features corresponding to the fixed effect. The
        dimension of the corresponding design matrix is then equal to
        fixed_effect_time_order + 1
    """

    def __init__(self, max_iter=100, verbose=True, print_every=10, tol=1e-5,
                 fixed_effect_time_order=5):
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

        (U, V, y, N), (U_L, V_L, y_L, N_L) = self.extract_features(Y,
                                                                   self.fixed_effect_time_order)
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
        log_lik = np.linalg.multi_dot(
            [y_c.transpose(), S, M]) - 0.5 * np.linalg.multi_dot(
            [M.transpose(), S, M]) - 0.5 * np.linalg.multi_dot(
            [y_c.transpose(), S, y_c]) \
                  - 0.5 * np.array(N).sum() * np.log(
            2 * np.pi) + 0.5 * log_det_S - 0.5 * n * r * np.log(
            2 * np.pi) - 0.5 * n * np.log(np.linalg.det(self.D)) \
                  - 0.5 * Eb_c.transpose().dot(
            np.kron(np.eye(n), np.linalg.inv(self.D))).dot(Eb_c)

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
        q_l = fixed_effect_time_order + 1
        r_l = 2  # linear time-varying features, so all r_l=2
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

        # features extraction
        (U, V, y, N), (U_L, V_L, y_L, N_L) = self.extract_features(Y,
                                                                   fixed_effect_time_order)

        n_samples, n_long_features = Y.shape
        n_iter = 0
        for n_iter in range(max_iter):
            if n_iter % print_every == 0:
                self.history.update(n_iter=n_iter, obj=obj,
                                    rel_obj=rel_obj)
                if verbose:
                    self.history.print_history()

            # E-Step
            mu_tilde, Omega, mu_tilde_L, Omega_L = [], []\
                , [np.array([])] * n_long_features\
                , [np.zeros((n_samples * r_l, n_samples * r_l))] * n_long_features
            mu= np.array([])
            pointer = 0
            for i in range(n_samples):
                N_i = N[i]
                U_i = U[pointer : pointer + sum(N_i),:]
                y_i= y[pointer : pointer + sum(N_i)]
                V_i = V[pointer : pointer + sum(N_i),
                      i*n_long_features*r_l : (i+1)*n_long_features*r_l]
                pointer += sum(N_i)

                # compute Sigma_i
                Phi_i = []
                for l in range(n_long_features):
                    n_il = N_i[l]
                    Phi_i += [1 / phi[l, 0]] * n_il
                Sigma_i = np.diag(Phi_i)

                # compute Omega_i
                D_inv = np.linalg.inv(D)
                Omega_i = np.linalg.inv(
                    V_i.transpose().dot(Sigma_i).dot(V_i) + D_inv)
                Omega.append(Omega_i)

                # compute mu_i
                mu_i = Omega_i.dot(V_i.transpose()).dot(Sigma_i).dot(
                    y_i - U_i.dot(beta))
                mu_tilde.append(mu_i)
                mu = np.append(mu, mu_i)

                for l in range(n_long_features):
                    mu_tilde_L[l] = np.append(
                        mu_tilde_L[l],
                        mu_tilde[i][r_l * l: r_l * (l + 1), 0]
                    )

                    Omega_L[l][i * r_l : (i + 1) * r_l, i * r_l : (i + 1) * r_l]\
                        = Omega_i[r_l * l : r_l * (l + 1), r_l * l : r_l * (l + 1)]

            mu = mu.reshape(-1, 1)
            mu_tilde = np.array(mu_tilde).reshape(n_samples, -1).transpose()

            # M-Step
            # Update beta
            beta = np.dot(np.linalg.inv(np.dot(U.transpose(), U)),
                np.dot(U.transpose(), (y - np.dot(V, mu))))

            # Update D
            D = (1 / n_samples) * (np.array(Omega).sum(axis=0)
                                   + np.dot(mu_tilde, mu_tilde.transpose()))

            # Update phi
            for l in range(n_long_features):
                N_l = sum(N_L[l])
                y_l = y_L[l]
                U_l = U_L[l]
                V_l = V_L[l]
                Omega_l = Omega_L[l]
                beta_l = beta[2 * l: 2 * (l + 1)]
                mu_l = mu_tilde_L[l].reshape(-1, 1)

                residFixed = y_l - U_l.dot(beta_l)
                phi[l] = (1 / N_l) * (np.dot(residFixed.transpose(),
                                             (residFixed - 2 * V_l.dot(mu_l))) \
                                      + np.trace(V_l.transpose().dot(V_l).dot(
                            Omega_l + np.dot(mu_l, mu_l.transpose()))))

            prev_obj = obj
            self.beta = beta
            self.D = D
            self.phi = phi
            # TODO: Update likelihood function
            # log_lik = self.log_lik(Y, mu_tilde)
            # print("Likelihood", log_lik)
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
                for t in range(fixed_effect_time_order - 1):
                    U = np.c_[U, times_il ** (t + 2)]
                V = U
                Y_il = Y.iloc[i][l].values
                n_il = len(times_il)
                data = data.append(pd.DataFrame(
                    data={'U': U, 'V': V, 'Y': Y_il, 'S': [s] * n_il}))
                s += 1
            md = smf.mixedlm("Y ~ U", data, groups=data["S"], re_formula="~V")
            mdf = md.fit()
            beta[q_l * l] = mdf.params["Intercept"]
            beta[q_l * l + 1: q_l * (l + 1)] = mdf.params["U"]
            D[r_l * l: r_l * (l + 1), r_l * l: r_l * (l + 1)] = np.array(
                [[mdf.params["Group Var"], mdf.params["Group x V Cov"]],
                 [mdf.params["Group x V Cov"], mdf.params["V Var"]]])
            phi[l] = mdf.resid.values.var()

        self.beta = beta
        self.D = D
        self.phi = phi
