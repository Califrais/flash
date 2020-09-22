# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

from lights.base import Learner, block_diag
import numpy as np
import statsmodels.formula.api as smf
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

    def log_lik(self, extracted_features):
        """Computes the log-likelihood of the multivariate linear mixed model

        Parameters
        ----------
        extracted_features : `tuple, tuple`,
            The extracted features from simulated longitudinal data.
            Each tuple is a combination of fixed-effect design features,
            random-effect design features, outcomes, number of the longitudinal
            measurements for all subject or arranged by l-th order.

        Returns
        -------
        output : `float`
            The value of the log-likelihood
        """
        (U_list, V_list, y_list, N), (U_L, V_L, y_L, N_L) = extracted_features
        n_samples = len(U_list)
        n_long_features = len(U_L)
        log_lik = 0

        for i in range(n_samples):
            U_i = U_list[i]
            V_i = V_list[i]
            n_i = sum(N[i])
            y_i = y_list[i]
            diag  = []
            for l in range(n_long_features):
                diag += [1 / self.phi[l, 0]] * N[i][l]
            Sigma_i = np.diag(diag)

            op1 = -0.5*n_i*np.log(2*np.pi)
            op2 = -0.5*np.log(np.linalg.det(np.linalg.multi_dot([V_i, self.D, V_i.T])+ Sigma_i))
            op3 = -0.5*np.linalg.multi_dot([(y_i - np.dot(U_i, self.beta)).T,
                np.linalg.inv(np.linalg.multi_dot([V_i, self.D, V_i.T]) + Sigma_i),
                                           (y_i - np.dot(U_i, self.beta))])

            log_lik += op1 + op2 + op3

        return log_lik

    def fit(self, extracted_features):
        """Fit the multivariate linear mixed model

        Parameters
        ----------
        extracted_features : `tuple, tuple`,
            The extracted features from simulated longitudinal data.
            Each tuple is a combination of fixed-effect design features,
            random-effect design features, outcomes, number of the longitudinal
            measurements for all subject or arranged by l-th order.
        """
        verbose = self.verbose
        max_iter = self.max_iter
        print_every = self.print_every
        tol = self.tol
        fixed_effect_time_order = self.fixed_effect_time_order
        (U_list, V_list, y_list, N), (U_L, V_L, y_L, N_L) = extracted_features
        n_samples = len(U_list)
        n_long_features = len(U_L)
        r_l = 2  # linear time-varying features, so all r_l=2

        self._start_solve()
        # We initialize parameters by fitting univariate linear mixed models
        ulmm = ULMM(fixed_effect_time_order)
        ulmm.fit(extracted_features)
        beta = ulmm.beta
        D = ulmm.D
        phi = ulmm.phi

        log_lik = 1.
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
            mu, Omega = [], []
            mu = np.zeros((n_long_features * r_l, n_samples))
            mu_tilde_L = [np.array([])] * n_long_features
            Omega_L = [np.zeros((n_samples * r_l, n_samples * r_l))] * n_long_features

            for i in range(n_samples):
                U_i, V_i, y_i, N_i = U_list[i], V_list[i], y_list[i], N[i]

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
                mu[:, i] = mu_i.flatten()

                for l in range(n_long_features):
                    mu_tilde_L[l] = np.append(
                        mu_tilde_L[l],
                        mu[r_l * l: r_l * (l + 1), i]
                    )
                    Omega_L[l][i * r_l: (i + 1) * r_l, i * r_l: (i + 1) * r_l] \
                        = Omega_i[r_l * l: r_l * (l + 1),
                          r_l * l: r_l * (l + 1)]

            mu_flat = mu.T.flatten().reshape(-1, 1)

            # M-Step
            U = np.concatenate(U_list)
            V = block_diag(V_list)
            y = np.concatenate(y_list)

            # Update beta
            U_T = U.transpose()
            beta = np.linalg.inv(U_T.dot(U)).dot(U_T.dot(y - V.dot(mu_flat)))

            # Update D
            D = (np.array(Omega).sum(axis=0) + mu.dot(mu.T)) / n_samples

            # Update phi
            for l in range(n_long_features):
                N_l = sum(N_L[l])
                y_l = y_L[l]
                U_l = U_L[l]
                V_l = V_L[l]
                Omega_l = Omega_L[l]
                beta_l = beta[2 * l: 2 * (l + 1)]
                mu_l = mu_tilde_L[l].reshape(-1, 1)

                tmp = y_l - U_l.dot(beta_l)
                phi[l] = (tmp.T.dot(tmp - 2 * V_l.dot(mu_l)) + np.trace(
                    V_l.T.dot(V_l).dot(Omega_l + mu_l.dot(mu_l.T)))) / N_l

            prev_obj = obj
            self.beta = beta
            self.D = D
            self.phi = phi

            log_lik = self.log_lik(extracted_features)
            print("Likelihood", log_lik)

            obj = -log_lik
            rel_obj = abs(obj - prev_obj) / abs(prev_obj)
            if (n_iter > max_iter) or (rel_obj < tol):
                break

        self.history.update(n_iter=n_iter + 1, obj=obj, rel_obj=rel_obj)
        if verbose:
            self.history.print_history()
        self._end_solve()


class ULMM(Learner):
    """Fit univariate linear mixed models
    """
    def __init__(self, fixed_effect_time_order=5):
        self.fixed_effect_time_order = fixed_effect_time_order

        # Attributes that will be instantiated afterwards
        self.beta = None
        self.D = None
        self.phi = None

    def fit(self, extracted_features):
        """Fit univariate linear mixed models

        Parameters
        ----------
        extracted_features : `tuple, tuple`,
            The extracted features from simulated longitudinal data.
            Each tuple is a combination of fixed-effect design features,
            random-effect design features, outcomes, number of the longitudinal
            measurements for all subject or arranged by l-th order.
        """
        fixed_effect_time_order = self.fixed_effect_time_order
        q_l = fixed_effect_time_order + 1
        r_l = 2  # linear time-varying features, so all r_l=2
        (U_list, V_list, y_list, N), (U_L, V_L, y_L, N_L) = extracted_features
        n_samples = len(U_list)
        n_long_features = len(U_L)
        q = q_l * n_long_features
        r = r_l * n_long_features

        beta = np.zeros(q)
        D = np.zeros((r, r))
        phi = np.ones(n_long_features)

        for l in range(n_long_features):
            U = U_L[l][:, 1:]
            V = U_L[l][:, 1:r_l]
            Y = y_L[l]
            S = np.array([])
            for i in range(n_samples):
                S = np.append(S, i*np.ones(N_L[l][i]))

            fixed_effect_columns = []
            other_columns = ['V', 'Y', 'S']
            for j in range(fixed_effect_time_order):
                fixed_effect_columns.append('U' + str(j + 1))
            data = pd.DataFrame(data = np.hstack((U,V,Y,S.reshape(-1, 1))),
                                columns=fixed_effect_columns + other_columns)

            md = smf.mixedlm("Y ~ " + ' + '.join(fixed_effect_columns), data,
                             groups=data["S"], re_formula="~V")
            mdf = md.fit()
            beta[q_l * l] = mdf.params["Intercept"]
            beta[q_l * l + 1: q_l * (l + 1)] = [mdf.params[features]
                                            for features in fixed_effect_columns]

            D[r_l * l: r_l * (l + 1), r_l * l: r_l * (l + 1)] = np.array(
                [[mdf.params["Group Var"], mdf.params["Group x V Cov"]],
                 [mdf.params["Group x V Cov"], mdf.params["V Var"]]])
            phi[l] = mdf.resid.values.var()

        self.beta = beta.reshape(-1, 1)
        self.D = D
        self.phi = phi.reshape(-1, 1)
