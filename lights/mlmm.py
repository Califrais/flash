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

    def __init__(self, max_iter=100, verbose=True, print_every=10, tol=1e-5):
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
    def log_lik(Y, beta, D, phi, Eb):
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

        for i in range(n):
            U_i, Y_i, n_i = MLMM.extract_features(Y.iloc[i])
            V_i = U_i
            diag = []
            for l in range(L):
                diag += [1 / phi[l]] * n_i[l]
            S_i = np.diag(diag)

            M_i = U_i.dot(beta) + V_i.dot(Eb[i])
            c_phi = 0.5 * M_i.transpose().dot(S_i).dot(M_i)
            d_phi = - 0.5 * Y_i.transpose().dot(S_i).dot(Y_i) - 0.5 * np.log(
                ((2 * np.pi) ** sum(n_i))) + 0.5 * np.log(np.linalg.det(S_i))

            f_yi = Y_i.transpose().dot(S_i).dot(M_i) - c_phi + d_phi

            r = Eb[i].shape[0]
            f_bi = -0.5 * r * np.log(2*np.pi) -0.5 * np.log(np.linalg.det(D)) \
                   -0.5 * Eb[i].transpose().dot(np.linalg.inv(D)).dot(Eb[i])

            log_lik += f_yi + f_bi

        return log_lik

    @staticmethod
    def extract_features(D_i, l = None):
        """Extract the longitudinal data  of subject i
        into features of the multivariate linear mixed model

        Parameters
        ----------
        D_i : `list of pandas.Series`, shape=(n_long_features)
            The simulated longitudinal data of i-th subject
        l   : `int`
            The index of l-th outcome

        Returns
        -------
        U_i : `list of np.array`
            The fixed-effect features for i-th subject
        Y_i : `list of np.array`
            The outcome i-th subject
        n_i : `list`
            The number samples for each feature of i-th subject
        """
        def extract_specified_features(D_il):
            """Extract the longitudinal data of subject i-th outcome l-th
            into features of the multivariate linear mixed model

            Parameters
            ----------
            D_il : `pandas.Series`
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
            times_il = D_i[l].index.values
            Y_il = D_i[l].values
            n_il = len(times_il)
            U_il = np.c_[np.ones(n_il), times_il]
            return U_il, Y_il, n_il

        if l is not None:
            U_il, Y_il, n_il = extract_specified_features(D_i[l])
            return U_il, Y_il, n_il

        else:
            L = len(D_i)
            for l in range(L):
                U_il, Y_il, n_il = extract_specified_features(D_i[l])
                if l == 0:
                    U_i = U_il
                    Y_i = Y_il
                    n_i = [n_il]
                else:
                    U_i = block_diag(U_i, U_il)
                    Y_i = np.concatenate((Y_i,Y_il))
                    n_i.append(n_il)
            return U_i, Y_i, n_i

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

        log_lik = 1.
        obj = -log_lik
        rel_obj = 1.
        self.history.update(n_iter=0, obj=obj, rel_obj=rel_obj)
        if verbose:
            self.history.print_history()

        n, L = Y.shape
        n_iter = 0
        for n_iter in range(max_iter):
            if n_iter % print_every == 0:
                self.history.update(n_iter=n_iter, obj=obj,
                                    rel_obj=rel_obj)
                if verbose:
                    self.history.print_history()
            # E-Step
            Eb, EbbT = [], []
            for i in range(n):
                U_i, Y_i, n_i = self.extract_features(Y.iloc[i])
                V_i = U_i
                diag = []
                for l in range(L):
                    n_il = n_i[l]
                    diag += [1 / phi[l]] * n_il
                Si = np.diag(diag)

                # MVN covariance matrix for [bi | yi]
                Dinv = np.linalg.inv(D)
                Ai = np.linalg.inv(V_i.transpose().dot(Si).dot(V_i) + Dinv)

                # MVN mean vector for [bi | yi]
                Eb.append(Ai.dot(V_i.transpose()).dot(Si).dot(Y_i - U_i.dot(beta)))

                # E[bibi^T]
                EbbT.append(Ai + Eb[i].dot(Eb[i].transpose()))

            # M-Step
            # Update beta
            op1, op2 = 0, 0
            for i in range(n):
                U_i, Y_i, n_i = self.extract_features(Y.iloc[i])
                V_i = U_i
                op1 += U_i.transpose().dot(U_i)
                op2 += U_i.transpose().dot(Y_i) - U_i.transpose().dot(V_i).dot(Eb[i])

            beta = np.linalg.inv(op1).dot(op2)

            # Update D
            op3 = 0
            for i in range(n):
                op3 += EbbT[i]

            D = op3 / n

            print("D", D)

            # Update phi
            for l in range(L):
                n_l = 0
                op4 = 0
                for i in range(n):
                    U_il, Y_il, n_il = self.extract_features(Y.iloc[i], l)
                    V_il = U_il
                    beta_l = beta[2*l : 2*(l+1)]
                    Eb_l = Eb[i][2*l : 2*(l+1)]
                    EbbT_l = EbbT[i][2*l : 2*(l+1), 2*l : 2*(l+1)]
                    # used for update phi
                    residFixed = Y_il - U_il.dot(beta_l)
                    op4 += np.dot(residFixed.transpose(), (residFixed - 2 * V_il.dot(Eb_l))) + np.trace(V_il.transpose().dot(V_il).dot(EbbT_l))
                    n_l += n_il

                phi[l] = op4 / n_l


            prev_obj = obj
            log_lik = self.log_lik(Y, beta, D, phi, Eb)
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

        fixed_effect_time_order = self.fixed_effect_time_order
        n, n_long_features = Y.shape
        q = 2 * n_long_features # so all q_l=2
        # TODO Update to be general later
        # q = (fixed_effect_time_order + 1) * n_long_features
        r = 2 * n_long_features  # all r_l=2
        beta = np.zeros(q)
        D = np.zeros((r, r))
        phi = np.ones(n_long_features)
        n, L = Y.shape
        for l in range(n_long_features):
            data = pd.DataFrame(columns=['U', 'V', 'Y', 'S'])
            s = 0
            for i in range(n):
                times_il = Y.iloc[i][l].index.values
                Y_il = Y.iloc[i][l].values
                n_il = len(times_il)
                data = data.append(pd.DataFrame(data = {'U': times_il, 'V': times_il, 'Y': Y_il, 'S': [s]*n_il}))
                s += 1
            md = smf.mixedlm("Y ~ U", data, groups=data["S"], re_formula="~V")
            mdf = md.fit()
            beta[2*l] = mdf.params["Intercept"]
            beta[2*l + 1] = mdf.params["U"]
            D[2*l:2*(l+1), 2*l:2*(l+1)] = np.array([[mdf.params["Group Var"], mdf.params["Group x V Cov"]], [mdf.params["Group x V Cov"], mdf.params["V Var"]]])
            phi[l] = mdf.resid.values.var()

        print(beta)
        print(phi)
        print(D)
        self.beta = beta
        self.D = D
        self.phi = phi