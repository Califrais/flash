from lights.base.base import Learner, block_diag
from lights.init.ulmm import ULMM
import numpy as np
from numpy.linalg import multi_dot


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

    initialize : `bool`, default=True
        If `True`, we initialize the parameters using ULMM model, otherwise we
        use arbitrarily chosen fixed initialization
    """

    def __init__(self, max_iter=100, verbose=True, print_every=10, tol=1e-5,
                 fixed_effect_time_order=5, initialize=True):
        Learner.__init__(self, verbose=verbose, print_every=print_every)
        self.max_iter = max_iter
        self.verbose = verbose
        self.print_every = print_every
        self.tol = tol
        self.fixed_effect_time_order = fixed_effect_time_order
        self.initialize = initialize

        # Attributes that will be instantiated afterwards
        self.fixed_effect_coeffs = None
        self.long_cov = None
        self.phi = None

    def log_lik(self, extracted_features):
        """Computes the log-likelihood of the multivariate linear mixed model

        Parameters
        ----------
        extracted_features : `tuple, tuple`,
            The extracted features from longitudinal data.
            Each tuple is a combination of fixed-effect design features,
            random-effect design features, outcomes, number of the longitudinal
            measurements for all subject or arranged by l-th order.

        Returns
        -------
        output : `float`
            The value of the log-likelihood
        """
        (U_list, V_list, y_list, N), (U_L, V_L, y_L, N_L) = extracted_features
        n_samples, n_long_features = len(U_list), len(U_L)
        D, beta, phi = self.long_cov, self.fixed_effect_coeffs, self.phi

        log_lik = 0
        for i in range(n_samples):
            U_i, V_i, y_i, n_i = U_list[i], V_list[i], y_list[i], sum(N[i])
            inv_Phi_i = [[phi[l, 0]] * N[i][l] for l in range(n_long_features)]
            inv_Sigma_i = np.diag(np.concatenate(inv_Phi_i))
            tmp_1 = multi_dot([V_i, D, V_i.T]) + inv_Sigma_i
            tmp_2 = y_i - U_i.dot(beta)

            op1 = n_i * np.log(2 * np.pi)
            op2 = np.log(np.linalg.det(tmp_1))
            op3 = multi_dot([tmp_2.T, np.linalg.inv(tmp_1), tmp_2])[0][0]

            log_lik -= .5 * (op1 + op2 + op3)

        return log_lik

    def update_theta(self, fixed_effect_coeffs, long_cov, phi):
        """Update class attributes corresponding to MLMM model parameters

        Parameters
        ----------
        fixed_effect_coeffs : `np.ndarray`,
            shape=((fixed_effect_time_order+1)*n_long_features,)
            Fixed effect coefficient vectors

        long_cov : `np.ndarray`, shape=(2*n_long_features, 2*n_long_features)
            Variance-covariance matrix that accounts for dependence between the
            different longitudinal outcome. Here r = 2*n_long_features since
            one choose affine random effects, so all r_l=2

        phi : `np.ndarray`, shape=(n_long_features,)
            Variance vector for the error term of the longitudinal processes
        """
        self.fixed_effect_coeffs = fixed_effect_coeffs
        self.long_cov = long_cov
        self.phi = phi

    def fit(self, extracted_features):
        """Fit the multivariate linear mixed model

        Parameters
        ----------
        extracted_features : `tuple, tuple`,
            The extracted features from longitudinal data.
            Each tuple is a combination of fixed-effect design features,
            random-effect design features, outcomes, number of the longitudinal
            measurements for all subject or arranged by l-th order.
        """
        self._start_solve()
        verbose = self.verbose
        max_iter = self.max_iter
        print_every = self.print_every
        tol = self.tol
        fixed_effect_time_order = self.fixed_effect_time_order

        (U_list, V_list, y_list, N), (U_L, V_L, y_L, N_L) = extracted_features
        n_samples, n_long_features = len(U_list), len(U_L)
        q_l = fixed_effect_time_order + 1
        r_l = 2  # Affine random effects

        if self.initialize:
            # initialize parameters by fitting univariate linear mixed models
            ulmm = ULMM(verbose=verbose,
                        fixed_effect_time_order=fixed_effect_time_order)
            ulmm.fit(extracted_features)
            beta = ulmm.fixed_effect_coeffs
            D = ulmm.long_cov
            phi = ulmm.phi
        else:
            # fixed initialization
            q = q_l * n_long_features
            r = r_l * n_long_features
            beta = np.zeros((q, 1))
            D = np.diag(np.ones(r))
            phi = np.ones((n_long_features, 1))

        self.update_theta(beta, D, phi)
        obj = -self.log_lik(extracted_features)
        # store init values
        self.history.update(n_iter=0, obj=obj, rel_obj=np.inf, phi=phi,
                            fixed_effect_coeffs=beta.ravel(), long_cov=D)
        if verbose:
            self.history.print_history()

        for n_iter in range(1, max_iter + 1):

            # E-Step
            Omega = []
            mu = np.zeros((n_long_features * r_l, n_samples))
            mu_tilde_L = [np.array([])] * n_long_features
            Omega_L = [np.zeros(
                (n_samples * r_l, n_samples * r_l))] * n_long_features

            for i in range(n_samples):
                U_i, V_i, y_i, N_i = U_list[i], V_list[i], y_list[i], N[i]

                # compute Sigma_i
                Phi_i = [[1 / phi[l, 0]] * N_i[l]
                         for l in range(n_long_features)]
                Sigma_i = np.diag(np.concatenate(Phi_i))

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
                        = Omega_i[
                          r_l * l: r_l * (l + 1), r_l * l: r_l * (l + 1)]

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
                N_l, y_l, U_l, V_l = sum(N_L[l]), y_L[l], U_L[l], V_L[l]
                Omega_l = Omega_L[l]
                beta_l = beta[q_l * l: q_l * (l + 1)]
                mu_l = mu_tilde_L[l].reshape(-1, 1)
                tmp = y_l - U_l.dot(beta_l)
                phi[l] = (tmp.T.dot(tmp - 2 * V_l.dot(mu_l)) + np.trace(
                    V_l.T.dot(V_l).dot(Omega_l + mu_l.dot(mu_l.T)))) / N_l

            self.update_theta(beta, D, phi)
            prev_obj = obj
            obj = -self.log_lik(extracted_features)
            rel_obj = abs(obj - prev_obj) / abs(prev_obj)

            if n_iter % print_every == 0:
                self.history.update(n_iter=n_iter, obj=obj, rel_obj=rel_obj,
                                    fixed_effect_coeffs=beta.ravel(), phi=phi,
                                    long_cov=D)
                if verbose:
                    self.history.print_history()
            if (n_iter > max_iter) or (rel_obj < tol):
                break

        self._end_solve()
