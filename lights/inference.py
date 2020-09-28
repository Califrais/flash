# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

from lights.base import Learner, extract_features
from lights.mlmm import MLMM
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from lifelines.utils import concordance_index as c_index_score
from sklearn.model_selection import KFold


class QNMCEM(Learner):
    """QNMCEM Algorithm for the lights model inference

    Parameters
    ----------
    fit_intercept : `bool`, default=True
        If `True`, include an intercept in the model for the time independant
        features

    l_elastic_net : `float`, default=0
        Level of ElasticNet penalization

    eta: `float`, default=0.1
        The ElasticNet mixing parameter, with 0 <= eta <= 1.
        For eta = 0 this is ridge (L2) regularization
        For eta = 1 this is lasso (L1) regularization
        For 0 < eta < 1, the regularization is a linear combination
        of L1 and L2

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

    warm_start : `bool`, default=False
        If true, learning will start from the last reached solution

    fixed_effect_time_order : `int`, default=5
        Order of the higher time monomial considered for the representations of
        the time-varying features corresponding to the fixed effect. The
        dimension of the corresponding design matrix is then equal to
        fixed_effect_time_order + 1

    initialize : `bool`, default=True
        If `True`, we initialize the parameters using MLMM model, otherwise we
        use arbitrarily chosen fixed initialization
    """
    def __init__(self, fit_intercept=False, l_elastic_net=0.,
                 eta=.1, max_iter=100, verbose=True, print_every=10, tol=1e-5,
                 warm_start=False, fixed_effect_time_order=5, initialize=True):
        Learner.__init__(self, verbose=verbose, print_every=print_every)
        self.l_elastic_net = l_elastic_net
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept
        self.fixed_effect_time_order = fixed_effect_time_order
        self.initialize = initialize

        # Attributes that will be instantiated afterwards
        self.n_time_indep_features = None
        self.n_samples = None
        self.beta_0 = None
        self.beta_1 = None
        self.D = None
        self.phi = None
        self.xi = None
        self.gamma_0 = None
        self.gamma_1 = None
        self.avg_scores = None
        self.scores = None
        self.l_elastic_net_best = None
        self.l_elastic_net_chosen = None
        self.grid_elastic_net = None
        self.adaptative_grid_el = None
        self.grid_size = None

    @staticmethod
    def logistic_grad(z):
        """Overflow proof computation of 1 / (1 + exp(-z)))
        """
        idx_pos = np.where(z >= 0.)
        idx_neg = np.where(z < 0.)
        res = np.empty(z.shape)
        res[idx_pos] = 1. / (1. + np.exp(-z[idx_pos]))
        res[idx_neg] = 1 - 1. / (1. + np.exp(z[idx_neg]))
        return res

    @staticmethod
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

    @staticmethod
    def get_vect_from_ext(v_ext):
        """Obtain the signed coefficient vector from its extension on positive
        and negative parts
        """
        dim = len(v_ext)
        if dim % 2 != 0:
            raise ValueError("``v_ext`` dimension cannot be odd, got %s" % dim)
        v = v_ext[:dim] - v_ext[dim:]
        return v

    def _get_xi_from_xi_ext(self, xi_ext):
        """Get the time-independent coefficient vector from its extension on
        positive and negative parts

        Parameters
        ----------
        xi_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
            The time-independent coefficient vector decomposed on positive and
            negative parts

        Returns
        -------

        xi : `np.ndarray`, shape=(n_time_indep_features,)
            The time-independent coefficient vector
        """
        n_time_indep_features = self.n_time_indep_features
        if self.fit_intercept:
            xi = xi_ext[:n_time_indep_features + 1] - \
                 xi_ext[n_time_indep_features + 1:]
            xi_0 = xi[0]
            xi = xi[1:]
        else:
            xi_0 = 0
            xi = xi_ext[:n_time_indep_features] - xi_ext[n_time_indep_features:]
        return xi_0, xi

    def _clean_xi_ext(self, xi_ext):
        """Removes potential intercept coefficients in the time-independent
        coefficient vector decomposed on positive and negative parts
        """
        if self.fit_intercept:
            n_time_indep_features = self.n_time_indep_features
            xi_ext = np.delete(xi_ext, [0, n_time_indep_features + 1])
        return xi_ext

    def _elastic_net_pen(self, xi_ext):
        """Computes the elasticNet penalization of the global objective to be
        minimized by the QNMCEM algorithm

        Parameters
        ----------
        xi_ext: `np.ndarray`, shape=(2*n_time_indep_features,)
            The time-independent coefficient vector decomposed on positive and
            negative parts

        Returns
        -------
        output : `float`
            The value of the elasticNet penalization part of the global
            objective
        """
        l_elastic_net = self.l_elastic_net
        eta = self.eta
        xi = self._get_xi_from_xi_ext(xi_ext)[1]
        xi_ext = self._clean_xi_ext(xi_ext)
        return l_elastic_net * ((1. - eta) * xi_ext.sum()
                                + 0.5 * eta * np.linalg.norm(xi) ** 2)

    def _grad_elastic_net_pen(self, xi):
        """Computes the gradient of the elasticNet penalization of the global
        objective to be minimized by the QNMCEM algorithm

        Parameters
        ----------
        xi : `np.ndarray`, shape=(n_time_indep_features,)
            The time-independent coefficient vector

        Returns
        -------
        output : `float`
            The gradient of the elasticNet penalization part of the global
            objective
        """
        l_elastic_net = self.l_elastic_net
        eta = self.eta
        n_time_indep_features = self.n_time_indep_features
        grad = np.zeros(2 * n_time_indep_features)
        # Gradient of lasso penalization
        grad += l_elastic_net * (1 - eta)
        # Gradient of ridge penalization
        grad_pos = (l_elastic_net * eta)
        grad[:n_time_indep_features] += grad_pos * xi
        grad[n_time_indep_features:] -= grad_pos * xi
        return grad

    def _log_lik(self, X, Y, T, delta):
        """Computes the likelihood of the lights model

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The simulated longitudinal data. Each element of the dataframe is
            a pandas.Series

        T : `np.ndarray`, shape=(n_samples,)
            Times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator

        Returns
        -------
        output : `float`
            The log-likelihood computed on the given data
        """
        prb = 1
        # TODO
        return np.mean(np.log(prb))

    def _func_obj(self, X, Y, T, delta, xi_ext):
        """The global objective to be minimized by the QNMCEM algorithm
        (including penalization)

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The simulated longitudinal data. Each element of the dataframe is
            a pandas.Series

        T : `np.ndarray`, shape=(n_samples,)
            Times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator

        xi_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
            The time-independent coefficient vector decomposed on positive and
            negative parts

        Returns
        -------
        output : `float`
            The value of the global objective to be minimized
        """
        log_lik = self._log_lik(X, Y, T, delta)
        pen = self._elastic_net_pen(xi_ext)
        return -log_lik + pen

    def _P_func(self, X, pi_est, xi_ext):
        """Computes the sub objective function denoted P in the lights paper,
        to be minimized at each QNMCEM iteration using fmin_l_bfgs_b.

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        pi_est : `np.ndarray`, shape=(n_samples,)
            The estimated posterior probability of the latent class membership
            obtained by the E-step

        xi_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
            The time-independent coefficient vector decomposed on positive and
            negative parts

        Returns
        -------
        output : `float`
            The value of the P sub objective to be minimized at each QNMCEM step
        """
        xi_0, xi = self._get_xi_from_xi_ext(xi_ext)
        pen = self._elastic_net_pen(xi_ext)
        u = xi_0 + X.dot(xi)
        sub_obj = (pi_est * u + self.logistic_loss(u)).mean()
        return sub_obj + pen

    def _grad_P(self, X, pi_est, xi_ext):
        """Computes the gradient of the sub objective P

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        pi_est : `np.ndarray`, shape=(n_samples,)
            The estimated posterior probability of the latent class membership
            obtained by the E-step

        xi_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
            The time-independent coefficient vector decomposed on positive and
            negative parts

        Returns
        -------
        output : `float`
            The value of the P sub objective gradient
        """
        n_time_indep_features = self.n_time_indep_features
        n_samples = self.n_samples
        xi_0, xi = self._get_xi_from_xi_ext(xi_ext)
        grad_pen = self._grad_elastic_net_pen(xi)
        u = xi_0 + X.dot(xi)
        if self.fit_intercept:
            X = np.concatenate((np.ones(n_samples).reshape(1, n_samples).T, X),
                               axis=1)
            grad_pen = np.concatenate([[0], grad_pen[:n_time_indep_features],
                                       [0], grad_pen[n_time_indep_features:]])
        grad = (X * (pi_est - self.logistic_grad(-u)).reshape(
            n_samples, 1)).mean(axis=0)
        grad_sub_obj = np.concatenate([grad, -grad])
        return grad_sub_obj + grad_pen

    def get_proba(self, X, xi_ext):
        """Probability estimates for being on the high-risk group given
        time-independent features

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        xi_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
            The time-independent coefficient vector decomposed on positive and
            negative parts

        Returns
        -------
        output : `np.ndarray`, shape=(n_samples,)
            Returns the probability of the sample for being on the high-risk
            group given time-independent features
        """
        xi_0, xi = self._get_xi_from_xi_ext(xi_ext)
        u = xi_0 + X.dot(xi)
        return QNMCEM.logistic_grad(u)

    def get_post_proba(self, pi_xi, Lambda_1):
        """Posterior probability estimates for being on the high-risk group
        given all observed data

        Parameters
        ----------
        pi_xi : `np.ndarray`, shape=(n_samples,)
            Comes from get_proba function

        Lambda_1 : `np.ndarray`, shape=(n_samples, 2)
            blabla

        Returns
        -------
        output : `np.ndarray`, shape=(n_samples,)
            Returns the posterior probability of the sample for being on the
            high-risk group given all observed data
        """
        # TODO
        return 1

    def predict_marker(self, X, Y):
        """Marker rule of the lights model for being on the high-risk group

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The simulated longitudinal data. Each element of the dataframe is
            a pandas.Series

        Returns
        -------
        output : `np.ndarray`, shape=(n_samples,)
            Returns the marker rule of the sample for being on the high-risk
            group
        """
        marker = None
        # TODO (only if self.fitted = True, else raise error)
        return marker

    def f_data_g_latent(self, Y, T, delta, S):
        """Computes f(Y, T, delta| S, G, theta)

        Parameters
        ----------
        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The simulated longitudinal data. Each element of the dataframe is
            a pandas.Series

        T : `np.ndarray`, shape=(n_samples,)
            The simulated censored times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            The simulated censoring indicator

        S: `np.ndarray`, , shape=(2*N, r)
            Set of constructed samples

        Returns
        -------
        f : `np.ndarray`, shape=(n_samples, 2)
            The value of the f(Y, T, delta| S, G, theta)
        """
        # TODO : return list for version G=0 and G=1 ; and fill a docstring
        N = S.shape[0]
        return [[0] * N, [1] * N]

    def construct_MC_samples(self, N):
        """Constructs the set of samples used for Monte Carlo approximation

        Parameters
        ----------
        N : `int`
            Number of constructed samples

        Returns
        -------
        S : `np.ndarray`, , shape=(2*N, r)
            Set of constructed samples
        """
        D = self.D
        C = np.linalg.cholesky(D)
        r = D.shape[0]
        Omega = np.random.multivariate_normal(np.zeros(r), np.eye(r), N)
        b = Omega.dot(C.T)
        S = np.vstack((b, -b))
        return S

    def _Lambda_g(self, g, f):
        """blabla

        Parameters
        ----------
        g : `np.array`, shape=(n_samples, 2)
            The value of g function for all samples

        f: `np.array`, shape=(n_samples, 2)
            The value of the f(Y, T, delta| S, G, theta)

        Returns
        -------
        Lambda_g : `list`, shape=(n_samples, )
            The expectation for g
        """
        Lambda_g = 0
        return Lambda_g

    def _Eg(self, pi_xi, Lambda_1, Lambda_g):
        """Computes approximated expectations of different functions g taking
        random effects as input, conditional on the observed data and the
        current estimate of the parameters

        Parameters
        ----------
        g : `np.array`, shape=(n_samples, 2)
            The value of g function for all samples

        f: `np.array`, shape=(n_samples, 2)
            The value of the f(Y, T, delta| S, G, theta)

        Returns
        -------
        Eg : `np.ndarray`, , shape=(n_samples, )
            The expectation for g
        """
        n_samples = f.size()
        Eg = np.zeros(n_samples)
        for i in range(n_samples):
            Eg[i] = ((1 - pi_xi) * (g * self.f[i][0]).sum(axis=0)
                    + self.pi_xi * (g * self.f[i][1]).sum(axis=0)) / \
                    ((1 - self.pi_xi) * (np.array(f[i][0]).sum()) +
                     self.pi_xi * (np.array(f[i][1]).sum()))

        return Eg

    def fit(self, X, Y, T, delta):
        """Fit the lights model

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The simulated longitudinal data. Each element of the dataframe is
            a pandas.Series

        T : `np.ndarray`, shape=(n_samples,)
            Times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator
        """
        verbose = self.verbose
        max_iter = self.max_iter
        print_every = self.print_every
        tol = self.tol
        warm_start = self.warm_start
        fit_intercept = self.fit_intercept
        fixed_effect_time_order = self.fixed_effect_time_order

        n_samples, n_time_indep_features = X.shape
        self.n_samples = n_samples
        self.n_time_indep_features = n_time_indep_features
        n_long_features = Y.shape[1]
        q_l = fixed_effect_time_order + 1
        r_l = 2  # linear time-varying features, so all r_l=2
        nb_asso_param = 4
        if fit_intercept:
            n_time_indep_features += 1
        nb_asso_features = n_long_features * nb_asso_param + n_time_indep_features

        self._start_solve()

        # features extraction
        extracted_features = extract_features(Y, fixed_effect_time_order)

        # initialization
        xi_ext = np.zeros(2 * n_time_indep_features)
        # TODO at the end : try to initialize gamma_0 with a standard Cox model
        #  from tick
        gamma_0_ext = np.zeros(2 * nb_asso_features)
        gamma_1_ext = gamma_0_ext.copy()

        # initialize longitudinal submodels
        if self.initialize:
            mlmm = MLMM(max_iter=max_iter, verbose=verbose, print_every=print_every,
                        tol=tol, fixed_effect_time_order=fixed_effect_time_order)
            mlmm.fit(extracted_features)
            beta = mlmm.beta
            D = mlmm.D
            phi = mlmm.phi
        else:
            # fixed initialization
            q = q_l * n_long_features
            r = r_l * n_long_features
            beta = np.zeros((q, 1))
            D = np.diag(np.ones(r))
            phi = np.ones((n_long_features, 1))

        beta_0_ext = np.concatenate((beta, -beta))
        beta_0_ext[beta_0_ext < 0] = 0
        beta_1_ext = beta_0_ext.copy()

        self.beta_0 = self.get_vect_from_ext(beta_0_ext)
        self.beta_1 = self.get_vect_from_ext(beta_1_ext)
        self.xi = self._get_xi_from_xi_ext(xi_ext)[1]
        self.gamma_0 = self.get_vect_from_ext(gamma_0_ext)
        self.gamma_1 = self.get_vect_from_ext(gamma_1_ext)
        self.D = D
        self.phi = phi

        func_obj = self._func_obj
        P_func = self._P_func
        grad_P = self._grad_P

        obj = func_obj(X, Y, T, delta, xi_ext)

        # bounds vector for the L-BGFS-B algorithms
        bounds_xi = [(0, None)] * n_time_indep_features * 2
        bounds_beta = [(0, None)] * n_long_features * \
                      (fixed_effect_time_order + 1) * 2
        bounds_gamma = [(0, None)] * nb_asso_features * 2

        for n_iter in range(max_iter):

            pi_xi = self.get_proba(X, xi_ext)

            # E-Step
            Lambda_1 = 0
            pi_est = self.get_post_proba(pi_xi, Lambda_1)
            N = 5
            S = self.construct_MC_samples(N)

            # M-Step

            # Update D
            f = self.f_data_g_latent(Y, T, delta, S)

            g0 = []
            for s in S:
                g0.append(s.dot(s.T))
            # TODO : g0(S)
            E_g0 = self._Eg(g0, f)
            D = np.array(E_g0).sum(axis=0) / n_samples

            if warm_start:
                x0 = xi_ext
            else:
                x0 = np.zeros(2 * n_time_indep_features)
            xi_ext = fmin_l_bfgs_b(
                func=lambda xi_ext_: P_func(X, pi_est, xi_ext_),
                x0=x0,
                fprime=lambda xi_ext_: grad_P(X, pi_est, xi_ext_),
                disp=False,
                bounds=bounds_xi,
                maxiter=60,
                pgtol=1e-5
            )[0]

            self.beta_0 = self.get_vect_from_ext(beta_0_ext)
            self.beta_1 = self.get_vect_from_ext(beta_1_ext)
            self.xi = self._get_xi_from_xi_ext(xi_ext)[1]
            self.gamma_0 = self.get_vect_from_ext(gamma_0_ext)
            self.gamma_1 = self.get_vect_from_ext(gamma_1_ext)
            self.D = D
            self.phi = phi

            prev_obj = obj
            obj = func_obj(X, Y, T, delta, xi_ext)
            rel_obj = abs(obj - prev_obj) / abs(prev_obj)

            if n_iter % print_every == 0:
                self.history.update(n_iter=n_iter, obj=obj, rel_obj=rel_obj,
                                    D=D, phi=phi, beta_0=self.beta_0,
                                    beta_1=self.beta_1, xi=self.xi,
                                    gamma_0=self.gamma_0, gamma_1=self.gamma_1)
                if verbose:
                    self.history.print_history()
            if (n_iter > max_iter) or (rel_obj < tol):
                break

        self._end_solve()

    def score(self, X, Y, T, delta, metric):
        """Computes the score with the trained parameters on the given data,
        either log-likelihood or C-index

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The simulated longitudinal data. Each element of the dataframe is
            a pandas.Series

        T : `np.ndarray`, shape=(n_samples,)
            Times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator

        metric : 'log_lik', 'C-index'
            Either computes log-likelihood or C-index

        Returns
        -------
        output : `float`
            The score computed on the given data
        """
        if metric == 'log_lik':
            return self._log_lik(X, Y, T, delta)
        elif metric == 'C-index':
            return c_index_score(T, self.predict_marker(X, Y), delta)
        else:
            raise ValueError("``metric`` must be 'log_lik' or 'C-index', got "
                             "%s instead" % metric)

    def cross_validate(self, X, Y, T, delta, n_folds=10, eta=0.1,
                       adaptative_grid_el=True, grid_size=30,
                       grid_elastic_net=np.array([0]), shuffle=True,
                       verbose=True, metric='C-index'):
        """Apply n_folds randomized search cross-validation using the given
        data, to select the best penalization hyper-parameters

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The simulated longitudinal data. Each element of the dataframe is
            a pandas.Series

        T : `np.ndarray`, shape=(n_samples,)
            Times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator

        n_folds : `int`, default=10
            Number of folds. Must be at least 2.

        eta : `float`, default=0.1
            The ElasticNet mixing parameter, with 0 <= eta <= 1.
            For eta = 0 this is ridge (L2) regularization
            For eta = 1 this is lasso (L1) regularization
            For 0 < eta < 1, the regularization is a linear combination
            of L1 and L2

        adaptative_grid_el : `bool`, default=True
            If `True`, adapt the ElasticNet strength parameter grid using the
            KKT conditions

        grid_size : `int`, default=30
            Grid size if adaptative_grid_el=`True`

        grid_elastic_net : `np.ndarray`, default=np.array([0])
            Grid of ElasticNet strength parameters to be run through, if
            adaptative_grid_el=`False`

        shuffle : `bool`, default=True
            Whether to shuffle the data before splitting into batches

        verbose : `bool`, default=True
            If `True`, we verbose things, otherwise the solver does not
            print anything (but records information in history anyway)

        metric : 'log_lik', 'C-index', default='C-index'
            Either computes log-likelihood or C-index
        """
        n_samples = T.shape[0]
        cv = KFold(n_splits=n_folds, shuffle=shuffle)
        self.grid_elastic_net = grid_elastic_net
        self.adaptative_grid_el = adaptative_grid_el
        self.grid_size = grid_size
        tol = self.tol
        warm_start = self.warm_start

        if adaptative_grid_el:
            # from KKT conditions
            gamma_max = 1. / np.log(10.) * np.log(
                1. / (1. - eta) * (.5 / n_samples)
                * np.absolute(X).sum(axis=0).max())
            grid_elastic_net = np.logspace(gamma_max - 4, gamma_max, grid_size)

        learners = [
            QNMCEM(verbose=False, tol=tol, eta=eta, warm_start=warm_start,
                   fit_intercept=self.fit_intercept)
            for _ in range(n_folds)
        ]

        # TODO Sim: adapt to randomized search

        n_grid_elastic_net = grid_elastic_net.shape[0]
        scores = np.empty((n_grid_elastic_net, n_folds))
        if verbose is not None:
            verbose = self.verbose
        for idx_elasticNet, l_elastic_net in enumerate(grid_elastic_net):
            if verbose:
                print("Testing l_elastic_net=%.2e" % l_elastic_net, "on fold ",
                      end="")
            for n_fold, (idx_train, idx_test) in enumerate(cv.split(X)):
                if verbose:
                    print(" " + str(n_fold), end="")
                X_train, X_test = X[idx_train], X[idx_test]
                T_train, T_test = Y[idx_train], T[idx_test]
                delta_train, delta_test = delta[idx_train], delta[idx_test]
                learner = learners[n_fold]
                learner.l_elastic_net = l_elastic_net
                learner.fit(X_train, T_train, delta_train)
                scores[idx_elasticNet, n_fold] = learner.score(
                    X_test, T_test, delta_test, metric)
            if verbose:
                print(": avg_score=%.2e" % scores[idx_elasticNet, :].mean())

        avg_scores = scores.mean(1)
        std_scores = scores.std(1)
        idx_best = avg_scores.argmax()
        l_elastic_net_best = grid_elastic_net[idx_best]
        idx_chosen = max([i for i, j in enumerate(
            list(avg_scores >= avg_scores.max() - std_scores[idx_best])) if j])
        l_elastic_net_chosen = grid_elastic_net[idx_chosen]

        self.grid_elastic_net = grid_elastic_net
        self.l_elastic_net_best = l_elastic_net_best
        self.l_elastic_net_chosen = l_elastic_net_chosen
        self.scores = scores
        self.avg_scores = avg_scores
