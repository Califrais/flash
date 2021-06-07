import numpy as np
import pandas as pd
import copt
import warnings
from scipy.optimize import fmin_l_bfgs_b
from numpy.linalg import multi_dot
from lifelines.utils import concordance_index as c_index_score
from lights.base.base import Learner, extract_features, normalize, block_diag, \
    get_xi_from_xi_ext, logistic_grad, get_times_infos, get_ext_from_vect, \
    get_vect_from_ext
from lights.init.mlmm import MLMM
from lights.init.cox import initialize_asso_params
from lights.model.e_step_functions import EstepFunctions
from lights.model.m_step_functions import MstepFunctions
from lights.model.regularizations import ElasticNet, SparseGroupL1


class QNMCEM(Learner):
    """QNMCEM Algorithm for the lights model inference

    Parameters
    ----------
    fit_intercept : `bool`, default=True
        If `True`, include an intercept in the model for the time independent
        features

    l_pen_EN : `float`, default=0.
        Level of penalization for the ElasticNet

    l_pen_SGL_beta : `float`, default=0.
        Level of penalization for the Sparse Group l1 on beta

    l_pen_SGL_gamma : `float`, default=0.
        Level of penalization for the Sparse Group l1 on gamma

    eta_elastic_net: `float`, default=0.1
        The ElasticNet mixing parameter, with 0 <= eta_elastic_net <= 1.
        For eta_elastic_net = 0 this is ridge (L2) regularization
        For eta_elastic_net = 1 this is lasso (L1) regularization
        For 0 < eta_elastic_net < 1, the regularization is a linear combination
        of L1 and L2

    eta_sp_gp_l1: `float`, default=0.1
        The Sparse Group l1 mixing parameter, with 0 <= eta_sp_gp_l1 <= 1

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

    warm_start : `bool`, default=True
        If true, learning will start from the last reached solution

    fixed_effect_time_order : `int`, default=5
        Order of the higher time monomial considered for the representations of
        the time-varying features corresponding to the fixed effect. The
        dimension of the corresponding design matrix is then equal to
        fixed_effect_time_order + 1

    asso_functions : `list` or `str`='all', default='all'
        List of association functions wanted or string 'all' to select all
        defined association functions. The available functions are :
            - 'lp' : linear predictor
            - 're' : random effects
            - 'tps' : time dependent slope
            - 'ce' : cumulative effects

    initialize : `bool`, default=True
        If `True`, we initialize the parameters using MLMM model, otherwise we
        use arbitrarily chosen fixed initialization

    copt_accelerate : `bool`, default=False
        If `True`, we choose copt solver with accelerated proximal
        gradient (FISTA), otherwise we use regular ISTA

    compute_obj : `bool`, default=False
        If `True`, we compute the global objective to be minimized by the QNMCEM
         algorithm and store it in history

    MC_sep: `bool`, default=False
        If `False`, we use the same set of MC samples for all subject,
        otherwise we sample a seperate set of MC samples for each subject

    copt_solver_step : function or `str`='backtracking', default='backtracking'
        Step size for optimization algorithm used in Copt colver
    """

    def __init__(self, fit_intercept=False, l_pen_EN=0., l_pen_SGL_beta=0.,
                 l_pen_SGL_gamma=0., eta_elastic_net=.1, eta_sp_gp_l1=.1,
                 max_iter=100, verbose=True, print_every=10, tol=1e-5,
                 warm_start=True, fixed_effect_time_order=5,
                 asso_functions='all', initialize=True, copt_accelerate=False,
                 compute_obj=False, MC_sep=False,
                 copt_solver_step='backtracking'):
        Learner.__init__(self, verbose=verbose, print_every=print_every)
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept
        self.fixed_effect_time_order = fixed_effect_time_order
        self.asso_functions = asso_functions
        self.initialize = initialize
        self.copt_accelerate = copt_accelerate
        self.l_pen_EN = l_pen_EN
        self.l_pen_SGL_beta = l_pen_SGL_beta
        self.l_pen_SGL_gamma = l_pen_SGL_gamma
        self.eta_elastic_net = eta_elastic_net
        self.eta_sp_gp_l1 = eta_sp_gp_l1
        self.ENet = ElasticNet(l_pen_EN, eta_elastic_net)
        self._fitted = False
        self.compute_obj = compute_obj
        self.MC_sep = MC_sep

        # Attributes that will be instantiated afterwards
        self.n_samples = None
        self.n_time_indep_features = None
        self.n_long_features = None
        self.S = None
        self.T_u = None
        self.theta = {
            "beta_0": np.empty(1),
            "beta_1": np.empty(1),
            "long_cov": np.empty(1),
            "phi": np.empty(1),
            "xi": np.empty(1),
            "baseline_hazard": pd.Series(),
            "gamma_0": np.empty(1),
            "gamma_1": np.empty(1)
        }
        self.copt_step = copt_solver_step

    @property
    def asso_functions(self):
        return self._asso_functions

    @asso_functions.setter
    def asso_functions(self, val):
        if not (val == 'all' or set(val).issubset({'lp', 're', 'tps', 'ce'})):
            raise ValueError("``asso_functions`` must be either 'all', or a "
                             "`list` in ['lp', 're', 'tps', 'ce']")
        self._asso_functions = val

    @property
    def copt_accelerate(self):
        return self._copt_accelerate

    @copt_accelerate.setter
    def copt_accelerate(self, val):
        if val and self.warm_start:
            warnings.warn("Careful using simultaneously ``warm_start`` and "
                          "``copt_accelerate``: warmstart can diminish FISTA "
                          "acceleration effectiveness")
        self._copt_accelerate = val

    @property
    def fitted(self):
        return self._fitted

    @staticmethod
    def _rel_theta(theta, prev_theta, eps):
        """Computes the relative difference of the current estimated parameters
        with the previous one.

        Parameters
        ----------
        theta : `dictionary`
            Dictionary of current estimated parameters

        pre_theta : `dictionary`
            Dictionary of previous iteration estimated parameters

        eps : float
            The value of epsilon

        Returns
        -------
        rel : `float`
            The computed relative difference
        """
        rel = 0
        for key_ in theta.keys():
            tmp = np.linalg.norm(theta[key_] - prev_theta[key_]) / \
                  (np.linalg.norm(theta[key_]) + eps)
            rel = max(rel, tmp)
        return rel

    @staticmethod
    def _log_lik(pi_xi, f_mean):
        """Computes the approximation of the likelihood of the lights model

        Parameters
        ----------
        pi_xi : `np.ndarray`, shape=(n_samples,)
            Probability estimates for being on the high-risk group given
            time-independent features

        f_mean : `np.ndarray`, shape=(n_samples, K)
            The mean value of f(Y, T, delta| S, G ; theta) over S

        Returns
        -------
        prb : `float`
            The approximated log-likelihood computed on the given data
        """
        pi_xi_ = np.vstack((1 - pi_xi, pi_xi)).T
        prb = np.log((pi_xi_ * f_mean).sum(axis=-1)).mean()
        return prb

    def _func_obj(self, pi_xi, f_mean):
        """The global objective to be minimized by the QNMCEM algorithm
        (including penalization)

        Parameters
        ----------
        pi_xi : `np.ndarray`, shape=(n_samples,)
            Probability estimates for being on the high-risk group given
            time-independent features

        f_mean : `np.ndarray`, shape=(n_samples, K)
            The mean value of f(Y, T, delta| S, G ; theta) over S

        Returns
        -------
        output : `float`
            The value of the global objective to be minimized
        """
        p, L = self.n_time_indep_features, self.n_long_features
        eta_sp_gp_l1 = self.eta_sp_gp_l1
        l_pen_SGL_beta = self.l_pen_SGL_beta
        l_pen_SGL_gamma = self.l_pen_SGL_gamma
        theta = self.theta
        log_lik = self._log_lik(pi_xi, f_mean)
        # xi elastic net penalty
        xi = theta["xi"]
        xi_pen = self.ENet.pen(xi)
        # beta sparse group l1 penalty
        beta_0, beta_1 = theta["beta_0"], theta["beta_1"]
        groups = np.arange(0, len(beta_0)).reshape(L, -1).tolist()
        SGL1 = SparseGroupL1(l_pen_SGL_beta, eta_sp_gp_l1, groups)
        beta_0_pen = SGL1.pen(beta_0)
        beta_1_pen = SGL1.pen(beta_1)
        # gamma sparse group l1 penalty
        gamma_0, gamma_1 = theta["gamma_0"], theta["gamma_1"]
        gamma_0_x, gamma_1_x = theta["gamma_0_x"], theta["gamma_1_x"]
        gamma_0_pen = self.ENet.pen(gamma_0_x)
        groups = np.arange(0, len(gamma_0)).reshape(L, -1).tolist()
        SGL1 = SparseGroupL1(l_pen_SGL_gamma, eta_sp_gp_l1, groups)
        gamma_0_pen += SGL1.pen(gamma_0)
        gamma_1_pen = self.ENet.pen(gamma_1_x)
        gamma_1_pen += SGL1.pen(gamma_1)
        pen = xi_pen + beta_0_pen + beta_1_pen + gamma_0_pen + gamma_1_pen
        return -log_lik + pen

    def _get_proba(self, X):
        """Probability estimates for being on the high-risk group given
        time-independent features

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        Returns
        -------
        output : `np.ndarray`, shape=(n_samples,)
            Returns the probability of the sample for being on the high-risk
            group given time-independent features
        """
        xi_0, xi = self.theta["xi_0"], self.theta["xi"]
        u = xi_0 + X.dot(xi)
        return logistic_grad(u)

    @staticmethod
    def _get_post_proba(pi_xi, Lambda_1):
        """Posterior probability estimates for being on the high-risk group
        given all observed data

        Parameters
        ----------
        pi_xi : `np.ndarray`, shape=(n_samples,)
            Comes from get_proba function

        Lambda_1 : `np.ndarray`, shape=(n_samples, K)
            Approximated integral (see (15) in the lights paper) with
            \tilde(g)=1

        Returns
        -------
        pi_est : `np.ndarray`, shape=(n_samples,)
            Returns the posterior probability of the sample for being on the
            high-risk group given all observed data
        """
        tmp = Lambda_1 * np.vstack((1 - pi_xi, pi_xi)).T
        pi_est = tmp[:, 1] / tmp.sum(axis=1)
        return pi_est

    @staticmethod
    def intensity(rel_risk, indicator):
        """Compute the intensity of f_data_given_latent

        Parameters
        ----------
        rel_risk: `np.ndarray`, shape=(N_MC, K, n_samples, J)
            The relative risk

        indicator: `np.ndarray`, shape=(n_samples, J)
            The indicator matrix for comparing event times (T == T_u)

        Returns
        -------
        intensity : `np.ndarray`, shape=(N_MC, K, n_samples)
            The value of intensity
        """
        intensity = (rel_risk * indicator).sum(axis=-1)
        return intensity

    @staticmethod
    def survival(rel_risk, indicator):
        """Computes the survival function

        Parameters
        ----------
        rel_risk: `np.ndarray`, shape=(N_MC, K, n_samples, J)
            The relative risk

        indicator: `np.ndarray`, shape=(n_samples, J)
            The indicator matrix for comparing event times (T <= T_u)

        Returns
        -------
        survival : `np.ndarray`, shape=(n_samples, K, N_MC)
            The value of the survival function
        """
        survival = np.exp(-(rel_risk * indicator).sum(axis=-1).T)
        return survival

    def f_y_given_latent(self, extracted_features, g3):
        """Computes the density of the longitudinal processes given latent
        variables

        Parameters
        ----------
        extracted_features :  `tuple, tuple`,
            The extracted features from longitudinal data.
            Each tuple is a combination of fixed-effect design features,
            random-effect design features, outcomes, number of the longitudinal
            measurements for all subject or arranged by l-th order.

        g3 : `list` of n_samples `np.array`s with shape=(K, n_i, N_MC)
            The values of g3 function

        Returns
        -------
        f_y : `np.ndarray`, shape=(n_samples, K, N_MC)
            The value of the f(Y | S, G ; theta)
        """
        (U_list, V_list, y_list, N_list) = extracted_features[0]
        n_samples, n_long_features = self.n_samples, self.n_long_features
        phi = self.theta["phi"]
        N_MC = g3[0].shape[2]
        K = 2  # 2 latent groups
        f_y = np.ones(shape=(n_samples, K, N_MC))
        for i in range(n_samples):
            n_i, y_i, M_iS = sum(N_list[i]), y_list[i], g3[i]
            inv_Phi_i = [[phi[l, 0]] * N_list[i][l] for l in
                         range(n_long_features)]
            inv_Phi_i = np.concatenate(inv_Phi_i).reshape(-1, 1)
            f_y[i] = (1 / (np.sqrt(((2 * np.pi) ** n_i) * np.prod(inv_Phi_i)))
                      * np.exp(
                        np.sum(-0.5 * ((y_i - M_iS) ** 2) / inv_Phi_i, axis=1)))
        return f_y

    def mlmm_density(self, extracted_features):
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
        theta = self.theta
        D, phi = theta["long_cov"], theta["phi"]
        beta_0, beta_1 = theta["beta_0"], theta["beta_1"]
        beta_stack = np.hstack((beta_0, beta_1))

        log_lik = np.zeros((n_samples, 2))
        for i in range(n_samples):
            U_i, V_i, y_i, n_i = U_list[i], V_list[i], y_list[i], sum(N[i])
            inv_Phi_i = [[phi[l, 0]] * N[i][l] for l in range(n_long_features)]
            inv_Sigma_i = np.diag(np.concatenate(inv_Phi_i))
            tmp_1 = multi_dot([V_i, D, V_i.T]) + inv_Sigma_i
            tmp_2 = y_i - U_i.dot(beta_stack)

            op1 = n_i * np.log(2 * np.pi)
            op2 = np.log(np.linalg.det(tmp_1))
            op3 = np.diag(multi_dot([tmp_2.T, np.linalg.inv(tmp_1), tmp_2]))

            log_lik[i] = np.exp(-.5 * (op1 + op2 + op3))

        return log_lik

    def f_data_given_latent(self, X, extracted_features, T, T_u, delta, S,
                            MC_sep):
        """Estimates the data density given latent variables

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        extracted_features :  `tuple, tuple`,
            The extracted features from longitudinal data.
            Each tuple is a combination of fixed-effect design features,
            random-effect design features, outcomes, number of the longitudinal
            measurements for all subject or arranged by l-th order.

        T : `np.ndarray`, shape=(n_samples,)
            Censored times of the event of interest

        T_u : `np.ndarray`, shape=(J,)
            The J unique training censored times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator

        S : `np.ndarray`, shape=(N_MC, r)
            Set of constructed Monte Carlo samples

        MC_sep: `bool`, default=False
        If `False`, we use the same set of MC samples for all subject,
        otherwise we sample a seperate set of MC samples for each subject

        Returns
        -------
        f : `np.ndarray`, shape=(n_samples, K, N_MC)
            The value of the f(Y, T, delta| S, G ; theta)
        """
        theta, alpha = self.theta, self.fixed_effect_time_order
        baseline_hazard, phi = theta["baseline_hazard"], theta["phi"]
        E_func = EstepFunctions(X, T, T_u, delta, extracted_features, alpha,
                                self.asso_functions, theta, MC_sep)
        beta_0, beta_1 = theta["beta_0"], theta["beta_1"]
        gamma_0, gamma_1 = theta["gamma_0"], theta["gamma_1"]
        gamma_0_x, gamma_1_x = theta["gamma_0_x"], theta["gamma_1_x"]
        E_func.compute_AssociationFunctions(S)
        g1 = E_func.g1(S, gamma_0, gamma_0_x, gamma_1, gamma_1_x, False)
        g3 = E_func.g3(S, beta_0, beta_1)
        baseline_val = baseline_hazard.values.flatten()
        rel_risk = g1.swapaxes(0, 2) * baseline_val
        _, ind_1, ind_2 = get_times_infos(T, T_u)
        intensity = self.intensity(rel_risk, ind_1)
        survival = self.survival(rel_risk, ind_2)
        f = (intensity ** delta).T * survival
        if not self.MC_sep:
            f_y = self.f_y_given_latent(extracted_features, g3)
            f *= f_y
        return f

    def predict_marker(self, X, Y, prediction_times=None):
        """Marker rule of the lights model for being on the high-risk group

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The longitudinal data. Each element of the dataframe is
            a pandas.Series

        prediction_times : `np.ndarray`, shape=(n_samples,), default=None
            Times for prediction, that is up to which one has longitudinal data.
            If `None`, takes the last measurement times in Y

        Returns
        -------
        marker : `np.ndarray`, shape=(n_samples,)
            Returns the marker rule of the sample for being on the high-risk
            group
        """
        if self._fitted:
            n_samples = X.shape[0]
            theta, alpha = self.theta, self.fixed_effect_time_order
            ext_feat = extract_features(Y, alpha)
            last_measurement = np.array(list(map(max, ext_feat[0][2])))
            if prediction_times is None:
                prediction_times = last_measurement
            else:
                if not (prediction_times > last_measurement).all():
                    raise ValueError('Prediction times must be greater than the'
                                     ' last measurement times for each subject')

            # predictions for alive subjects only
            delta_prediction = np.zeros(n_samples)
            T_u = self.T_u
            f = self.f_data_given_latent(X, ext_feat, prediction_times, T_u,
                                         delta_prediction, self.S, self.MC_sep)
            pi_xi = self._get_proba(X)
            marker = self._get_post_proba(pi_xi, f.mean(axis=-1))
            return marker
        else:
            raise ValueError('You must fit the model first')

    def _update_theta(self, **kwargs):
        """Update class attributes corresponding to lights model parameters
        """
        for key, value in kwargs.items():
            if key in ["long_cov", "phi", "baseline_hazard",
                       "beta_0", "beta_1", "gamma_0", "gamma_1",
                       "gamma_0_x", "gamma_1_x"]:
                self.theta[key] = value
            elif key == "xi":
                xi_0, xi = get_xi_from_xi_ext(value, self.fit_intercept)
                self.theta["xi_0"], self.theta["xi"] = xi_0, xi
            else:
                raise ValueError('Parameter {} is not defined'.format(key))

    def fit(self, X, Y, T, delta):
        """Fits the lights model

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The longitudinal data. Each element of the dataframe is
            a pandas.Series

        T : `np.ndarray`, shape=(n_samples,)
            Censored times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator
        """
        self._start_solve()
        verbose = self.verbose
        max_iter = self.max_iter
        print_every = self.print_every
        tol = self.tol
        warm_start = self.warm_start
        fit_intercept = self.fit_intercept
        alpha = self.fixed_effect_time_order
        n_samples, p = X.shape
        L = Y.shape[1]
        self.n_samples = n_samples
        self.n_time_indep_features = p
        self.n_long_features = L
        q_l = alpha + 1
        r_l = 2  # Affine random effects
        if fit_intercept:
            p += 1

        if self.asso_functions == 'all':
            self.asso_functions = ['lp', 're', 'tps', 'ce']
        asso_functions = self.asso_functions
        nb_asso_param = len(asso_functions)
        if 're' in asso_functions:
            nb_asso_param += 1
        N = 10  # Number of initial Monte Carlo sample for S

        X = normalize(X)  # Normalize time-independent features
        ext_feat = extract_features(Y, alpha)  # Features extraction
        T_u = np.unique(T)
        self.T_u = T_u
        J, ind_1, ind_2 = get_times_infos(T, T_u)

        # Initialization
        xi_ext = .5 * np.concatenate((np.ones(p), np.zeros(p)))

        if self.initialize:
            # Initialize longitudinal submodels
            mlmm = MLMM(max_iter=max_iter, verbose=verbose, tol=tol,
                        print_every=print_every, fixed_effect_time_order=alpha)
            mlmm.fit(ext_feat)
            beta = mlmm.fixed_effect_coeffs
            D = mlmm.long_cov
            phi = mlmm.phi
            est = initialize_asso_params(X, T, delta)
            time_indep_cox_coeffs, baseline_hazard = est
        else:
            # Fixed initialization
            q = q_l * L
            r = r_l * L
            beta = np.zeros((q, 1))
            D = np.diag(np.ones(r))
            phi = np.ones((L, 1))
            time_indep_cox_coeffs = np.zeros(p)
            baseline_hazard = pd.Series(data=.5 * np.ones(J), index=T_u)

        beta_0 = beta.reshape(-1, 1)
        beta_1 = beta_0.copy()

        gamma_0_x = time_indep_cox_coeffs.reshape(-1, 1)
        gamma_0_x_ext = get_ext_from_vect(gamma_0_x)
        gamma_0 = 1e-4 * np.ones((L * nb_asso_param, 1))
        gamma_1_x = gamma_0_x.copy()
        gamma_1_x_ext = gamma_0_x_ext.copy()
        gamma_1 = gamma_0.copy()

        self._update_theta(beta_0=beta_0, beta_1=beta_1, xi=xi_ext,
                           gamma_0=gamma_0, gamma_1=gamma_1,
                           gamma_0_x=gamma_0_x, gamma_1_x=gamma_1_x, long_cov=D,
                           phi=phi, baseline_hazard=baseline_hazard)

        # Stopping criteria and bounds vector for the L-BGFS-B algorithm
        maxiter, pgtol = 60, 1e-5
        bounds_xi = [(0, None)] * 2 * p
        bounds_gamma_time_indep = [(0, None)] * 2 * p

        # Instanciates E-step and M-step functions
        E_func = EstepFunctions(X, T, T_u, delta, ext_feat, alpha,
                                asso_functions, self.theta, self.MC_sep)
        F_func = MstepFunctions(fit_intercept, X, T, delta, L, p, self.l_pen_EN,
                                self.eta_elastic_net, alpha, asso_functions)

        S = E_func.construct_MC_samples(N)
        f = self.f_data_given_latent(X, ext_feat, T, self.T_u, delta, S,
                                     self.MC_sep)
        Lambda_1 = E_func.Lambda_g(np.ones(shape=(n_samples, 2, 2 * N)), f)
        pi_xi = self._get_proba(X)

        # Store init values
        if self.compute_obj:
            f_mean = f.mean(axis=-1)
            if self.MC_sep:
                f_mean *= self.mlmm_density(ext_feat)
            obj = self._func_obj(pi_xi, f_mean)
            self.history.update(n_iter=0, obj=obj,
                                rel_obj=np.inf, theta=self.theta)
        else:
            self.history.update(n_iter=0, theta=self.theta)
        if verbose:
            self.history.print_history()

        prev_theta = self.theta.copy()
        stopping_criterion_count = 0
        for n_iter in range(1, max_iter + 1):

            # E-Step
            pi_est = self._get_post_proba(pi_xi, Lambda_1)
            E_g4 = E_func.Eg(E_func.g4(S), Lambda_1, pi_xi, f)
            E_g5 = E_func.Eg(E_func.g5(S), Lambda_1, pi_xi, f)

            def E_g1(gamma_0_, gamma_0_x_, gamma_1_, gamma_1_x_):
                return E_func.Eg(
                    E_func.g1(S, gamma_0_, gamma_0_x_, gamma_1_,
                              gamma_1_x_), Lambda_1, pi_xi, f)

            def E_log_g1(gamma_0_, gamma_0_x_, gamma_1_, gamma_1_x_):
                return E_func.Eg(
                    np.log(E_func.g1(S, gamma_0_, gamma_0_x_, gamma_1_,
                                     gamma_1_x_)), Lambda_1, pi_xi, f)

            def E_g6(gamma_0_, gamma_0_x_, gamma_1_, gamma_1_x_):
                return E_func.Eg(
                    E_func.g6(S, gamma_0_, gamma_0_x_, gamma_1_, gamma_1_x_),
                    Lambda_1, pi_xi, f)

            # M-Step
            D = E_g4.sum(axis=0) / n_samples  # D update

            if warm_start:
                xi_init = xi_ext
                beta_init = [beta_0.flatten(), beta_1.flatten()]
                gamma_x_init = [gamma_0_x_ext.flatten(),
                                gamma_1_x_ext.flatten()]
                gamma_init = [gamma_0.flatten(), gamma_1.flatten()]
            else:
                xi_init = np.zeros(2 * p)
                beta_init = [np.zeros(L * q_l),
                             np.zeros(L * q_l)]
                gamma_x_init = [np.zeros(2 * p), np.zeros(2 * p)]
                gamma_init = [np.zeros(L * nb_asso_param),
                              np.zeros(L * nb_asso_param)]

            # xi update
            xi_ext = fmin_l_bfgs_b(
                func=lambda xi_ext_: F_func.P_pen_func(pi_est, xi_ext_),
                x0=xi_init,
                fprime=lambda xi_ext_: F_func.grad_P_pen(pi_est, xi_ext_),
                disp=False, bounds=bounds_xi, maxiter=maxiter, pgtol=pgtol)[0]

            # beta update
            K = 2
            pi_est_K = np.vstack((1 - pi_est, pi_est))
            (U_list, V_list, y_list, _) = ext_feat[0]
            num = np.zeros((K, L * q_l))
            den = np.zeros((K, L * q_l, L * q_l))
            if self.MC_sep:
                None
                #TODO: Update later
            else:
                for i in range(n_samples):
                    U_i, V_i, y_i = U_list[i], V_list[i], y_list[i]
                    tmp_num = U_i.T.dot((y_i.flatten() - V_i.dot(E_g5[i])))
                    tmp_den = U_i.T.dot(U_i)
                    for k in range(K):
                        num[k] += pi_est_K[k, i] * tmp_num
                        den[k] += pi_est_K[k, i] * tmp_den

            # beta_0
            beta_0 = np.linalg.inv(den[0]).dot(num[0]).reshape(-1, 1)

            # beta_1
            beta_0 = np.linalg.inv(den[1]).dot(num[1]).reshape(-1, 1)

            self._update_theta(beta_0=beta_0, beta_1=beta_1)

            # gamma_0 update
            beta_K = [beta_0, beta_1]
            gamma = [gamma_0, gamma_1]
            gamma_x = [gamma_0_x, gamma_1_x]
            groups = np.arange(0, len(gamma_0)).reshape(L, -1).tolist()
            eta_sp_gp_l1 = self.eta_sp_gp_l1
            l_pen_SGL_gamma = self.l_pen_SGL_gamma
            prox = SparseGroupL1(l_pen_SGL_gamma, eta_sp_gp_l1, groups).prox
            copt_max_iter = 1
            args_all = {"pi_est": pi_est_K, "E_g5": E_g5,
                        "phi": phi, "beta": beta_K,
                        "baseline_hazard": baseline_hazard,
                        "extracted_features": ext_feat,
                        "ind_1": ind_1 * 1, "ind_2": ind_2 * 1,
                        "gamma": gamma,
                        "gamma_x": gamma_x}
            E_func.compute_AssociationFunctions(S)
            F_func.grad_Q_fixed_stuff(beta_K, E_g5, args_all["ind_1"])
            args_0_x = {"E_g1": lambda v: E_g1(gamma_0, v, gamma_1, gamma_1_x),
                        "E_log_g1": lambda v: E_log_g1(gamma_0, v, gamma_1,
                                                       gamma_1_x),
                        "E_g6": lambda v: E_g6(gamma_0, v, gamma_1, gamma_1_x),
                        "group": 0}
            gamma_0_prev = gamma_0.copy()
            gamma_0_x_prev = gamma_0_x.copy()
            # time independence part
            gamma_0_x_ext = fmin_l_bfgs_b(
                func=lambda gamma_0_x_ext_: F_func.Q_x_pen_func(
                    gamma_0_x_ext_, *[{**args_all, **args_0_x}]),
                x0=gamma_x_init[0],
                fprime=lambda gamma_0_x_ext_: F_func.grad_Q_x_pen(
                    gamma_0_x_ext_, *[{**args_all, **args_0_x}]),
                disp=False, bounds=bounds_gamma_time_indep, maxiter=maxiter,
                pgtol=pgtol)[0]
            gamma_0_x = get_vect_from_ext(gamma_0_x_ext).reshape(-1, 1)

            # time dependence part
            args_0 = {"E_g1": lambda v: E_g1(v, gamma_0_x, gamma_1, gamma_1_x),
                      "E_log_g1": lambda v: E_log_g1(v, gamma_0_x, gamma_1,
                                                     gamma_1_x),
                      "E_g6": lambda v: E_g6(v, gamma_0_x, gamma_1, gamma_1_x),
                      "group": 0}
            gamma_0 = copt.minimize_proximal_gradient(
                fun=F_func.Q_func, x0=gamma_init[0], prox=prox,
                max_iter=copt_max_iter,
                args=[{**args_all, **args_0}], jac=F_func.grad_Q,
                step=self.copt_step,
                accelerated=self.copt_accelerate).x.reshape(-1, 1)

            # gamma_1 update
            args_1_x = {"E_g1": lambda v: E_g1(gamma_0_prev, gamma_0_x_prev,
                                               gamma_1, v),
                      "E_log_g1": lambda v: E_log_g1(gamma_0_prev,
                                            gamma_0_x_prev, gamma_1, v),
                      "E_g6": lambda v: E_g6(gamma_0_prev, gamma_0_x_prev,
                                            gamma_1, v),
                      "group": 1}
            # time independence part
            gamma_1_x_ext = fmin_l_bfgs_b(
                func=lambda gamma_1_x_ext_: F_func.Q_x_pen_func(
                    gamma_1_x_ext_, *[{**args_all, **args_1_x}]),
                x0=gamma_x_init[1],
                fprime=lambda gamma_1_indep_ext_: F_func.grad_Q_x_pen(
                    gamma_1_indep_ext_, *[{**args_all, **args_1_x}]),
                disp=False, bounds=bounds_gamma_time_indep, maxiter=maxiter,
                pgtol=pgtol)[
                0]
            gamma_1_x = get_vect_from_ext(gamma_1_x_ext).reshape(-1, 1)

            # time dependence part
            args_1 = {"E_g1": lambda v: E_g1(gamma_0_prev, gamma_0_x_prev,
                                               v, gamma_1_x),
                      "E_log_g1": lambda v: E_log_g1(gamma_0_prev,
                                            gamma_0_x_prev, v, gamma_1_x),
                      "E_g6": lambda v: E_g6(gamma_0_prev, gamma_0_x_prev,
                                            v, gamma_1_x),
                      "group": 1}
            gamma_1 = copt.minimize_proximal_gradient(
                fun=F_func.Q_func, x0=gamma_init[1], prox=prox,
                max_iter=copt_max_iter,
                args=[{**args_all, **args_1}], jac=F_func.grad_Q,
                step=self.copt_step,
                accelerated=self.copt_accelerate).x.reshape(-1, 1)

            # beta, gamma needs to be updated before the baseline
            self._update_theta(gamma_0=gamma_0, gamma_1=gamma_1,
                               gamma_0_x=gamma_0_x, gamma_1_x=gamma_1_x)
            E_func.theta = self.theta
            E_g1 = E_func.Eg(E_func.g1(S, gamma_0, gamma_0_x, gamma_1,
                                       gamma_1_x), Lambda_1, pi_xi, f)

            # baseline hazard update
            baseline_hazard = pd.Series(
                data=((((ind_1 * 1).T * delta).sum(axis=1)) /
                      ((E_g1.T * (ind_2 * 1).T).swapaxes(0, 1) * pi_est_K)
                      .sum(axis=2).sum(axis=1)), index=T_u)

            # phi update
            beta_stack = np.hstack((beta_0, beta_1))
            (U_L, V_L, y_L, N_L) = ext_feat[1]
            phi = np.zeros((L, 1))
            for l in range(L):
                pi_est_ = np.concatenate([[pi_est[i]] * N_L[l][i]
                                          for i in range(n_samples)])
                pi_est_stack = np.vstack((1 - pi_est_, pi_est_)).T  # K = 2
                N_l, y_l, U_l, V_l = sum(N_L[l]), y_L[l], U_L[l], V_L[l]
                beta_l = beta_stack[q_l * l: q_l * (l + 1)]
                E_g5_l = E_g5.reshape(n_samples, L, q_l)[:, l].reshape(-1, 1)
                E_g4_l = block_diag(E_g4[:, r_l * l: r_l * (l + 1),
                                    r_l * l: r_l * (l + 1)])
                tmp = y_l - U_l.dot(beta_l)
                phi_l = (pi_est_stack * (
                        tmp * (tmp - 2 * (V_l.dot(E_g5_l))))).sum() \
                        + np.trace(V_l.T.dot(V_l).dot(E_g4_l))
                phi[l] = phi_l / N_l

            self._update_theta(phi=phi, baseline_hazard=baseline_hazard,
                               long_cov=D, xi=xi_ext)
            pi_xi = self._get_proba(X)
            E_func.theta = self.theta
            S = E_func.construct_MC_samples(N)
            f = self.f_data_given_latent(X, ext_feat, T, T_u, delta, S, self.MC_sep)

            rel_theta = self._rel_theta(self.theta, prev_theta, 1e-2)
            prev_theta = self.theta.copy()
            if n_iter % print_every == 0:
                if self.compute_obj:
                    prev_obj = obj
                    f_mean = f.mean(axis=-1)
                    if self.MC_sep:
                        f_mean *= self.mlmm_density(ext_feat)
                    obj = self._func_obj(pi_xi, f_mean)
                    rel_obj = abs(obj - prev_obj) / abs(prev_obj)
                    self.history.update(n_iter=n_iter, theta=self.theta,
                                        obj=obj, rel_obj=rel_obj)
                else:
                    self.history.update(n_iter=n_iter, theta=self.theta)
                if verbose:
                    self.history.print_history()
            if rel_theta < tol:
                stopping_criterion_count += 1
            else:
                stopping_criterion_count = 0

            if (n_iter + 1 > max_iter) or (stopping_criterion_count == 3):
                self._fitted = True
                self.S = S  # useful for predictions
                break
            else:
                # Update for next iteration
                Lambda_1 = E_func.Lambda_g(np.ones((n_samples, 2, 2 * N)), f)

        self._end_solve()

    def score(self, X, Y, T, delta):
        """Computes the C-index score with the trained parameters on the given
        data

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The longitudinal data. Each element of the dataframe is
            a pandas.Series

        T : `np.ndarray`, shape=(n_samples,)
            Censored times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator

        Returns
        -------
        output : `float`
            The C-index score computed on the given data
        """
        if self._fitted:
            c_index = c_index_score(T, self.predict_marker(X, Y), delta)
            return max(c_index, 1 - c_index)
        else:
            raise ValueError('You must fit the model first')
