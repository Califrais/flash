import numpy as np
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b
from lifelines.utils import concordance_index as c_index_score
from lights.base.base import Learner, extract_features, normalize, block_diag, \
    get_xi_from_xi_ext, logistic_grad, get_times_infos
from lights.init.mlmm import MLMM
from lights.init.cox import initialize_asso_params
from lights.model.e_step_functions import EstepFunctions
from lights.model.m_step_functions import MstepFunctions
from lights.model.regularizations import ElasticNet, SparseGroupL1
import copt


class QNMCEM(Learner):
    """QNMCEM Algorithm for the lights model inference

    Parameters
    ----------
    fit_intercept : `bool`, default=True
        If `True`, include an intercept in the model for the time independent
        features

    l_pen : `float`, default=0
        Level of penalization for the ElasticNet and the Sparse Group l1

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
    """

    def __init__(self, fit_intercept=False, l_pen=0., eta_elastic_net=.1,
                 eta_sp_gp_l1=.1, max_iter=100, verbose=True, print_every=10,
                 tol=1e-5, warm_start=True, fixed_effect_time_order=5,
                 asso_functions='all', initialize=True):
        Learner.__init__(self, verbose=verbose, print_every=print_every)
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept
        self.fixed_effect_time_order = fixed_effect_time_order
        self.asso_functions = asso_functions
        self.initialize = initialize
        self.l_pen = l_pen
        self.eta_elastic_net = eta_elastic_net
        self.eta_sp_gp_l1 = eta_sp_gp_l1
        self.ENet = ElasticNet(l_pen, eta_elastic_net)
        self._fitted = False

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
    def fitted(self):
        return self._fitted

    @staticmethod
    def _log_lik(pi_xi, f):
        """Computes the approximation of the likelihood of the lights model

        Parameters
        ----------
        pi_xi : `np.ndarray`, shape=(n_samples,)
            Probability estimates for being on the high-risk group given
            time-independent features

        f : `np.ndarray`, shape=(n_samples, K, N_MC)
            The value of f(Y, T, delta| S, G ; theta)

        Returns
        -------
        prb : `float`
            The approximated log-likelihood computed on the given data
        """
        pi_xi_ = np.vstack((1 - pi_xi, pi_xi)).T
        prb = np.log((pi_xi_ * f.mean(axis=-1)).sum(axis=-1)).mean()
        return prb

    def _func_obj(self, pi_xi, f):
        """The global objective to be minimized by the QNMCEM algorithm
        (including penalization)

        Parameters
        ----------
        pi_xi : `np.ndarray`, shape=(n_samples,)
            Probability estimates for being on the high-risk group given
            time-independent features

        f : `np.ndarray`, shape=(n_samples, K, N_MC)
            The value of f(Y, T, delta| S, G ; theta)

        Returns
        -------
        output : `float`
            The value of the global objective to be minimized
        """
        p, L = self.n_time_indep_features, self.n_long_features
        eta, l_pen = self.eta_sp_gp_l1, self.l_pen
        theta = self.theta
        log_lik = self._log_lik(pi_xi, f)
        # xi elastic net penalty
        xi = theta["xi"]
        xi_pen = self.ENet.pen(xi)
        # beta sparse group l1 penalty
        beta_0, beta_1 = theta["beta_0"], theta["beta_1"]
        groups = np.arange(0, len(beta_0)).reshape(L, -1).tolist()
        SGL1 = SparseGroupL1(l_pen, eta, groups)
        beta_0_pen = SGL1.pen(beta_0)
        beta_1_pen = SGL1.pen(beta_1)
        # gamma sparse group l1 penalty
        gamma_0, gamma_1 = theta["gamma_0"], theta["gamma_1"]
        gamma_0_indep = gamma_0[:p]
        gamma_0_dep = gamma_0[p:]
        gamma_0_pen = self.ENet.pen(gamma_0_indep)
        groups = np.arange(0, len(gamma_0) - p).reshape(L, -1).tolist()
        SGL1 = SparseGroupL1(l_pen, eta, groups)
        gamma_0_pen += SGL1.pen(gamma_0_dep)
        gamma_1_indep = gamma_1[:p]
        gamma_1_dep = gamma_1[p:]
        gamma_1_pen = self.ENet.pen(gamma_1_indep)
        gamma_1_pen += SGL1.pen(gamma_1_dep)
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

    def f_data_given_latent(self, X, extracted_features, T, T_u, delta, S):
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

        Returns
        -------
        f : `np.ndarray`, shape=(n_samples, K, N_MC)
            The value of the f(Y, T, delta| S, G ; theta)
        """
        theta, alpha = self.theta, self.fixed_effect_time_order
        baseline_hazard, phi = theta["baseline_hazard"], theta["phi"]
        E_func = EstepFunctions(X, T, T_u, delta, extracted_features, alpha,
                                self.asso_functions, theta)
        beta_0, beta_1 = theta["beta_0"], theta["beta_1"]
        gamma_0, gamma_1 = theta["gamma_0"], theta["gamma_1"]
        g1 = E_func.g1(S, gamma_0, beta_0, gamma_1, beta_1, False)
        g3 = E_func.g3(S, beta_0, beta_1)
        baseline_val = baseline_hazard.values.flatten()
        rel_risk = g1.swapaxes(0, 2) * baseline_val
        _, ind_1, ind_2 = get_times_infos(T, T_u)
        intensity = self.intensity(rel_risk, ind_1)
        survival = self.survival(rel_risk, ind_2)
        f_y = self.f_y_given_latent(extracted_features, g3)
        f = (intensity ** delta).T * survival * f_y
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
                                         delta_prediction, self.S)
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
                       "beta_0", "beta_1", "gamma_0", "gamma_1"]:
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
        nb_asso_feat = L * nb_asso_param + p
        N = 10  # Number of initial Monte Carlo sample for S

        X = normalize(X)  # Normalize time-independent features
        ext_feat = extract_features(Y, alpha)  # Features extraction
        T_u = np.unique(T)
        self.T_u = T_u
        J, ind_1, ind_2 = get_times_infos(T, T_u)

        # Initialization
        # TODO: for debugging and update hyper-params if not useful
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

        # TODO: for debugging and update hyper-params if not useful
        gamma_0 = 1e-4 * np.ones(nb_asso_feat)
        gamma_0[:p] = time_indep_cox_coeffs
        gamma_0 = gamma_0.reshape(-1, 1)
        gamma_1 = gamma_0.copy()

        beta_0 = beta.reshape(-1, 1)
        beta_1 = beta_0.copy()

        self._update_theta(beta_0=beta_0, beta_1=beta_1, xi=xi_ext,
                           gamma_0=gamma_0, gamma_1=gamma_1, long_cov=D,
                           phi=phi, baseline_hazard=baseline_hazard)

        # Stopping criteria and bounds vector for the L-BGFS-B algorithm
        maxiter, pgtol = 60, 1e-5
        bounds_xi = [(0, None)] * 2 * p

        # Instanciates E-step and M-step functions
        E_func = EstepFunctions(X, T, T_u, delta, ext_feat, alpha,
                                asso_functions, self.theta)
        F_func = MstepFunctions(fit_intercept, X, T, delta, L, p, self.l_pen,
                                self.eta_elastic_net, nb_asso_feat, alpha,
                                asso_functions)

        S = E_func.construct_MC_samples(N)
        f = self.f_data_given_latent(X, ext_feat, T, self.T_u, delta, S)
        Lambda_1 = E_func.Lambda_g(np.ones(shape=(n_samples, 2, 2 * N)), f)
        pi_xi = self._get_proba(X)
        obj = self._func_obj(pi_xi, f)

        # Store init values
        self.history.update(n_iter=0, obj=obj, rel_obj=np.inf, theta=self.theta)
        if verbose:
            self.history.print_history()

        for n_iter in range(1, max_iter + 1):

            # E-Step
            pi_est = self._get_post_proba(pi_xi, Lambda_1)
            E_g4 = E_func.Eg(E_func.g4(S), Lambda_1, pi_xi, f)
            E_g5 = E_func.Eg(E_func.g5(S), Lambda_1, pi_xi, f)

            def E_g1(gamma_0_, beta_0_, gamma_1_, beta_1_):
                return E_func.Eg(
                    E_func.g1(S, gamma_0_, beta_0_, gamma_1_, beta_1_),
                    Lambda_1, pi_xi, f)

            def E_log_g1(gamma_0_, beta_0_, gamma_1_, beta_1_):
                return E_func.Eg(
                    np.log(E_func.g1(S, gamma_0_, beta_0_, gamma_1_, beta_1_)),
                    Lambda_1, pi_xi, f)

            def E_g6(gamma_0_, beta_0_, gamma_1_, beta_1_):
                return E_func.Eg(
                    E_func.g6(S, gamma_0_, beta_0_, gamma_1_, beta_1_),
                    Lambda_1, pi_xi, f)

            if False:  # TODO: condition to be defined
                N *= 1.1
                fctr *= .1

            # M-Step
            D = E_g4.sum(axis=0) / n_samples  # D update

            if warm_start:
                xi_init = xi_ext
                beta_init = [beta_0.flatten(), beta_1.flatten()]
                gamma_init = [gamma_0.flatten(), gamma_1.flatten()]
            else:
                xi_init = np.zeros(2 * p)
                beta_init = [np.zeros(L * q_l),
                             np.zeros(L * q_l)]
                gamma_init = [np.zeros(nb_asso_feat),
                              np.zeros(nb_asso_feat)]

            # xi update
            xi_ext = fmin_l_bfgs_b(
                func=lambda xi_ext_: F_func.P_pen_func(pi_est, xi_ext_),
                x0=xi_init,
                fprime=lambda xi_ext_: F_func.grad_P_pen(pi_est, xi_ext_),
                disp=False, bounds=bounds_xi, maxiter=maxiter, pgtol=pgtol)[0]

            # beta_0 update
            eta_sp_gp_l1, l_pen = self.eta_sp_gp_l1, self.l_pen
            pi_est_K = np.vstack((1 - pi_est, pi_est))
            gamma_K = [gamma_0, gamma_1]
            groups = np.arange(0, len(beta_0)).reshape(L, -1).tolist()
            prox = SparseGroupL1(l_pen, eta_sp_gp_l1, groups).prox
            args_all = {"pi_est": pi_est_K, "E_g5": E_g5, "E_g4": E_g4,
                        "gamma": gamma_K, "baseline_hazard": baseline_hazard,
                        "extracted_features": ext_feat, "phi": phi,
                        "ind_1": ind_1, "ind_2": ind_2}
            args_0 = {"E_g1": lambda v: E_g1(gamma_0, v, gamma_1, beta_1),
                      "group": 0}
            beta_0_prev = beta_0.copy()
            copt_max_iter = 100
            beta_0 = copt.minimize_proximal_gradient(
                fun=F_func.R_func, x0=beta_init[0], prox=prox, max_iter=copt_max_iter,
                args=[{**args_all, **args_0}], jac=F_func.grad_R, step="backtracking",
                accelerated=True).x.reshape(-1, 1)

            # beta_1 update
            args_1 = {"E_g1": lambda v: E_g1(gamma_0, beta_0_prev, gamma_1, v),
                      "group": 1}
            beta_1 = copt.minimize_proximal_gradient(
                fun=F_func.R_func, x0=beta_init[1], prox=prox, max_iter=copt_max_iter,
                args=[{**args_all, **args_1}], jac=F_func.grad_R,  step="backtracking",
                accelerated=True).x.reshape(-1, 1)

            # gamma_0 update
            beta_K = [beta_0, beta_1]
            groups = np.arange(0, len(gamma_0) - p).reshape(L, -1).tolist()
            prox = SparseGroupL1(l_pen, eta_sp_gp_l1, groups).prox
            args_all = {"pi_est": pi_est_K, "E_g5": E_g5,
                        "phi": phi, "beta": beta_K,
                        "baseline_hazard": baseline_hazard,
                        "extracted_features": ext_feat,
                        "ind_1": ind_1, "ind_2": ind_2}
            args_0 = {"E_g1": lambda v: E_g1(v, beta_0, gamma_1, beta_1),
                      "E_log_g1": lambda v: E_log_g1(v, beta_0, gamma_1, beta_1),
                      "E_g6": lambda v: E_g6(v, beta_0, gamma_1, beta_1),
                      "group": 0}
            gamma_0_prev = gamma_0.copy()
            gamma_0 = copt.minimize_proximal_gradient(
                fun=F_func.Q_func, x0=gamma_init[0], prox=prox, max_iter=copt_max_iter,
                args=[{**args_all, **args_0}], jac=F_func.grad_Q, step="backtracking",
                accelerated=True).x.reshape(-1, 1)

            # gamma_1 update
            args_1 = {"E_g1": lambda v: E_g1(gamma_0_prev, beta_0, v, beta_1),
                      "E_log_g1": lambda v: E_log_g1(gamma_0_prev, beta_0, v, beta_1),
                      "E_g6": lambda v: E_g6(gamma_0_prev, beta_0, v, beta_1),
                      "group": 1}
            gamma_1 = copt.minimize_proximal_gradient(
                fun=F_func.Q_func, x0=gamma_init[1], prox=prox, max_iter=copt_max_iter,
                args=[{**args_all, **args_1}], jac=F_func.grad_Q, step="backtracking",
                accelerated=True).x.reshape(-1, 1)

            # beta, gamma needs to be updated before the baseline
            self._update_theta(beta_0 = beta_0, beta_1 = beta_1,
                               gamma_0 = gamma_0, gamma_1 = gamma_1)
            E_func.theta = self.theta
            E_g1 = E_func.Eg(E_func.g1(S, gamma_0, beta_0, gamma_1, beta_1),
                             Lambda_1, pi_xi, f)

            # baseline hazard update
            baseline_hazard = pd.Series(
                data=((ind_1 * 1).T * delta).sum(axis=1) /
                     ((E_g1.T * (ind_2 * 1).T).swapaxes(0, 1) * pi_est_K)
                         .sum(axis=2).sum(axis=1), index=T_u)

            # phi update
            (U_L, V_L, y_L, N_L) = ext_feat[1]
            for l in range(L):
                pi_est_ = np.concatenate([[pi_est[i]] * N_L[l][i]
                                          for i in range(n_samples)])
                pi_est_ = np.vstack((1 - pi_est_, pi_est_)).T  # K = 2
                N_l, y_l, U_l, V_l = sum(N_L[l]), y_L[l], U_L[l], V_L[l]
                beta_l = beta[q_l * l: q_l * (l + 1)]
                E_g5_l = E_g5.reshape(n_samples, L, q_l)[:, l].reshape(-1, 1)
                E_g4_l = block_diag(E_g4[:, r_l * l: r_l * (l + 1),
                                    r_l * l: r_l * (l + 1)])
                tmp = y_l - U_l * beta_l.flatten()
                phi_l = (pi_est_ * (tmp * (tmp - 2 * (V_l.dot(E_g5_l))))).sum() \
                        + np.trace(V_l.T.dot(V_l).dot(E_g4_l))
                phi[l] = phi_l / N_l

            self._update_theta(phi=phi, baseline_hazard=baseline_hazard,
                               long_cov=D, xi=xi_ext)
            pi_xi = self._get_proba(X)
            E_func.theta = self.theta
            S = E_func.construct_MC_samples(N)
            f = self.f_data_given_latent(X, ext_feat, T, T_u, delta, S)

            prev_obj = obj
            obj = self._func_obj(pi_xi, f)
            rel_obj = abs(obj - prev_obj) / abs(prev_obj)
            if n_iter % print_every == 0:
                self.history.update(n_iter=n_iter, obj=obj, rel_obj=rel_obj,
                                    theta=self.theta)
                if verbose:
                    self.history.print_history()
            if (n_iter > max_iter) or (rel_obj < tol):
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
            return c_index_score(T, self.predict_marker(X, Y), delta)
        else:
            raise ValueError('You must fit the model first')
