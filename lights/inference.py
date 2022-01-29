import numpy as np
import pandas as pd
import copt
import warnings
from scipy.optimize import fmin_l_bfgs_b
from numpy.linalg import multi_dot
from lifelines.utils import concordance_index as c_index_score
from lights.base.base import Learner, extract_features, normalize, block_diag, \
    get_xi_from_xi_ext, logistic_grad, get_times_infos
from lights.init.mlmm import MLMM
from lights.init.cox import initialize_baseline_hazard
from lights.model.e_step_functions import EstepFunctions
from lights.model.m_step_functions import MstepFunctions
from lights.model.regularizations import ElasticNet, SparseGroupL1
from scipy.stats import multivariate_normal


class prox_QNMCEM(Learner):
    """prox-QNMCEM Algorithm for the lights model inference

    Parameters
    ----------
    fit_intercept : `bool`, default=True
        If `True`, include an intercept in the model for the time independent
        features

    l_pen_EN : `float`, default=0.
        Level of penalization for the ElasticNet

    l_pen_SGL : `float`, default=0.
        Level of penalization for the Sparse Group l1 on gamma

    eta_elastic_net: `float`, default=0.1
        The ElasticNet mixing parameter, with 0 <= eta_elastic_net <= 1.
        For eta_elastic_net = 1 this is ridge (L2) regularization
        For eta_elastic_net = 0 this is lasso (L1) regularization
        For 0 < eta_elastic_net < 1, the regularization is a linear combination
        of L1 and L2

    eta_sp_gp_l1: `float`, default=0.1
        The Sparse Group L1 mixing parameter, with 0 <= eta_sp_gp_l1 <= 1
        For eta_sp_gp_l1 = 1 this is Group L1

    max_iter: `int`, default=100
        Maximum number of iterations of the prox-QNMCEM algorithm

    max_iter_lbfgs: `int`, default=50
        Maximum number of iterations of the L-BFGS-B solver

    max_iter_proxg: `int`, default=10
        Maximum number of iterations of the proximal gradient solver

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

    n_MC : `int`, default=50
        Number of Monte Carlo sample used in the E-step

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
        If `True`, we compute the global objective to be minimized by the prox-QNMCEM
         algorithm and store it in history

    copt_solver_step : function or `str`='backtracking', default='backtracking'
        Step size for optimization algorithm used in Copt colver

    simu : `bool`, defaut=True
        If `True` we do the inference with simulated data.

    S_k : `list`
        Set of nonactive group for 2 classes (will be useful in case of
        simulated data).

    cov_corr_rdn_long : `float`
        Correlation coefficient of the toeplitz correlation matrix of
        random longitudinal features (will be useful in case of
        simulated data).

    """

    def __init__(self, fit_intercept=False, l_pen_EN=0., l_pen_SGL=0.,
                 eta_elastic_net=.1, eta_sp_gp_l1=.1,
                 max_iter=100, max_iter_lbfgs=50, max_iter_proxg=10,
                 verbose=True, print_every=10, tol=1e-5,
                 warm_start=True, fixed_effect_time_order=5, n_MC=50,
                 asso_functions='all', initialize=True, copt_accelerate=False,
                 compute_obj=False, copt_solver_step='backtracking', simu=True,
                 S_k=None, cov_corr_rdn_long=.05):
        Learner.__init__(self, verbose=verbose, print_every=print_every)
        self.max_iter = max_iter
        self.max_iter_lbfgs = max_iter_lbfgs
        self.max_iter_proxg = max_iter_proxg
        self.tol = tol
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept
        self.fixed_effect_time_order = fixed_effect_time_order
        self.asso_functions = asso_functions
        self.initialize = initialize
        self.copt_accelerate = copt_accelerate
        self.l_pen_EN = l_pen_EN
        self.l_pen_SGL = l_pen_SGL
        self.eta_elastic_net = eta_elastic_net
        self.eta_sp_gp_l1 = eta_sp_gp_l1
        self.n_MC = n_MC
        self.compute_obj = compute_obj
        self.ENet = ElasticNet(l_pen_EN, eta_elastic_net)
        self._fitted = False
        self.simu = simu
        self.S_k = S_k
        self.cov_corr_rdn_long = cov_corr_rdn_long

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

        prev_theta : `dictionary`
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
    def _log_lik(pi_xi, f):
        """Computes the approximation of the likelihood of the lights model

        Parameters
        ----------
        pi_xi : `np.ndarray`, shape=(n_samples,)
            Probability estimates for being on the high-risk group given
            time-independent features

        f : `np.ndarray`, shape=(n_samples, K)
            The value of f(Y, T, delta| b, G, asso_feats ; theta)

        Returns
        -------
        prb : `float`
            The approximated log-likelihood computed on the given data
        """
        pi_xi_ = np.vstack((1 - pi_xi, pi_xi)).T
        prb = np.log((pi_xi_ * f).sum(axis=-1)).mean()
        return prb

    def _func_obj(self, pi_xi, f):
        """The global objective to be minimized by the prox-QNMCEM algorithm
        (including penalization)

        Parameters
        ----------
        pi_xi : `np.ndarray`, shape=(n_samples,)
            Probability estimates for being on the high-risk group given
            time-independent features

        f : `np.ndarray`, shape=(n_samples, K)
            The value of f(Y, T, delta|b, G, asso_feats ; theta)

        Returns
        -------
        output : `float`
            The value of the global objective to be minimized
        """
        p, L = self.n_time_indep_features, self.n_long_features
        eta_sp_gp_l1 = self.eta_sp_gp_l1
        l_pen_SGL = self.l_pen_SGL
        theta = self.theta
        log_lik = self._log_lik(pi_xi, f)
        # xi elastic net penalty
        xi = theta["xi"]
        xi_pen = self.ENet.pen(xi)

        # gamma sparse group l1 penalty
        gamma_0, gamma_1 = theta["gamma_0"], theta["gamma_1"]
        groups = np.arange(0, len(gamma_0)).reshape(L, -1).tolist()
        SGL1 = SparseGroupL1(l_pen_SGL, eta_sp_gp_l1, groups)
        gamma_0_pen = SGL1.pen(gamma_0)
        gamma_1_pen = SGL1.pen(gamma_1)
        pen = xi_pen + gamma_0_pen + gamma_1_pen

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

    def f_y_given_latent(self, extracted_features, beta):
        """Computes the density of the longitudinal processes given latent
        variables

        Parameters
        ----------
        extracted_features :  `tuple, tuple`,
            The extracted features from longitudinal data.
            Each tuple is a combination of fixed-effect design features,
            random-effect design features, outcomes, number of the longitudinal
            measurements for all subject or arranged by l-th order.

        beta : `list`
            list of fixed effect parameters

        Returns
        -------
        f_y : `np.ndarray`, shape=(n_samples, K)
            The value of the f(Y | G ; theta)
        """
        (U_list, V_list, y_list, N_list) = extracted_features[0]
        n_samples, n_long_features = len(y_list), self.n_long_features
        phi = self.theta["phi"]
        D = self.theta["long_cov"]
        K = 2  # 2 latent groups
        f_y = np.ones(shape=(n_samples, K))
        for i in range(n_samples):
            U_i, V_i, y_i, n_i = U_list[i], V_list[i], \
                                 np.array(y_list[i]).flatten(), N_list[i]
            inv_Phi_i = []
            for l in range(n_long_features):
                inv_Phi_i += [phi[l, 0]] * N_list[i][l]
            cov = np.diag(inv_Phi_i) + multi_dot([V_i, D, V_i.T])
            for k in range(K):
                mean = U_i.dot(beta[k].flatten())
                f_y[i, k] = multivariate_normal.pdf(y_i, mean, cov)
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

    def f_data_given_latent(self, extracted_features, asso_feats, T, T_u, delta):
        """Estimates the data density given latent variables

        Parameters
        ----------
        extracted_features :  `tuple, tuple`,
            The extracted features from longitudinal data.
            Each tuple is a combination of fixed-effect design features,
            random-effect design features, outcomes, number of the longitudinal
            measurements for all subject or arranged by l-th order.

        asso_feats : `np.ndarray`, shape=(n_samples, n_asso_params)
            Association features extracted from tsfresh

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
        f : `np.ndarray`, shape=(n_samples, K)
            The value of the f(Y, T, delta| asso_feats, b, G ; theta)
        """
        theta, alpha = self.theta, self.fixed_effect_time_order
        L = self.n_long_features
        baseline_hazard, phi = theta["baseline_hazard"], theta["phi"]
        E_func = EstepFunctions(T_u, L, alpha, self.asso_functions, theta)
        beta_0, beta_1 = theta["beta_0"], theta["beta_1"]
        gamma_0, gamma_1 = theta["gamma_0"], theta["gamma_1"]
        gamma_stack = np.hstack((gamma_0, gamma_1))
        tmp = np.exp(asso_feats.dot(gamma_stack))
        baseline_val = baseline_hazard.values.flatten()
        _, ind_1, ind_2 = get_times_infos(T, T_u)
        intensity = tmp.T * (ind_1.dot(baseline_val))
        survival = np.exp(-tmp.T * (ind_2.dot(baseline_val)))
        f = ((intensity ** delta) * survival).T
        f_y = self.f_y_given_latent(extracted_features, [beta_0, beta_1])
        f *= f_y
        return f

    def predict_marker(self, X, Y, asso_feats, prediction_times=None):
        """Marker rule of the lights model for being on the high-risk group

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The longitudinal data. Each element of the dataframe is
            a pandas.Series

        asso_feats : `np.ndarray`, shape=(n_samples, n_asso_params)
            Association features extracted from tsfresh

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
            n_samples, n_long_features = Y.shape
            last_measurement = np.zeros(n_samples)
            for i in range(n_samples):
                t_i_max = 0
                Y_i = Y.iloc[i]
                for l in range(n_long_features):
                    times_il = Y_i[l].index.values
                    t_i_max = max(t_i_max, np.array(times_il).max())
                last_measurement[i] = t_i_max
            if prediction_times is None:
                prediction_times = last_measurement
            else:
                if not (prediction_times >= last_measurement).all():
                    raise ValueError('Prediction times must be greater than the'
                                     ' last measurement times for each subject')

            # predictions for alive subjects only
            delta_prediction = np.zeros(n_samples)
            T_u = self.T_u
            f = self.f_data_given_latent(ext_feat, asso_feats, prediction_times
                                         , T_u, delta_prediction)
            pi_xi = self._get_proba(X)
            marker = self._get_post_proba(pi_xi, f)
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

    def fit(self, X, Y, T, delta, asso_feats):
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
        K = 2
        if fit_intercept:
            p += 1

        nb_asso_param = asso_feats.shape[1] // L
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
            baseline_hazard = initialize_baseline_hazard(X, T, delta)
        else:
            # Fixed initialization
            q = q_l * L
            r = r_l * L
            beta = np.zeros((q, 1))
            D = np.diag(np.ones(r))
            phi = np.ones((L, 1))
            baseline_hazard = pd.Series(data=.5 * np.ones(J), index=T_u)

        #TODO: just for testing, remove later
        phi = np.ones(L).reshape(-1, 1)
        D = .01 * np.diag(np.ones(r_l * L))
        beta_0 = beta.reshape(-1, 1)
        beta_1 = beta_0.copy()
        gamma_0 = 1e-4 * np.ones((L * nb_asso_param, 1))
        gamma_1 = gamma_0.copy()
        self._update_theta(beta_0=beta_0, beta_1=beta_1, xi=xi_ext,
                           gamma_0=gamma_0, gamma_1=gamma_1, long_cov=D,
                           phi=phi, baseline_hazard=baseline_hazard)

        # Stopping criteria and bounds vector for the optim algorithm
        max_iter_lbfgs, pgtol = self.max_iter_lbfgs, self.tol
        bounds_xi = [(0, None)] * 2 * p
        max_iter_proxg = self.max_iter_proxg

        # Instanciates E-step and M-step functions
        E_func = EstepFunctions(T_u, L, alpha, asso_feats, self.theta)
        E_func.b_stats(ext_feat)
        F_func = MstepFunctions(fit_intercept, X, delta, p, self.l_pen_EN,
                                self.eta_elastic_net)

        f = self.f_data_given_latent(ext_feat, asso_feats, T, self.T_u, delta)
        Lambda_1 = E_func.Lambda_g(np.ones((n_samples, K)), f)
        pi_xi = self._get_proba(X)

        # Store init values
        if self.compute_obj:
            obj = self._func_obj(pi_xi, f)
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
            self.pi_est = pi_est
            E_g1 = E_func.Eg(E_func.Eb, Lambda_1, pi_xi, f)
            E_g2 = E_func.Eg(E_func.EbbT, Lambda_1, pi_xi, f)

            # M-Step
            D = E_g2.sum(axis=0) / n_samples  # D update

            if warm_start:
                xi_init = xi_ext
                if self.simu:
                    gamma_init = [gamma_0.flatten(), gamma_1.flatten()]
                else:
                    gamma_init = [1e-4 * np.ones((L * (nb_asso_param))),
                                  1e-4 * np.ones((L * (nb_asso_param)))]
            else:
                xi_init = np.zeros(2 * p)
                gamma_init = [np.zeros(L * nb_asso_param),
                              np.zeros(L * nb_asso_param)]

            # xi update
            xi_ext = fmin_l_bfgs_b(
                func=lambda xi_ext_: F_func.P_pen_func(pi_est, xi_ext_),
                x0=xi_init,
                fprime=lambda xi_ext_: F_func.grad_P_pen(pi_est, xi_ext_),
                disp=False, bounds=bounds_xi, maxiter=max_iter_lbfgs,
                pgtol=pgtol)[0]

            # beta update
            K = 2
            pi_est_K = np.vstack((1 - pi_est, pi_est))
            (U_list, V_list, y_list, _) = ext_feat[0]
            num = np.zeros((K, L * q_l))
            den = np.zeros((K, L * q_l, L * q_l))
            for i in range(n_samples):
                U_i, V_i, y_i = U_list[i], V_list[i], y_list[i]
                tmp_num = U_i.T.dot((y_i.flatten() - V_i.dot(E_g1[i])))
                tmp_den = U_i.T.dot(U_i)
                for k in range(K):
                    num[k] += pi_est_K[k, i] * tmp_num
                    den[k] += pi_est_K[k, i] * tmp_den

            beta_0 = np.linalg.inv(den[0]).dot(num[0]).reshape(-1, 1)
            beta_1 = np.linalg.inv(den[1]).dot(num[1]).reshape(-1, 1)
            self._update_theta(beta_0=beta_0, beta_1=beta_1)

            # gamma_0 update
            beta_K = [beta_0, beta_1]
            gamma = [gamma_0, gamma_1]
            groups = np.arange(0, len(gamma_0)).reshape(L, -1).tolist()
            eta_sp_gp_l1 = self.eta_sp_gp_l1
            l_pen_SGL = self.l_pen_SGL
            prox = SparseGroupL1(l_pen_SGL, eta_sp_gp_l1, groups).prox
            args_all = {"pi_est": pi_est_K, "E_g1": E_g1,
                        "phi": phi, "beta": beta_K,
                        "baseline_hazard": baseline_hazard,
                        "extracted_features": ext_feat,
                        "ind_1": ind_1, "ind_2": ind_2, "gamma": gamma,
                        "asso_feats": asso_feats}

            args_0 = {"group": 0}
            res0 = copt.minimize_proximal_gradient(
                fun=F_func.Q_func, x0=gamma_init[0], prox=prox,
                max_iter=max_iter_proxg,
                args=[{**args_all, **args_0}], jac=F_func.grad_Q,
                step=self.copt_step,
                accelerated=self.copt_accelerate)
            gamma_0 = res0.x.reshape(-1, 1)

            # gamma_1 update
            args_1 = {"group": 1}
            res1 = (copt.minimize_proximal_gradient(
                fun=F_func.Q_func, x0=gamma_init[1], prox=prox,
                max_iter=max_iter_proxg,
                args=[{**args_all, **args_1}], jac=F_func.grad_Q,
                step=self.copt_step,
                accelerated=self.copt_accelerate))
            gamma_1 = res1.x.reshape(-1, 1)

            # beta, gamma needs to be updated before the baseline
            self._update_theta(gamma_0=gamma_0, gamma_1=gamma_1)
            E_func.theta = self.theta
            E_g4 = E_func.Eg(E_func.g4(gamma_0, gamma_1), Lambda_1, pi_xi, f)

            # baseline hazard update
            baseline_hazard = pd.Series(
                data=  (((ind_1.T * delta).sum(axis=1)) /
                      ((E_g4.T * ind_2.T).swapaxes(0, 1) * pi_est_K)
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
                E_g1_l = E_g1.reshape((n_samples, L, r_l))[:, l].reshape(-1, 1)
                E_g2_l = block_diag(E_g2[:, r_l * l: r_l * (l + 1),
                                    r_l * l: r_l * (l + 1)])
                tmp = y_l - U_l.dot(beta_l)
                phi_l = (pi_est_stack * (
                        tmp * (tmp - 2 * (V_l.dot(E_g1_l))))).sum() \
                        + np.trace(V_l.T.dot(V_l).dot(E_g2_l))
                phi[l] = phi_l / N_l

            self._update_theta(phi=phi, baseline_hazard=baseline_hazard,
                               long_cov=D, xi=xi_ext)
            pi_xi = self._get_proba(X)
            E_func.theta = self.theta
            S = E_func.construct_MC_samples(N)
            f = self.f_data_given_latent(ext_feat, T, T_u, delta, S)

            rel_theta = self._rel_theta(self.theta, prev_theta, 1e-2)
            prev_theta = self.theta.copy()
            if n_iter % print_every == 0:
                if self.compute_obj:
                    prev_obj = obj
                    f_mean = f.mean(axis=-1)
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
                break
            else:
                # Update for next iteration
                Lambda_1 = E_func.Lambda_g(np.ones((n_samples, K)), f)

        self._end_solve()

    def score(self, X, Y, T, delta, asso_feats):
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
            c_index = c_index_score(T, self.predict_marker(X, Y, asso_feats), delta)
            return max(c_index, 1 - c_index)
        else:
            raise ValueError('You must fit the model first')
