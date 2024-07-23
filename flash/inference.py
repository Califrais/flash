import numpy as np
import pandas as pd
import copt
import warnings
from scipy.optimize import fmin_l_bfgs_b
from lifelines.utils import concordance_index as c_index_score
from flash.base.base import Learner, extract_features, normalize, block_diag, \
    get_xi_from_xi_ext, logistic_grad, get_times_infos
from flash.init.mlmm import MLMM
from flash.init.cox import initialize_baseline_hazard
from flash.model.e_step_functions import EstepFunctions
from flash.model.m_step_functions import MstepFunctions
from flash.model.regularizations import ElasticNet, SparseGroupL1
from scipy.stats import multivariate_normal
from numpy.linalg import multi_dot
from flash.base.base import feat_representation_extraction, feat_representation_extraction_sig

class ext_EM(Learner):
    """Extension of EM Algorithm for the flash model inference

    Parameters
    ----------

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

    eta_sp_gp_l1: `float`, default=0.9
        The Sparse Group L1 mixing parameter, with 0 <= eta_sp_gp_l1 <= 1
        For eta_sp_gp_l1 = 1 this is Group L1

    max_iter: `int`, default=100
        Maximum number of iterations of the prox-QNEM algorithm

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

    fixed_effect_time_order : `int`, default=1
        Order of the higher time monomial considered for the representations of
        the time-varying features corresponding to the fixed effect. The
        dimension of the corresponding design matrix is then equal to
        fixed_effect_time_order + 1

    initialize : `bool`, default=True
        If `True`, we initialize the parameters using MLMM model, otherwise we
        use arbitrarily chosen fixed initialization

    copt_accelerate : `bool`, default=False
        If `True`, we choose copt solver with accelerated proximal
        gradient (FISTA), otherwise we use regular ISTA

    copt_solver_step : function or `str`='backtracking', default='backtracking'
        Step size for optimization algorithm used in Copt colver

    """

    def __init__(self, l_pen_EN=0., l_pen_SGL=0.,
                 eta_elastic_net=.1, eta_sp_gp_l1=.9,
                 max_iter=40, max_iter_lbfgs=50, max_iter_proxg=10,
                 verbose=True, print_every=10, tol=1e-4,
                 warm_start=True, fixed_effect_time_order=1, initialize=True,
                 copt_accelerate=False, copt_solver_step='backtracking',
                 fc_parameters=None, sig_ext=False, sig_order=3):
        Learner.__init__(self, verbose=verbose, print_every=print_every)
        self.max_iter = max_iter
        self.max_iter_lbfgs = max_iter_lbfgs
        self.max_iter_proxg = max_iter_proxg
        self.tol = tol
        self.warm_start = warm_start
        self.fixed_effect_time_order = fixed_effect_time_order
        self.initialize = initialize
        self.copt_accelerate = copt_accelerate
        self.l_pen_EN = l_pen_EN
        self.l_pen_SGL = l_pen_SGL
        self.eta_elastic_net = eta_elastic_net
        self.sig_ext = sig_ext
        if self.sig_ext:
            self.eta_sp_gp_l1 = 0.
            self.sig_order = sig_order
        else:
            self.eta_sp_gp_l1 = eta_sp_gp_l1
        self.ENet = ElasticNet(l_pen_EN, eta_elastic_net)
        self._fitted = False

        # Attributes that will be instantiated afterwards
        self.n_samples = None
        self.n_time_indep_features = None
        self.n_long_features = None
        self.T_u = None
        self.theta = {
            "beta": np.empty(1),
            "long_cov": np.empty(1),
            "phi": np.empty(1),
            "xi": np.empty(1),
            "baseline_hazard": pd.Series(),
            "gamma": np.empty(1)
        }
        self.copt_step = copt_solver_step
        self.fc_parameters = fc_parameters

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
    def _log_lik(pi_xi, f):
        """Computes the the likelihood of the flash model

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
            The log-likelihood computed on the given data
        """
        prb = np.log((pi_xi * f).sum(axis=-1)).mean()
        return prb

    def _func_obj(self, pi_xi, f):
        """The global objective to be minimized by the prox-QNEM algorithm
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
        gamma = theta["gamma"]
        if self.sig_ext:
            groups = np.arange(0, len(gamma[0])).reshape(1, -1).tolist()
        else:
            groups = np.arange(0, len(gamma[0])).reshape(L, -1).tolist()
        SGL1 = SparseGroupL1(l_pen_SGL, eta_sp_gp_l1, groups)
        K = xi.shape[1]
        pen = 0
        for k in range(K):
            pen += self.ENet.pen(xi[:, k])
            pen += SGL1.pen(gamma[k])

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
        u = X.dot(self.theta["xi"])

        #return logistic_grad(u)
        return (np.exp(u).T / np.exp(u).sum(axis=1)).T

    @staticmethod
    def _get_post_proba(pi_xi, Lambda_1):
        """Posterior probability estimates for being on the high-risk group
        given all observed data

        Parameters
        ----------
        pi_xi : `np.ndarray`, shape=(n_samples, K)
            Comes from get_proba function

        Lambda_1 : `np.ndarray`, shape=(n_samples, K)
            Computed integral (see (15) in the flash paper) with
            \tilde(g)=1

        Returns
        -------
        pi_est : `np.ndarray`, shape=(n_samples, K)
            Returns the posterior probability of the sample for being on the
            high-risk group given all observed data
        """
        tmp = Lambda_1 * pi_xi
        pi_est = (tmp.T / tmp.sum(axis=1)).T
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
        K = len(beta)  # nb latent groups
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

        Returns
        -------
        f : `np.ndarray`, shape=(n_samples, K)
            The value of the f(Y, T, delta| asso_feats, b, G ; theta)
        """
        theta, alpha = self.theta, self.fixed_effect_time_order
        baseline_hazard, phi = theta["baseline_hazard"], theta["phi"]
        beta_all = theta["beta"]
        gamma_all = theta["gamma"]
        gamma_stack = np.hstack(gamma_all)
        tmp = np.exp(asso_feats.dot(gamma_stack))
        baseline_val = baseline_hazard.values.flatten()
        _, ind_1, ind_2 = get_times_infos(T, T_u)
        intensity = (tmp.T * ind_1.T).swapaxes(1, -1).dot(baseline_val)
        survival = np.exp(-(tmp.T * ind_2.T).swapaxes(1, -1).dot(baseline_val))
        f = ((intensity ** delta) * survival).T
        f_y = self.f_y_given_latent(extracted_features, beta_all)
        f *= f_y
        return f

    def predict_marker(self, X, Y, asso_feats=None, prediction_times=None):
        """Marker rule of the flash model for being on the high-risk group

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The longitudinal data. Each element of the dataframe is
            a pandas.Series

        asso_feats : `np.ndarray`, shape=(n_samples, n_asso_params)
            Extracted association features

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
            theta, alpha = self.theta, self.fixed_effect_time_order
            asso_feats = asso_feats
            if asso_feats is None:
                asso_feats, _ = feat_representation_extraction(Y, self.n_long_features,
                                                            self.T_u, self.fc_parameters)

            ext_feat = extract_features(Y, self.time_dep_feats, alpha)
            id_list = list(np.unique(Y.id.values))
            n_samples = len(id_list)
            last_measurement = np.zeros(n_samples)
            for i in range(n_samples):
                Y_i  = Y[(Y["id"] == id_list[i])]
                t_i_max = np.array(Y_i["T_long"].values).max()
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

    def predict_marker_sample(self, X, Y):
        """Marker rule of the flash model for being on the high-risk group

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
            The time-independent features matrix

        Y : `pandas.DataFrame`, shape=(n_samples, n_long_features)
            The longitudinal data. Each element of the dataframe is
            a pandas.Series

        Y_rep : `pandas.DataFrame`, shape=(n_samples, _)
            The longitudinal data in the format to be used later for extracting
            representation features.

        Returns
        -------
        marker : `np.ndarray`, shape=(n_samples,)
            Returns the marker rule of the sample for being on the high-risk
            group
        """
        if self._fitted:
            theta, alpha = self.theta, self.fixed_effect_time_order
            id_list = list(np.unique(Y.id.values))
            n_samples = len(id_list)
            markers = []
            if self.sig_ext:
                asso_feats, _ = feat_representation_extraction_sig(Y, self.T_u, self.sig_order)
            else:
                asso_feats, _ = feat_representation_extraction(Y, self.n_long_features,
                                                               self.T_u, self.fc_parameters)
            for i in range(n_samples):
                # predictions for alive subjects only
                T_u = self.T_u
                pi_xi = self._get_proba(X[i:(i + 1)])
                Y_i = Y[(Y["id"] == id_list[i])]
                time_measurement = Y_i["T_long"].values
                marker = np.zeros(len(time_measurement))
                for t in range(len(time_measurement)):
                    ext_feat = extract_features(Y_i, self.time_dep_feats,
                                                alpha, time_measurement[t])
                    f = self.f_data_given_latent(ext_feat,
                                                 asso_feats[i:(i + 1)],
                                                 time_measurement[t], T_u, 0)
                    marker[t] = self._get_post_proba(pi_xi, f)[:, 1]

                markers.append(marker)

            return markers
        else:
            raise ValueError('You must fit the model first')

    def _update_theta(self, **kwargs):
        """Update class attributes corresponding to flash model parameters
        """
        for key, value in kwargs.items():
            if key in ["long_cov", "phi", "baseline_hazard"]:
                self.theta[key] = value
            elif key == "beta":
                self.theta[key] = value
                K = len(value)
                for k in range(K):
                    self.theta["beta_" + str(k)] = value[k]
            elif key == "gamma":
                self.theta[key] = value
                K = len(value)
                for k in range(K):
                    self.theta["gamma_" + str(k)] = value[k]
            elif key == "xi":
                xi = get_xi_from_xi_ext(value)
                self.theta["xi"] = xi
                K = xi.shape[1]
                for k in range(K):
                    self.theta["xi_" + str(k)] = xi[:, k]
            else:
                raise ValueError('Parameter {} is not defined'.format(key))

    def fit(self, X, Y, T, delta, asso_feats=None, K=None,
            xi_support=None, gamma_supports=None):
        """Fits the flash model

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
        alpha = self.fixed_effect_time_order
        n_samples, p = X.shape
        self.time_dep_feats = [feat for feat in Y.columns.values
                          if feat not in ["id", "T_long"]]
        L = len(self.time_dep_feats)
        self.n_samples = n_samples
        self.n_time_indep_features = p
        self.n_long_features = L
        q_l = alpha + 1
        r_l = 2  # Affine random effects

        X = normalize(X)  # Normalize time-independent features
        ext_feat = extract_features(Y, self.time_dep_feats, alpha)  # Features extraction
        T_u = np.unique(T[delta==1])
        self.T_u = T_u
        J, ind_1, ind_2 = get_times_infos(T, T_u)
        asso_feats = asso_feats
        if asso_feats is None:
            if self.sig_ext:
                asso_feats, nb_extracted_feat = feat_representation_extraction_sig(Y, T_u, self.sig_order)
            else:
                asso_feats, nb_extracted_feat = feat_representation_extraction(Y, L, T_u, self.fc_parameters)
        else:
            nb_extracted_feat = None
        self.asso_feats = asso_feats
        if self.sig_ext:
            nb_asso_param = asso_feats.shape[-1]
        else:
            nb_asso_param = asso_feats.shape[-1] // L
        self.nb_extracted_feat = nb_extracted_feat

        # Initialization
        if (L == 1):
            max_iter = 15
            self.initialize = False
        xi_ext = .5 * np.concatenate((np.ones(p), np.zeros(p)))
        # Initialize longitudinal submodels
        mlmm = MLMM(max_iter=max_iter, verbose=verbose, tol=tol,
                    print_every=print_every, fixed_effect_time_order=alpha)
        mlmm.fit(ext_feat)
        if self.initialize:
            beta = mlmm.fixed_effect_coeffs
            phi = np.ones((L, 1))
            D = 1e-2 * np.identity(r_l * L)
            baseline_hazard = initialize_baseline_hazard(X, T, delta)
            baseline_hazard = baseline_hazard[
                [item in T_u for item in np.unique(T)]]
        else:
            beta = mlmm.fixed_effect_coeffs
            D = mlmm.long_cov
            phi = mlmm.phi
            baseline_hazard = initialize_baseline_hazard(X, T, delta)
            baseline_hazard = baseline_hazard[
                [item in T_u for item in np.unique(T)]]
        beta_all = []
        gamma_all = []
        xi_ext_all = [np.zeros((2 * p, 1))]
        for k in range(K):
            beta_all.append(beta.reshape(-1, 1))
            if self.sig_ext:
                gamma_all.append(1e-4 * np.ones((nb_asso_param, 1)))
            else:
                gamma_all.append(1e-4 * np.ones((L * nb_asso_param, 1)))
        if gamma_supports is None:
            if self.sig_ext:
                gamma_supports = [np.ones(nb_asso_param)] * K
            else:
                gamma_supports = [np.ones(L * nb_asso_param)] * K

        for k in range(1, K):
            xi_ext_all.append(xi_ext.reshape(-1, 1))
        if xi_support is None:
            xi_support = np.ones(p)
        X = X * xi_support
        self._update_theta(beta=beta_all, xi=xi_ext_all, gamma=gamma_all,
                           long_cov=D, phi=phi, baseline_hazard=baseline_hazard)

        # Stopping criteria and bounds vector for the optim algorithm
        max_iter_lbfgs, pgtol = self.max_iter_lbfgs, self.tol
        bounds_xi = [(0, None)] * 2 * p
        max_iter_proxg = self.max_iter_proxg

        # Instanciates E-step and M-step functions
        E_func = EstepFunctions(T_u, L, alpha, self.theta)
        E_func.b_stats(ext_feat)
        F_func = MstepFunctions(X, delta, p, self.l_pen_EN, self.eta_elastic_net)
        f = self.f_data_given_latent(ext_feat, asso_feats, T, self.T_u, delta)
        Lambda_1 = E_func.Lambda_g(np.ones((n_samples, K)), f)
        pi_xi = self._get_proba(X)

        # Store init values
        obj = self._func_obj(pi_xi, f)
        self.history.update(n_iter=0, obj=obj,
                            rel_obj=np.inf, theta=self.theta)
        if verbose:
            self.history.print_history()

        for n_iter in range(1, max_iter + 1):
            try:
                # E-Step
                pi_est = self._get_post_proba(pi_xi, Lambda_1)
                self.pi_est = pi_est
                E_g1 = E_func.Eg(E_func.Eb, Lambda_1, pi_xi, f)
                E_g2 = E_func.Eg(E_func.EbbT, Lambda_1, pi_xi, f)

                # M-Step
                D = E_g2.sum(axis=0) / n_samples  # D update

                if warm_start:
                    xi_init = xi_ext_all
                    gamma_init = gamma_all
                else:
                    xi_init = [np.zeros(2*p)] * K
                    if self.sig_ext:
                        gamma_init = [np.zeros(nb_asso_param, 1)] * K
                    else:
                        gamma_init = [np.zeros(L * nb_asso_param, 1)] * K

                # xi update
                xi_ext_update = [np.zeros((2 * p, 1))]
                for k in range(1, K):
                    xi_ext_update.append(fmin_l_bfgs_b(
                        func=lambda xi_ext_: F_func.P_pen_func(pi_est, xi_ext_, k, xi_ext_all),
                        x0=xi_init[k],
                        fprime=lambda xi_ext_: F_func.grad_P_pen(pi_est, xi_ext_, k, xi_ext_all),
                        disp=False, bounds=bounds_xi, maxiter=max_iter_lbfgs,
                        pgtol=pgtol)[0].reshape(-1, 1))

                xi_ext_all = xi_ext_update

                # beta update
                pi_est_K = pi_est.T
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
                beta_update = []
                for k in range(K):
                    beta_update.append(np.linalg.inv(den[k]).dot(num[k]).reshape(-1, 1))
                self._update_theta(beta = beta_update)

                # gamma_1 update
                beta_K = beta_update
                gamma = gamma_all
                groups = np.arange(0, len(gamma[0])).reshape(L, -1).tolist()
                eta_sp_gp_l1 = self.eta_sp_gp_l1
                l_pen_SGL = self.l_pen_SGL
                prox_0 = SparseGroupL1(l_pen_SGL, eta_sp_gp_l1, groups).prox
                args_all = {"pi_est": pi_est_K, "E_g1": E_g1,
                            "phi": phi, "beta": beta_K,
                            "baseline_hazard": baseline_hazard,
                            "extracted_features": ext_feat,
                            "ind_1": ind_1, "ind_2": ind_2, "gamma": gamma}

                gamma_update = []
                for k in range(K):
                    args_sup = {"group": k,
                                "asso_feats": asso_feats * gamma_supports[k]}
                    res = copt.minimize_proximal_gradient(
                        fun=F_func.Q_func, x0=gamma_init[k].flatten(), prox=prox_0,
                        max_iter=max_iter_proxg,
                        args=[{**args_all, **args_sup}], jac=F_func.grad_Q,
                        step=self.copt_step,
                        accelerated=self.copt_accelerate)
                    gamma_update.append(res.x.reshape(-1, 1))

                # beta, gamma needs to be updated before the baseline
                self._update_theta(gamma = gamma_update)
                E_func.theta = self.theta

                # baseline hazard update
                tmp = np.exp(asso_feats.dot(np.hstack(gamma_update)))
                baseline_hazard = pd.Series(
                    data=  (((ind_1.T * delta).sum(axis=1)) /
                          ((tmp.T * ind_2.T).swapaxes(0, 1) * pi_est_K)
                          .sum(axis=2).sum(axis=1)), index=T_u)

                # phi update
                beta_stack = np.hstack(beta_update)
                (U_L, V_L, y_L, N_L) = ext_feat[1]
                phi = np.zeros((L, 1))
                for l in range(L):
                    pi_est_stack = []
                    for k in range(K):
                        pi_est_ = np.concatenate([[pi_est[i, k]] * N_L[l][i]
                                                  for i in range(n_samples)])
                        pi_est_stack.append(pi_est_)
                    pi_est_stack = np.vstack(pi_est_stack).T
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
                                   long_cov=D, xi=xi_ext_all)
                pi_xi = self._get_proba(X)
                E_func.theta = self.theta
                E_func.b_stats(ext_feat)
                f = self.f_data_given_latent(ext_feat, asso_feats, T, T_u, delta)

                if n_iter % print_every == 0:
                    prev_obj = obj
                    obj = self._func_obj(pi_xi, f)
                    rel_obj = abs(obj - prev_obj) / abs(prev_obj)
                    self.history.update(n_iter=n_iter, theta=self.theta,
                                        obj=obj, rel_obj=rel_obj)
                    if verbose:
                        self.history.print_history()

                if (n_iter + 1 > max_iter) or (rel_obj < tol):
                    self._fitted = True
                    break
                else:
                    # Update for next iteration
                    Lambda_1 = E_func.Lambda_g(np.ones((n_samples, K)), f)
            except np.linalg.LinAlgError as e:
                self._fitted = True
                break
        self._end_solve()

    def score(self, X, Y, T, delta, asso_feats=None):
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
            c_index = c_index_score(T, self.predict_marker(X, Y, asso_feats)[:, 1], delta)
            return max(c_index, 1 - c_index)
        else:
            raise ValueError('You must fit the model first')