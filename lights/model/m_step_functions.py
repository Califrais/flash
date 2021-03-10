import numpy as np
from numpy.linalg import multi_dot
from lights.base.base import logistic_loss, get_xi_from_xi_ext, get_vect_from_ext
from lights.model.regularizations import ElasticNet
from lights.model.associations import AssociationFunctions


class MstepFunctions:
    """A class to define functions relative to the M-step of the QNMCEM

    Parameters
    ----------
    fit_intercept : `bool`
        If `True`, include an intercept in the model for the time independent
        features

    X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
        The time-independent features matrix

    T : `np.ndarray`, shape=(n_samples,)
        The censored times of the event of interest

    delta : `np.ndarray`, shape=(n_samples,)
        The censoring indicator

    n_time_indep_features : `int`
        Number of time-independent features

    n_long_features : `int`
        Number of longitudinal features

    l_pen : `float`, default=0
        Level of penalization for the ElasticNet and the Sparse Group l1

    eta_elastic_net: `float`, default=0.1
        The ElasticNet mixing parameter, with 0 <= eta_elastic_net <= 1.
        For eta_elastic_net = 0 this is ridge (L2) regularization
        For eta_elastic_net = 1 this is lasso (L1) regularization
        For 0 < eta_elastic_net < 1, the regularization is a linear combination
        of L1 and L2
    """

    def __init__(self, fit_intercept, X, T, delta, n_long_features,
                 n_time_indep_features, l_pen, eta_elastic_net,
                 nb_asso_features, fixed_effect_time_order, asso_functions):
        self.fit_intercept = fit_intercept
        self.X, self.T, self.delta = X, T, delta
        self.n_long_features = n_long_features
        self.n_time_indep_features = n_time_indep_features
        n_samples = len(T)
        self.n_samples = n_samples
        self.nb_asso_features = nb_asso_features
        self.fixed_effect_time_order = fixed_effect_time_order
        self.asso_functions = asso_functions
        self.ENet = ElasticNet(l_pen[0], eta_elastic_net)
        T_u = np.unique(self.T)
        alpha, L = self.fixed_effect_time_order, self.n_long_features
        self.F_f, self.F_r = AssociationFunctions(asso_functions, T_u,
                                        alpha, L).get_asso_feat()

    def P_pen_func(self, pi_est, xi_ext):
        """Computes the sub objective function P with penalty, to be minimized
        at each QNMCEM iteration using fmin_l_bfgs_b.

        Parameters
        ----------
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
        xi_0, xi = get_xi_from_xi_ext(xi_ext, self.fit_intercept)
        pen = self.ENet.pen(xi)
        P = self.P_func(pi_est, xi_ext)
        sub_obj = P + pen
        return sub_obj

    def P_func(self, pi_est, xi_ext):
        """Computes the function denoted P in the lights paper.

        Parameters
        ----------
        pi_est : `np.ndarray`, shape=(n_samples,)
            The estimated posterior probability of the latent class membership
            obtained by the E-step

        xi_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
            The time-independent coefficient vector decomposed on positive and
            negative parts

        Returns
        -------
        P : `float`
            The value of the P sub objective
        """
        xi_0, xi = get_xi_from_xi_ext(xi_ext, self.fit_intercept)
        u = xi_0 + self.X.dot(xi)
        P = (pi_est * logistic_loss(u)).mean()
        return P

    def grad_P(self, pi_est, xi_ext):
        """Computes the gradient of the function P

        Parameters
        ----------
        pi_est : `np.ndarray`, shape=(n_samples,)
            The estimated posterior probability of the latent class membership
            obtained by the E-step

        xi_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
            The time-independent coefficient vector decomposed on positive and
            negative parts

        Returns
        -------
        output : `np.ndarray`
            The value of the P sub objective gradient
        """
        fit_intercept = self.fit_intercept
        X, n_samples = self.X, self.n_samples
        xi_0, xi = get_xi_from_xi_ext(xi_ext, fit_intercept)
        u = xi_0 + X.dot(xi)
        if fit_intercept:
            X = np.concatenate((np.ones(n_samples).reshape(1, n_samples).T, X),
                               axis=1)
        grad = X * (pi_est * np.exp(-logistic_loss(-u))).reshape(-1, 1)
        grad = - grad.mean(axis=0)
        grad_sub_obj = np.concatenate([grad, -grad])
        return grad_sub_obj

    def grad_P_pen(self, pi_est, xi_ext):
        """Computes the gradient of the sub objective P with penalty

        Parameters
        ----------
        pi_est : `np.ndarray`, shape=(n_samples,)
            The estimated posterior probability of the latent class membership
            obtained by the E-step

        xi_ext : `np.ndarray`, shape=(2*n_time_indep_features,)
            The time-independent coefficient vector decomposed on positive and
            negative parts

        Returns
        -------
        output : `np.ndarray`
            The value of the P sub objective gradient
        """
        fit_intercept = self.fit_intercept
        n_time_indep_features = self.n_time_indep_features
        xi_0, xi = get_xi_from_xi_ext(xi_ext, fit_intercept)
        grad_pen = self.ENet.grad(xi)
        if fit_intercept:
            grad_pen = np.concatenate([[0], grad_pen[:n_time_indep_features],
                                       [0], grad_pen[n_time_indep_features:]])
        grad_P = self.grad_P(pi_est, xi_ext)
        return grad_P + grad_pen

    def R_func(self, beta_k, *args):
        """Computes the function denoted R in the lights paper

        Parameters
        ----------
        beta_k : `np.ndarray`, shape=(n_long_features * q_l,)
            Fixed effect parameters for group k

        Returns
        -------
        output : `float`
            The value of the R function
        """
        p, L = self.n_time_indep_features, self.n_long_features
        delta = self.delta.reshape(-1, 1)
        arg = args[0]
        baseline_val = arg["baseline_hazard"].values.flatten()
        ind_1, ind_2 = arg["ind_1"] * 1, arg["ind_2"] * 1
        group = arg["group"]
        beta_k = beta_k.reshape(-1, 1)
        gamma_k = arg["gamma"][group][p:]
        pi_est = arg["pi_est"][group]
        E_g1 = arg["E_g1"](beta_k).T[group].T
        E_g4, E_g5 = arg["E_g4"], arg["E_g5"]
        phi = arg["phi"]

        op1 = (delta * (self.F_f.dot(beta_k.flatten())
                        + (self.F_r.swapaxes(0, 1)[..., np.newaxis]
                        * E_g5.T).sum(axis=2).T).dot(gamma_k.flatten()) * ind_1
               - (E_g1 * baseline_val * ind_2)).sum(axis=1)

        extracted_features = arg["extracted_features"]
        U_list, V_list, y_list, N_list = extracted_features[0]
        n_samples, n_long_features = self.n_samples, self.n_long_features
        op2 = np.zeros(shape=n_samples)
        for i in range(n_samples):
            U_i, V_i, y_i, n_i = U_list[i], V_list[i], y_list[i], N_list[i]
            M_i = U_i.dot(beta_k) + V_i.dot(E_g5[i]).reshape(-1, 1)
            Phi_i = [[1 / phi[l, 0]] * n_i[l] for l in range(n_long_features)]
            Phi_i = np.concatenate(Phi_i).reshape(-1, 1)
            Sigma_i = np.diag(Phi_i.flatten())
            op2[i] = (M_i * y_i).T.dot(Phi_i) - .5 * multi_dot(
                [beta_k.T, U_i.T, Sigma_i]).dot(
                U_i.dot(beta_k) + 2 * V_i.dot(E_g5[i].reshape(-1, 1))) + np.trace(
                multi_dot([V_i.T, Sigma_i, V_i]).dot(E_g4[i]))
        sub_obj = (pi_est * (op1 + op2)).sum()

        return -sub_obj / n_samples

    def grad_R(self, beta_k, *args):
        """Computes the gradient of the function R

        Parameters
        ----------
        beta_k : `np.ndarray`, shape=(n_long_features * q_l,)
            Fixed effect parameters for group k

        Returns
        -------
        output : `np.ndarray`
            The value of the R gradient
        """
        p, L = self.n_time_indep_features, self.n_long_features
        n_samples, alpha = self.n_samples, self.fixed_effect_time_order
        delta = self.delta
        q_l = alpha + 1
        arg = args[0]
        baseline_val = arg["baseline_hazard"].values.flatten()
        ind_1, ind_2 = arg["ind_1"] * 1, arg["ind_2"] * 1
        group = arg["group"]
        beta_k = beta_k.reshape(-1, 1)
        E_g1 = arg["E_g1"](beta_k).T[group].T
        E_g5 = arg["E_g5"]
        pi_est = arg["pi_est"][group]
        extracted_features = arg["extracted_features"]
        phi = arg["phi"]
        gamma_k = arg["gamma"][group][p:].flatten()
        tmp = self.F_f.swapaxes(1, 2).dot(gamma_k)
        m1 = ind_1.dot(tmp).T * delta - (baseline_val * E_g1 * ind_2).dot(tmp).T

        (U_list, V_list, y_list, N_list) = extracted_features[0]
        m2 = np.zeros((n_samples, L * q_l))
        for i in range(n_samples):
            U_i, V_i, n_i, y_i = U_list[i], V_list[i], N_list[i], y_list[i]
            y_i = y_i.flatten()
            Phi_i = [[1 / phi[l, 0]] * n_i[l] for l in range(L)]
            Phi_i = np.diag(np.concatenate(Phi_i))
            m2[i] = U_i.T.dot(Phi_i.dot(y_i - U_i.dot(beta_k.flatten()) -
                                        V_i.dot(E_g5[i]))).flatten()

        grad = (m1 + m2.T).dot(pi_est)
        return -grad / n_samples

    def Q_func(self, gamma_k, *args):
        """Computes the function denoted Q in the lights paper.

        Parameters
        ----------
        gamma_k : `np.ndarray`, shape=(nb_asso_feat,)
            Association parameters for group k

        Returns
        -------
        output : `float`
            The value of the Q function
        """
        n_samples, delta = self.n_samples, self.delta
        arg = args[0]
        group = arg["group"]
        gamma_k = gamma_k.reshape(-1, 1)
        baseline_val = arg["baseline_hazard"].values.flatten()
        ind_1, ind_2 = arg["ind_1"] * 1, arg["ind_2"] * 1
        E_g1 = arg["E_g1"](gamma_k).T[group].T
        E_log_g1 = arg["E_log_g1"](gamma_k).T[group].T
        pi_est = arg["pi_est"][group]
        sub_obj = (E_log_g1 * ind_1).sum(axis=1) * delta - \
                  (E_g1 * ind_2 * baseline_val).sum(axis=1)
        sub_obj = (pi_est * sub_obj).sum()
        return -sub_obj / n_samples

    def Q_dep_func(self, gamma_dep, *args):
        """Computes the sub objective function Q with time
         dependence association variable, to be minimized
        at each QNMCEM iteration using fmin_l_bfgs_b.

        Parameters
        ----------
        gamma_dep : `np.ndarray`, shape=(L * A,)
            Time dependence association parameters for group k

        Returns
        -------
        output : `float`
            The value of the Q sub objective to be minimized at each QNMCEM step
        """
        arg = args[0]
        group = arg["group"]
        gamma_indep = arg["gamma_indep"][group]
        gamma_k = np.concatenate((gamma_indep, gamma_dep)).reshape(-1, 1)
        Q = self.Q_func(gamma_k, arg)
        return Q

    def Q_indep_pen_func(self, gamma_indep_ext, *args):
        """Computes the sub objective function Q with penalty, to be minimized
        at each QNMCEM iteration using fmin_l_bfgs_b.

        Parameters
        ----------
        gamma_indep_ext : `np.ndarray`, shape=(2 * n_time_indep_features,)
            The extension version of time independence association
            parameters for group k

        Returns
        -------
        output : `float`
            The value of the Q sub objective to be minimized at each QNMCEM step
        """
        gamma_indep = get_vect_from_ext(gamma_indep_ext)
        arg = args[0]
        group = arg["group"]
        gamma_dep = arg["gamma_dep"][group]
        gamma_k = np.concatenate((gamma_indep, gamma_dep)).reshape(-1, 1)
        pen = self.ENet.pen(gamma_indep)
        Q = self.Q_func(gamma_k, arg)
        sub_obj = Q + pen
        return sub_obj

    def grad_Q_indep_pen(self, gamma_indep_ext, *args):
        """Computes the gradient of the sub objective Q along with time
         independence association variable and penalty

        Parameters
        ----------
        gamma_indep_ext : `np.ndarray`, shape=(2 * n_time_indep_features,)
            The extension version of time independence association
            parameters for group k

        Returns
        -------
        output : `np.ndarray`
            The value of the Q sub objective gradient with time
         independence association variable and penalty
        """
        p = self.n_time_indep_features
        gamma_indep = get_vect_from_ext(gamma_indep_ext)
        grad_pen = self.ENet.grad(gamma_indep)
        grad_Q = self.grad_Q_indep(gamma_indep, *args)
        return grad_Q + grad_pen

    def grad_Q_indep(self, gamma_indep, *args):
        """Computes the gradient of the function Q with time independence
        association variable

        Parameters
        ----------
        gamma_indep : `np.ndarray`, shape=(n_time_indep_features,)
            Time independence association parameters for group k

        Returns
        -------
        output : `np.ndarray`
            The value of the Q gradient with time independence
             association variable
        """
        n_samples, delta = self.n_samples, self.delta
        arg = args[0]
        baseline_val = arg["baseline_hazard"].values.flatten()
        ind_2 = arg["ind_2"] * 1
        group = arg["group"]
        gamma_dep = arg["gamma_dep"][group]
        gamma_k = np.concatenate((gamma_indep, gamma_dep)).reshape(-1, 1)
        E_g1 = arg["E_g1"](gamma_k).T[group].T
        pi_est = arg["pi_est"][group]
        grad = (self.X.T * (pi_est * (delta -
                    (E_g1 * baseline_val * ind_2).sum(axis=1)))).sum(axis=1)
        grad_sub_obj = np.concatenate([grad, -grad])
        return -grad_sub_obj / n_samples

    def grad_Q_dep(self, gamma_k_dep, *args):
        """Computes the gradient of the function Q  with time dependence
        association variable

        Parameters
        ----------
        gamma_k_dep : `np.ndarray`, shape=(L*A,)
            Time dependence association parameters for group k

        Returns
        -------
        output : `np.ndarray`
            The value of the Q gradient with time dependence
        association variable
        """
        n_samples, delta = self.n_samples, self.delta
        arg = args[0]
        baseline_val = arg["baseline_hazard"].values.flatten()
        ind_1, ind_2 = arg["ind_1"] * 1, arg["ind_2"] * 1
        group = arg["group"]
        gamma_indep = arg["gamma_indep"][group]
        gamma_k = np.concatenate((gamma_indep, gamma_k_dep)).reshape(-1, 1)
        beta_k = arg["beta"][group]
        E_g1 = arg["E_g1"](gamma_k).T[group].T
        E_g6 = arg["E_g6"](gamma_k).T[group].T
        E_g5 = arg["E_g5"]
        pi_est = arg["pi_est"][group]
        F_f, F_r = self.F_f, self.F_r
        op1 = (delta * (F_f.dot(beta_k.flatten())
                 + (F_r.swapaxes(0, 1)[..., np.newaxis] * E_g5.T).sum(
                    axis=2).T).T * ind_1.T * delta).sum(axis=1)
        op2 = (((F_f.dot(beta_k.flatten()).T[..., np.newaxis] * E_g1.T)
                + (F_r.swapaxes(0, 1)[..., np.newaxis] * E_g6.T).sum(axis=2))
               .swapaxes(1,2) * baseline_val * ind_2).sum(axis=-1)
        grad = ((op1 - op2) * pi_est).sum(axis=1)
        return -grad / n_samples
