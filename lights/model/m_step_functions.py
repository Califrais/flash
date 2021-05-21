import numpy as np
from numpy.linalg import multi_dot
from lights.base.base import logistic_loss, get_xi_from_xi_ext, get_vect_from_ext
from lights.model.regularizations import ElasticNet
from lights.model.associations import AssociationFunctionFeatures


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

    l_pen_EN : `float`, default=0.
        Level of penalization for the ElasticNet

    eta_elastic_net: `float`, default=0.1
        The ElasticNet mixing parameter, with 0 <= eta_elastic_net <= 1.
        For eta_elastic_net = 0 this is ridge (L2) regularization
        For eta_elastic_net = 1 this is lasso (L1) regularization
        For 0 < eta_elastic_net < 1, the regularization is a linear combination
        of L1 and L2
    """

    def __init__(self, fit_intercept, X, T, delta, n_long_features,
                 n_time_indep_features, l_pen_EN, eta_elastic_net,
                 fixed_effect_time_order, asso_functions_list):
        self.fit_intercept = fit_intercept
        self.X, self.T, self.delta = X, T, delta
        self.n_long_features = n_long_features
        self.n_time_indep_features = n_time_indep_features
        n_samples = len(T)
        self.n_samples = n_samples
        self.fixed_effect_time_order = fixed_effect_time_order
        self.ENet = ElasticNet(l_pen_EN, eta_elastic_net)
        T_u = np.unique(self.T)
        alpha, L = self.fixed_effect_time_order, self.n_long_features
        self.F_f, self.F_r = AssociationFunctionFeatures(asso_functions_list, T_u,
                                        alpha, L).get_asso_feat()
        self.grad_Q_fixed = None

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
        arg = args[0]
        group = arg["group"]
        beta_k = beta_k.reshape(-1, 1)
        pi_est = arg["pi_est"][group]
        E_g4, E_g5 = arg["E_g4"], arg["E_g5"]
        phi = arg["phi"]

        extracted_features = arg["extracted_features"]
        U_list, V_list, y_list, N_list = extracted_features[0]
        n_samples, n_long_features = self.n_samples, self.n_long_features
        op = np.zeros(shape=n_samples)
        for i in range(n_samples):
            U_i, V_i, y_i, n_i = U_list[i], V_list[i], y_list[i], N_list[i]
            M_i = U_i.dot(beta_k) + V_i.dot(E_g5[i]).reshape(-1, 1)
            Phi_i = [[1 / phi[l, 0]] * n_i[l] for l in range(n_long_features)]
            Phi_i = np.concatenate(Phi_i).reshape(-1, 1)
            Sigma_i = np.diag(Phi_i.flatten())
            op[i] = (M_i * y_i).T.dot(Phi_i) - .5 * multi_dot(
                [beta_k.T, U_i.T, Sigma_i]).dot(
                U_i.dot(beta_k) + 2 * V_i.dot(E_g5[i].reshape(-1, 1))) + np.trace(
                multi_dot([V_i.T, Sigma_i, V_i]).dot(E_g4[i]))
        sub_obj = (pi_est * op).sum()

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
        n_samples, alpha = self.n_samples, self.fixed_effect_time_order
        delta, L = self.delta, self.n_long_features
        q_l = alpha + 1
        arg = args[0]
        group = arg["group"]
        beta_k = beta_k.reshape(-1, 1)
        E_g5 = arg["E_g5"]
        pi_est = arg["pi_est"][group]
        extracted_features = arg["extracted_features"]
        phi = arg["phi"]

        (U_list, V_list, y_list, N_list) = extracted_features[0]
        m = np.zeros((n_samples, L * q_l))
        for i in range(n_samples):
            U_i, V_i, n_i, y_i = U_list[i], V_list[i], N_list[i], y_list[i]
            y_i = y_i.flatten()
            Phi_i = [[1 / phi[l, 0]] * n_i[l] for l in range(L)]
            Phi_i = np.diag(np.concatenate(Phi_i))
            m[i] = U_i.T.dot(Phi_i.dot(y_i - U_i.dot(beta_k.flatten()) -
                                        V_i.dot(E_g5[i]))).flatten()

        grad =  m.T.dot(pi_est)
        return -grad / n_samples

    def Q_func(self, gamma_k, *args):
        """Computes the function denoted Q in the lights paper.

        Parameters
        ----------
        gamma_k : `np.ndarray`, shape=(L * A,) or (n_time_indep_features,)
            Association parameters (time dependence or independence) for group k

        Returns
        -------
        output : `float`
            The value of the Q function
        """
        n_samples, delta = self.n_samples, self.delta
        arg = args[0]
        group = arg["group"]
        gamma_k = gamma_k.reshape(-1, 1)
        baseline_val = arg["baseline_hazard"].flatten()
        ind_1, ind_2 = arg["ind_1"], arg["ind_2"]
        E_g1 = arg["E_g1"](gamma_k).T[group].T
        E_log_g1 = arg["E_log_g1"](gamma_k).T[group].T
        pi_est = arg["pi_est"][group]
        sub_obj = (E_log_g1 * ind_1).sum(axis=1) * delta - \
                  (E_g1 * ind_2 * baseline_val).sum(axis=1)
        sub_obj = (pi_est * sub_obj).sum()
        return -sub_obj / n_samples

    def Q_x_pen_func(self, gamma_x_k_ext, *args):
        """Computes the sub objective function Q with penalty, to be minimized
        at each QNMCEM iteration using fmin_l_bfgs_b.

        Parameters
        ----------
        gamma_x_k_ext : `np.ndarray`, shape=(2 * n_time_indep_features,)
            The extension version of time independence association
            parameters for group k

        Returns
        -------
        output : `float`
            The value of the Q sub objective to be minimized at each QNMCEM step
        """
        gamma_x_k = get_vect_from_ext(gamma_x_k_ext)
        arg = args[0]
        pen = self.ENet.pen(gamma_x_k)
        Q = self.Q_func(gamma_x_k, arg)
        sub_obj = Q + pen
        return sub_obj

    def grad_Q_x_pen(self, gamma_x_k_ext, *args):
        """Computes the gradient of the sub objective Q along with time
         independence association variable and penalty

        Parameters
        ----------
        gamma_x_k_ext : `np.ndarray`, shape=(2 * n_time_indep_features,)
            The extension version of time independence association
            parameters for group k

        Returns
        -------
        output : `np.ndarray`
            The value of the Q sub objective gradient with time
         independence association variable and penalty
        """
        gamma_x_k = get_vect_from_ext(gamma_x_k_ext)
        grad_pen = self.ENet.grad(gamma_x_k)
        grad_Q = self.grad_Q_x(gamma_x_k, *args)
        return grad_Q + grad_pen

    def grad_Q_x(self, gamma_x_k, *args):
        """Computes the gradient of the function Q with time independence
        association variable

        Parameters
        ----------
        gamma_x_k : `np.ndarray`, shape=(n_time_indep_features,)
            Time independence association parameters for group k

        Returns
        -------
        output : `np.ndarray`
            The value of the Q gradient with time independence
             association variable
        """
        n_samples, delta = self.n_samples, self.delta
        arg = args[0]
        baseline_val = arg["baseline_hazard"].flatten()
        ind_2 = arg["ind_2"]
        group = arg["group"]
        E_g1 = arg["E_g1"](gamma_x_k.reshape(-1, 1)).T[group].T
        pi_est = arg["pi_est"][group]
        grad = (self.X.T * (pi_est * (delta -
                    (E_g1 * baseline_val * ind_2).sum(axis=1)))).sum(axis=1)
        grad_sub_obj = np.concatenate([grad, -grad])
        return -grad_sub_obj / n_samples

    def grad_Q_fixed_stuff(self, beta, E_g5, ind_1):
        """
        Compute a fixed stuff of gradient of Q function

        Parameters
        ----------
        beta : `list`,
            Fixed effect parameters for both groups

        E_g5 : `np.ndarray`, shape=(n_samples, r)
            The values of g5 function

        ind_1 : `np.ndarray`, shape=(n_samples, J)
            The indicator matrix for comparing event times (T == T_u)
        """
        beta = np.hstack((beta[0].reshape(-1, 1), beta[1].reshape(-1, 1)))
        self.grad_Q_fixed = (self.delta * ((self.F_f.swapaxes(0, 1).dot(beta)
                            + (self.F_r[..., np.newaxis]
                            * E_g5.T).sum(axis=2).T[..., np.newaxis]).T
                        .swapaxes(0, 1)) * ind_1.T).sum(axis=2).swapaxes(0, 1)

    def grad_Q(self, gamma_k, *args):
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
        baseline_val = arg["baseline_hazard"].flatten()
        ind_2, group = arg["ind_2"], arg["group"]
        gamma_k = gamma_k.reshape(-1, 1)
        beta_k = arg["beta"][group]
        E_g1 = arg["E_g1"](gamma_k).T[group].T
        E_g6 = arg["E_g6"](gamma_k).T[group].T
        pi_est = arg["pi_est"][group]
        F_f, F_r = self.F_f, self.F_r
        op1 = self.grad_Q_fixed[group]
        op2 = (((F_f.swapaxes(0, 1).dot(beta_k.flatten()).T[..., np.newaxis] * E_g1.T)
                + (F_r[..., np.newaxis] * E_g6.T).sum(axis=2))
               .swapaxes(1,2) * baseline_val * ind_2).sum(axis=-1)
        grad = ((op1 - op2) * pi_est).sum(axis=1)
        return -grad / n_samples
