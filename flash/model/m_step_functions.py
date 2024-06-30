import numpy as np
from flash.base.base import logistic_loss, get_xi_from_xi_ext
from flash.model.regularizations import ElasticNet


class MstepFunctions:
    """A class to define functions relative to the M-step of the prox_QNMCEM

    Parameters
    ----------

    X : `np.ndarray`, shape=(n_samples, n_time_indep_features)
        The time-independent features matrix

    delta : `np.ndarray`, shape=(n_samples,)
        The censoring indicator

    n_time_indep_features : `int`
        Number of time-independent features

    l_pen_EN : `float`, default=0.
        Level of penalization for the ElasticNet

    eta_elastic_net: `float`, default=0.1
        The ElasticNet mixing parameter, with 0 <= eta_elastic_net <= 1.
        For eta_elastic_net = 0 this is ridge (L2) regularization
        For eta_elastic_net = 1 this is lasso (L1) regularization
        For 0 < eta_elastic_net < 1, the regularization is a linear combination
        of L1 and L2
    """

    def __init__(self, X, delta, n_time_indep_features,
                 l_pen_EN, eta_elastic_net):
        self.X, self.delta = X, delta
        self.n_time_indep_features = n_time_indep_features
        self.n_samples = X.shape[0]
        self.ENet = ElasticNet(l_pen_EN, eta_elastic_net)

    def P_pen_func(self, pi_est, xi_ext, k, xi_ext_all):
        """Computes the sub objective function P with penalty, to be minimized
        at each prox_QNMCEM iteration using fmin_l_bfgs_b.

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
            The value of the P sub objective to be minimized at each prox_QNMCEM step
        """
        xi_ext_all_ = xi_ext_all.copy()
        xi_ext_all_[k] = xi_ext
        xi = get_xi_from_xi_ext(xi_ext_all_)
        pen = self.ENet.pen(xi[:, k])
        P = self.P_func(pi_est, xi)
        sub_obj = P + pen
        return sub_obj

    def P_func(self, pi_est, xi):
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
        K = pi_est.shape[1]
        u = self.X.dot(xi)
        #P = ((1 - pi_est) * logistic_loss(-u) +
        #     pi_est * logistic_loss(u)).mean()
        P_sum = 0
        pi_xi = (np.exp(u).T / np.exp(u).sum(axis=1)).T
        for k in range(K):
            P_sum += pi_est[:, k] * pi_xi[:, k]

        P = np.mean(P_sum)

        return P

    def grad_P(self, pi_est, pi_xi):
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
        X = self.X
        grad = (X.T * (pi_est - pi_xi)).T
        grad = - grad.mean(axis=0)
        grad_sub_obj = np.concatenate([grad, -grad])

        return grad_sub_obj

    def grad_P_pen(self, pi_est, xi_ext, k, xi_ext_all):
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
        xi_ext_all_ = xi_ext_all.copy()
        xi_ext_all_[k] = xi_ext
        xi = get_xi_from_xi_ext(xi_ext_all_)
        u = self.X.dot(xi)
        pi_xi = (np.exp(u).T / np.exp(u).sum(axis=1)).T

        grad_pen = self.ENet.grad(xi[:, k])
        grad_P = self.grad_P(pi_est[:, k], pi_xi[:, k])
        return grad_P + grad_pen

    def Q_func(self, gamma_k, *args):
        """Computes the function denoted Q in the lights paper.

        Parameters
        ----------
        gamma_k : `np.ndarray`, shape=(L * nb_asso_param,)
            Association parameters  for group k

        Returns
        -------
        output : `float`
            The value of the Q function
        """
        n_samples, delta = self.n_samples, self.delta
        arg = args[0]
        ind_1, ind_2 = arg["ind_1"], arg["ind_2"]
        group = arg["group"]
        asso_feats = arg["asso_feats"]
        baseline_val = arg["baseline_hazard"].values.flatten()
        pi_est = arg["pi_est"][group]
        tmp = asso_feats.dot(gamma_k)
        sub_obj = (tmp * ind_1).sum(axis=1) * delta  - \
                  ((np.exp(tmp) * baseline_val) * ind_2).sum(axis=1)
        sub_obj = (pi_est * sub_obj).sum()
        return -sub_obj / n_samples

    def grad_Q(self, gamma_k, *args):
        """Computes the gradient of the function Q  with association variable

        Parameters
        ----------
        gamma_k : `np.ndarray`, shape=(L * nb_asso_param,)
            Association parameters for group k

        Returns
        -------
        output : `np.ndarray`
            The value of the Q gradient with association variable
        """
        n_samples, delta = self.n_samples, self.delta
        arg = args[0]
        ind_1, ind_2 = arg["ind_1"], arg["ind_2"]
        group=  arg["group"]
        asso_feats = arg["asso_feats"]
        baseline_val = arg["baseline_hazard"].values.flatten()
        pi_est = arg["pi_est"][group]
        tmp = asso_feats.dot(gamma_k)
        grad = (asso_feats.T * ind_1.T).sum(axis=1) * delta - \
               ((asso_feats.T * np.exp(tmp).T).swapaxes(1, 2) * baseline_val * ind_2).sum(axis=-1)
        grad = (grad * pi_est).sum(axis=-1)
        return -grad / n_samples
