import numpy as np
from lights.base.base import block_diag


class AssociationFunctions:
    """A class to define all the association functions

    Parameters
    ----------
    T_u : `np.ndarray`, shape=(J,)
        The J unique training censored times of the event of interest

    fixed_effect_time_order: `int`, default=5
        Order of the higher time monomial considered for the representations of
        the time-varying features corresponding to the fixed effect. The
        dimension of the corresponding design matrix is then equal to
        fixed_effect_time_order + 1

    n_long_features: `int`, default=5
        Number of longitudinal features
    """

    def __init__(self, asso_functions, T_u,
                 fixed_effect_time_order=5, n_long_features=5):
        self.K = 2  # 2 latent groups
        self.J = len(T_u)
        self.n_long_features = n_long_features
        self.q_l = fixed_effect_time_order + 1
        self.asso_functions = asso_functions
        q_l, r_l = self.q_l, 2
        J = self.J

        # U, integral over U, derivative of U
        U_l, iU_l, dU_l = np.ones(J), T_u, np.zeros(J)
        for t in range(1, self.q_l):
            U_l = np.c_[U_l, T_u ** t]
            iU_l = np.c_[iU_l, (T_u ** (t + 1)) / (t + 1)]
            dU_l = np.c_[dU_l, t * T_u ** (t - 1)]

        V_l = np.c_[np.ones(J), T_u]
        iV_l = np.c_[T_u, (T_u ** 2) / 2]
        dV_l = np.c_[np.zeros(J), np.ones(J)]
        self.fixed_feat = {"lp" : U_l,
                           "re" : np.zeros(shape=(J, r_l, q_l)),
                           "tps" : dU_l,
                           "ce" : iU_l}
        self.rand_feat = {"lp" :  V_l,
                           "re" : [np.eye(r_l, r_l)] * J,
                           "tps" : dV_l,
                           "ce" : iV_l}
        self.U_l, self.iU_l, self.dU_l = U_l, iU_l, dU_l
        self.V_l, self.iV_l, self.dV_l = V_l, iV_l, dV_l

    def get_asso_feat(self):
        """
        Produces matrices of stacked association fixed and random features
        Returns
        -------
        fixed_feat : `np.ndarray`, shape=(J, A*r, q)
            Feature corresponding to fixed effect

        rand_feat : `np.ndarray`, shape=(J, A*r, r)
            Feature corresponding to random effect
        """
        L = self.n_long_features
        q_l, r_l = self.q_l, 2
        r, q = L * r_l, L * q_l
        J = self.J
        asso_functions = self.asso_functions
        nb_asso_param = len(asso_functions)
        if 're' in asso_functions:
            nb_asso_param += 1
        fixed_feat = np.zeros(shape=(J, nb_asso_param * L, q))
        rand_feat = np.zeros(shape=(J, nb_asso_param * L, r))
        for j in range(J):
            tmp_U = np.array([]).reshape(0, q_l)
            tmp_V = np.array([]).reshape(0, r_l)
            for asso_function in asso_functions:
                tmp_U = np.vstack((tmp_U, self.fixed_feat[asso_function][j]))
                tmp_V = np.vstack((tmp_V, self.fixed_feat[asso_function][j]))
            fixed_feat[j] = block_diag((tmp_U,) * L)
            rand_feat[j] = block_diag((tmp_V,) * L)
        return fixed_feat, rand_feat
