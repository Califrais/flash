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
    def __init__(self, T_u, fixed_effect_time_order=5, n_long_features=5):
        self.K = 2  # 2 latent groups
        self.J = len(T_u)
        self.n_long_features = n_long_features
        self.q_l = fixed_effect_time_order + 1
        r_l = 2
        J = self.J

        # U, integral over U, derivative of U
        U_l, iU_l, dU_l = np.ones(J), T_u, np.zeros(J)
        for t in range(1, self.q_l):
            U_l = np.c_[U_l, T_u ** t]
            iU_l = np.c_[iU_l, (T_u ** (t + 1)) / (t + 1)]
            dU_l = np.c_[dU_l, t * T_u ** (t - 1)]

        L = n_long_features
        self.U = np.zeros(shape=(J, L, L * self.q_l))
        self.iU = np.zeros(shape=(J, L, L * self.q_l))
        self.dU = np.zeros(shape=(J, L, L * self.q_l))
        for j in range(J):
            self.U[j] = block_diag((U_l[j].reshape(1, -1), ) * L)
            self.iU[j] = block_diag((iU_l[j].reshape(1, -1),) * L)
            self.dU[j] = block_diag((dU_l[j].reshape(1, -1),) * L)

        V_l = np.c_[np.ones(J), T_u]
        iV_l = np.c_[T_u, (T_u ** 2) / 2]
        dV_l = np.c_[np.zeros(J), np.ones(J)]

        self.V = np.zeros(shape=(J, L, L * r_l))
        self.iV = np.zeros(shape=(J, L, L * r_l))
        self.dV = np.zeros(shape=(J, L, L * r_l))
        for j in range(J):
            self.V[j] = block_diag((V_l[j].reshape(1, -1),) * L)
            self.iV[j] = block_diag((iV_l[j].reshape(1, -1),) * L)
            self.dV[j] = block_diag((dV_l[j].reshape(1, -1),) * L)

        self.U_l, self.iU_l, self.dU_l = U_l, iU_l, dU_l
        self.V_l, self.iV_l, self.dV_l = V_l, iV_l, dV_l


    def _get_assoc_feat(self):
        L = self.n_long_features
        q_l = self.q_l
        r_l = 2
        J = self.J
        tmp = np.zeros(shape=(J, L * r_l, L * q_l))
        fixed_feat = np.concatenate((self.U, tmp, self.dU, self.iU), axis=1)
        tmp = np.broadcast_to(np.eye(L * r_l, L * r_l), (J, L * r_l, L * r_l))
        rand_feat = np.concatenate((self.V, tmp, self.dV, self.iV), axis=1)

        return fixed_feat, rand_feat

