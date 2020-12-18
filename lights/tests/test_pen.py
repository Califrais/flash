# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
import numpy as np
from lights.model.regularizations import ElasticNet, SparseGroupL1


class Test(unittest.TestCase):

    def setUp(self):
        l_pen = 2.
        eta_elastic_net = .2
        eta_sp_gp_l1 = .3
        groups = [[0, 1], [2, 3, 4]]
        self.ENet = ElasticNet(l_pen, eta_elastic_net)
        self.v = np.array([-1, 0, 3, 4, -2, 5])
        self.n_long_features = 3
        self.step_size = 0.3
        self.SGL1 = SparseGroupL1(l_pen, eta_sp_gp_l1, groups)

    def test_elastic_net(self):
        """Test Elastic net penalty
        """
        self.setUp()
        pen = self.ENet.pen(self.v)
        pen_ = 35
        np.testing.assert_almost_equal(pen, pen_)

    def test_grad_elastic_net(self):
        """Test gradient of Elastic net penalty
        """
        self.setUp()
        grad = self.ENet.grad(self.v)
        grad_ = np.array([1.2, 1.6, 2.8, 3.2, .8, 3.6, 2, 1.6, .4, 0, 2.4, -.4])
        np.testing.assert_almost_equal(grad, grad_)

    def test_sgl1_pen(self):
        """Test Sparse Group Lasso penalty
        """
        self.setUp()
        pen = self.SGL1.pen(self.v)
        pen_ = 24.831
        np.testing.assert_almost_equal(pen, pen_, 3)

    def test_sgl1_prox(self):
        """Test Sparse Group Lasso proximal operator
        """
        self.setUp()
        prox = self.SGL1.prox(self.v, self.step_size)
        prox_ = np.array([-0.4, 0., 2.481, 3.443, -1.519, 4.58])
        np.testing.assert_almost_equal(prox, prox_, 3)


if __name__ == "main":
    unittest.main()
