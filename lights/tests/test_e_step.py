# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import unittest
from lights.tests.test_simu import get_train_data


class Test(unittest.TestCase):

    def test_g1(self):
        """Tests the g1 function
        """
        X, Y, T, delta = get_train_data()

    def test_g2(self):
        """Tests the g2 function
        """
        X, Y, T, delta = get_train_data()



if __name__ == "main":
    unittest.main()
