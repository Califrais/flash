# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

import os
from time import time
import pandas as pd
import numpy as np

import logging

from sklearn.model_selection import ShuffleSplit
from rpy2 import robjects
from rpy2.robjects import pandas2ri, numpy2ri

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


def extract_R_feat(data):
    data_id = data.drop_duplicates(subset=["id"])
    T = data_id[["T_survival"]].values.flatten()
    delta = data_id[["delta"]].values.flatten()
    with robjects.conversion.localconverter(robjects.default_converter +
                                            pandas2ri.converter +
                                            numpy2ri.converter):
        data_R = robjects.conversion.py2rpy(data)
        T_R = robjects.conversion.py2rpy(T)
        delta_R = robjects.conversion.py2rpy(delta)

    return (data_R, T_R, delta_R)

def extract_lights_feat(data, time_indep_feat, time_dep_feat):
    X = np.float_(data[time_indep_feat].values)
    Y = data[time_dep_feat]
    T = np.float_(data[["T_survival"]].values.flatten())
    delta = np.int_(data[["delta"]].values.flatten())

    return (X, Y, T, delta)

class Dataset:
    """

    test_split : None or float

    """

    def __init__(
        self,
        name,
        *,
        label_column=None,
        time_indep_columns=None,
        time_dep_columns=None,
        df_lights=None,
        df_competing=None,
        Y_rep=None,
        test_size=0.3,
        verbose=False,
    ):
        self.name = name
        self.label_column = label_column
        self.time_indep_columns = time_indep_columns
        self.time_dep_columns = time_dep_columns
        self.test_size = test_size
        self.verbose = verbose
        self.df_lights = df_lights
        self.df_competing = df_competing
        self.Y_rep = Y_rep

    def __repr__(self):
        repr = "Dataset("
        repr += "name=%r" % self.name
        repr += ", label_column=%r" % self.label_column
        repr += ", time_indep_columns=%r" % self.time_indep_columns
        repr += ", time_dep_columns=%r" % self.time_dep_columns
        repr += ")"
        return repr

    def split(self, data_lights, Y_rep, data_competing, test_size=.3):
        id_list = data_lights["id"]
        nb_test_sample = int(test_size * len(id_list))
        id_test = np.random.choice(id_list, size=nb_test_sample, replace=False)
        data_lights_train = data_lights[~data_lights.id.isin(id_test)]
        data_lights_test = data_lights[data_lights.id.isin(id_test)]
        Y_rep_train = Y_rep[~Y_rep.id.isin(id_test)]
        Y_rep_test = Y_rep[Y_rep.id.isin(id_test)]
        data_competing_train = data_competing[~data_competing.id.isin(id_test)]
        data_competing_test = data_competing[data_competing.id.isin(id_test)]

        return data_lights_train, Y_rep_train, data_competing_train, \
               data_lights_test, Y_rep_test, data_competing_test


    def extract(self, data_lights, data_competing):
        # lights's data extraction
        data_lights_extract = extract_lights_feat(data_lights,
                                                  self.time_dep_columns,
                                                  self.time_indep_columns)
        # competing's data extraction
        data_competing_extract = extract_R_feat(data_competing)

        return data_lights_extract, data_competing_extract
