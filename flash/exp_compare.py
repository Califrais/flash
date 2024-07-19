import os
os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Versions/4.0/Resources"
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
from flash.inference import ext_EM
from flash.base.base import feat_representation_extraction
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines.utils import concordance_index as c_index_score
from time import time
from competing_methods.all_model import load_data, extract_flash_feat, \
	extract_R_feat, truncate_data
import pickle as pkl

import rpy2.robjects as robjects

def run():
    test_size = .3
    params_path = "flash/params/"
    res = pd.DataFrame(columns=["Algo", "Dataset", "C_index", "Time"])
    datasets = ["FLASH_simu", "joineRML_simu", "PBCseq", "Aids"]
    for j in range(len(datasets)):
        dataset = datasets[j]
        data, time_dep_feat, time_indep_feat = load_data(data_name=dataset)
        data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
        id_list = np.unique(data["id"])
        nb_test_sample = int(test_size * len(id_list))

        with open(params_path + dataset + '/final_selected_fc_parameters', 'rb') as f:
            final_selected_fc_parameters = pkl.load(f)

        with open(params_path + dataset + '/Flash_penalties', 'rb') as f:
            flash_pens = np.load(f)

        for i in range(50):
            np.random.seed(i)
            id_test = np.random.choice(id_list, size=nb_test_sample, replace=False)
            data_test = data[data.id.isin(id_test)]
            data_train = data[~data.id.isin(id_test)]
            data_train["T_max"] = data_train["T_survival"]
            X_flash_train, Y_flash_train, T_train, delta_train = extract_flash_feat(
                data_train, time_indep_feat, time_dep_feat)
            data_R_train, T_train, delta_train = extract_R_feat(data_train,
                                                                time_indep_feat,
                                                                time_dep_feat)
            data_test_truncated = truncate_data(data_test)
            X_flash_test, Y_flash_test, T_test, delta_test = extract_flash_feat(
                data_test_truncated, time_indep_feat, time_dep_feat)
            data_R_test, T_test, delta_test = extract_R_feat(data_test_truncated,
                                                             time_indep_feat,
                                                             time_dep_feat)

            T_u = np.unique(T_train[delta_train == 1])
            n_long_features = len(time_dep_feat)
            asso_feat_test, _ = feat_representation_extraction(Y_flash_test,
                                                               n_long_features, T_u,
                                                               final_selected_fc_parameters)
            asso_feat_train, _ = feat_representation_extraction(Y_flash_train,
                                                                n_long_features,
                                                                T_u,
                                                                final_selected_fc_parameters)
            # Training with best cross-validation params
            l_pen_EN, l_pen_SGL = flash_pens[0], flash_pens[1]
            learner = ext_EM(print_every=1, l_pen_SGL=l_pen_SGL, l_pen_EN=l_pen_EN,
                             fc_parameters=final_selected_fc_parameters,
                             verbose=False)
            start = time()
            learner.fit(X_flash_train, Y_flash_train, T_train, delta_train,
                        asso_feat_train, K=2)
            Flash_c_index_train = learner.score(X_flash_train, Y_flash_train,
                                                T_train, delta_train,
                                                asso_feat_train)
            Flash_c_index = learner.score(X_flash_test, Y_flash_test, T_test,
                                          delta_test, asso_feat_test)
            if Flash_c_index_train < .5:
                Flash_c_index_train = 1 - Flash_c_index_train
                Flash_c_index = 1 - Flash_c_index
            Flash_exe_time = time() - start

            # Multivariate shared random effect model.
            start = time()
            robjects.r.source(os.getcwd() + "/competing_methods/JMBayes.R")
            trained_JMBayes = robjects.r["fit"](data_R_train,
                                                robjects.StrVector(time_dep_feat),
                                                robjects.StrVector(time_indep_feat))

            JMBayes_marker_train = np.array(
                robjects.r["score"](trained_JMBayes, data_R_train))
            JMBayes_marker = np.array(
                robjects.r["score"](trained_JMBayes, data_R_test))
            JMBayes_probs = robjects.r["prob"](trained_JMBayes, data_R_test)
            T_max = data_test_truncated.drop_duplicates(subset=['id']).T_max.values
            JMBayes_probs_ = np.zeros(len(T_max))
            for j in range(len(T_max)):
                JMBayes_probs_[j] = \
                JMBayes_probs[j][JMBayes_probs[j][:, 0] == T_max[j]][:, 2][0]
            JMBayes_c_index_train = c_index_score(T_train, JMBayes_marker_train,
                                                  delta_train)
            JMBayes_c_index = c_index_score(T_test, JMBayes_marker, delta_test)
            if JMBayes_c_index_train < .5:
                JMBayes_c_index_train = 1 - JMBayes_c_index_train
                JMBayes_c_index = 1 - JMBayes_c_index

            JMBayes_exe_time = time() - start

            # Multivariate joint latent class model.
            start = time()
            robjects.r.source(os.getcwd() + "/competing_methods/MPJLCMM.R")
            trained_long_model, trained_mjlcmm = robjects.r["MPJLCMM_fit"](
                data_R_train,
                robjects.StrVector(time_dep_feat),
                robjects.StrVector(time_indep_feat))
            MPJLCMM_pred_train = robjects.r["MPJLCMM_score"](trained_long_model,
                                                             trained_mjlcmm,
                                                             time_indep_feat,
                                                             data_R_train)
            MPJLCMM_marker_train = np.array(MPJLCMM_pred_train.rx2('pprob')[2])
            MPJLCMM_c_index_train = c_index_score(T_train, MPJLCMM_marker_train,
                                                  delta_train)
            MPJLCMM_pred = robjects.r["MPJLCMM_score"](trained_long_model,
                                                       trained_mjlcmm,
                                                       time_indep_feat, data_R_test)
            MPJLCMM_marker = np.array(MPJLCMM_pred.rx2('pprob')[2])
            MPJLCMM_c_index = c_index_score(T_test, MPJLCMM_marker, delta_test)
            if MPJLCMM_c_index_train < .5:
                MPJLCMM_c_index_train = 1 - MPJLCMM_c_index_train
                MPJLCMM_c_index = 1 - MPJLCMM_c_index
            MPJLCMM_exe_time = time() - start

            tmp_df = pd.DataFrame(
                [["FLASH", dataset, Flash_c_index, Flash_exe_time],
                 ["JMBayes", dataset, JMBayes_c_index, JMBayes_exe_time],
                 ["LCMM", dataset, MPJLCMM_c_index, MPJLCMM_exe_time]],
                columns=["Algo", "Dataset", "C_index", "Time"])
            res = res.append(tmp_df)


    fontsize=20
    # Create layout
    layout = [
        [ "A", "A", "A", "A"],
        [ "B", "C", "D", "E"]
    ]
    _, ax = plt.subplot_mosaic(layout, figsize=(16,16))
    sns.set(style="white", font="STIXGeneral", context='talk',palette='colorblind')

    sns.boxplot(x="Dataset", y="C_index", hue="Algo", data=res, ax=ax["A"])
    ax["A"].set_ylabel(r"C_index", fontsize=fontsize, fontdict=dict(weight='bold'))
    ax["A"].set(xticklabels=[])
    ax["A"].set(xlabel=None)
    ax["A"].legend(fontsize = fontsize)
    plt.subplots_adjust(wspace=0.5)


    sns.boxplot(data=res[res.Dataset == "FLASH_simu"], ax=ax["B"], x="Dataset", y="Time", hue='Algo')
    ax["B"].set_xlabel('FLASH_simu', fontsize=fontsize, fontdict=dict(weight='bold'))
    ax["B"].set_yscale("log")
    ax["B"].get_legend().remove()
    ax["B"].set_ylabel(r"Running time (second)", fontsize=fontsize, fontdict=dict(weight='bold'))
    ax["B"].set(xticklabels=[])

    sns.boxplot(data=res[res.Dataset == "joineRML_simu"], ax=ax["C"], x="Dataset", y="Time", hue='Algo')
    ax["C"].set_xlabel('joineRML_simu', fontsize=fontsize, fontdict=dict(weight='bold'))
    ax["C"].set_yscale("log")
    ax["C"].get_legend().remove()
    ax["C"].set_ylabel('')
    ax["C"].set(xticklabels=[])

    sns.boxplot(data=res[res.Dataset == "PBCseq"], ax=ax["D"], x="Dataset", y="Time", hue='Algo')
    ax["D"].set_xlabel('PBCseq', fontsize=fontsize, fontdict=dict(weight='bold'))
    ax["D"].set_yscale("log")
    ax["D"].get_legend().remove()
    ax["D"].set_ylabel('')
    ax["D"].set(xticklabels=[])

    sns.boxplot(data=res[res.Dataset == "Aids"], ax=ax["E"], x="Dataset", y="Time", hue='Algo')
    ax["E"].set_xlabel('Aids', fontsize=fontsize, fontdict=dict(weight='bold'))
    ax["E"].set_yscale("log")
    ax["E"].get_legend().remove()
    ax["E"].set_ylabel('')
    ax["E"].set(xticklabels=[])

    plt.savefig('./flash_competing.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    run()