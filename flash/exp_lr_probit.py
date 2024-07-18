import os
os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Versions/4.0/Resources"
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
from flash.inference import ext_EM
from flash.base.base import feat_representation_extraction
import numpy as np
from competing_methods.all_model import load_data, extract_flash_feat, truncate_data
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

import rpy2.robjects as robjects

def run():
    test_size = .3
    result_home_path = "results/"
    res = pd.DataFrame(columns=["Dataset", "C_index"])
    datasets = ["FLASH_simu", "FLASH_simu_probit"]
    for j in range(len(datasets)):
        dataset = datasets[j]
        data, time_dep_feat, time_indep_feat = load_data(data_name=dataset)
        data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
        id_list = np.unique(data["id"])
        nb_test_sample = int(test_size * len(id_list))

        with open(result_home_path + 'FLASH_simu/final_selected_fc_parameters', 'rb') as f:
            final_selected_fc_parameters = pkl.load(f)

        with open(result_home_path + 'FLASH_simu/Flash_penalties', 'rb') as f:
            flash_pens = np.load(f)

        for i in range(50):
            np.random.seed(i)
            id_test = np.random.choice(id_list, size=nb_test_sample, replace=False)
            data_test = data[data.id.isin(id_test)]
            data_train = data[~data.id.isin(id_test)]
            data_train["T_max"] = data_train["T_survival"]
            X_flash_train, Y_flash_train, T_train, delta_train = extract_flash_feat(
                data_train, time_indep_feat, time_dep_feat)
            data_test_truncated = truncate_data(data_test)
            X_flash_test, Y_flash_test, T_test, delta_test = extract_flash_feat(
                data_test_truncated, time_indep_feat, time_dep_feat)

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
            learner.fit(X_flash_train, Y_flash_train, T_train, delta_train,
                        asso_feat_train, K=2)
            Flash_c_index_train = learner.score(X_flash_train, Y_flash_train,
                                                T_train, delta_train,
                                                asso_feat_train)
            Flash_c_index = learner.score(X_flash_test, Y_flash_test, T_test,
                                          delta_test, asso_feat_test)
            if Flash_c_index_train < .5:
                Flash_c_index = 1 - Flash_c_index



            tmp_df = pd.DataFrame([[dataset, Flash_c_index]],
                                  columns=["Dataset", "C_index"])
            res = res.append(tmp_df)

    fontsize = 20
    _ = plt.figure(figsize=(3, 4))
    sns.set(style="white", font="STIXGeneral", context='talk',
            palette='colorblind')
    axs = sns.boxplot(x="Dataset", y="C_index", hue="Dataset",
                      data=res, width=0.6, linewidth=2)
    axs.set_ylabel(r"C_index", fontsize=fontsize, fontdict=dict(weight='bold'))
    axs.set(xticklabels=[])
    axs.set(xlabel=None)
    axs.set_ylabel("C_index", size=15)
    axs.set_ylim([0., 1.])
    handles, _ = axs.get_legend_handles_labels()
    axs.legend(handles, ["LR", "Probit"], fontsize=15)
    plt.savefig('./flash_sensity_probit.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    run()