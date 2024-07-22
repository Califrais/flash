# Library setup
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', ConvergenceWarning)

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sksurv.linear_model import CoxPHSurvivalAnalysis
from tsfresh import extract_features as extract_rep_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

from flash.base.base import normalize
from flash.inference import ext_EM
from flash.base.utils import visualize_vect_per_group
from competing_methods.all_model import extract_flash_feat, truncate_data

np.random.seed(0)


def get_NASAdata(nb_file):
    # get data from file and pre process it (normalization and convert to pandas)
    features_col_name = ['setting1', 'setting2', 'setting3'] + ["s" + str(i) for i in range(1, 22)]
    col_names = ['id', 'T_long'] + features_col_name
    dataset_train = pd.read_csv('competing_methods/NASA_data/train_FD00{}.txt'.format(nb_file),
                                sep='\s+', header=None, names=col_names)
    dataset_test = pd.read_csv('competing_methods/NASA_data/test_FD00{}.txt'.format(nb_file),
                               sep='\s+', header=None, names=col_names)

    dataset_train['T_survival'] = dataset_train.groupby(['id'])[
        'T_long'].transform(max)
    dataset_train['delta'] = 1
    dataset_test['T_survival'] = dataset_test.groupby(['id'])[
        'T_long'].transform(max)
    dataset_test['delta'] = 0
    dataset_test["id"] = dataset_test["id"] + max(dataset_train["id"])
    data = pd.concat([dataset_train, dataset_test])
    relevant_features_col_name = []
    for col in features_col_name:
        if not (len(dataset_train[col].unique()) < 10):
            relevant_features_col_name.append(col)
    data = data[
        ["id", "T_long", "T_survival", "delta"] + relevant_features_col_name]

    time_dep_features_col_name = relevant_features_col_name
    time_indep_features_col_name = [feat + "_time_indep" for feat in
                                    relevant_features_col_name]
    data_time_indep = data.groupby('id')[
        relevant_features_col_name].first().reset_index()
    data_time_indep.columns = ["id"] + time_indep_features_col_name

    data = pd.merge(data, data_time_indep, on="id", how="left")
    data[time_dep_features_col_name] = data[time_dep_features_col_name].values - \
                                       data[time_indep_features_col_name].values
    data[time_dep_features_col_name + time_indep_features_col_name] = normalize(
        data[time_dep_features_col_name + time_indep_features_col_name])

    id_list = np.unique(data["id"])
    n_samples = len(id_list)
    for i in range(n_samples):
        data_i = data[(data["id"] == id_list[i])]
        n_i = data_i.shape[0]
        times_i = data_i["T_long"].values
        if n_i > 10:
            times_i = np.sort(np.random.choice(times_i, size=15,
                                               replace=False))
            n_i = 10
        data = data[(data["id"] != id_list[i]) |
                    ((data["id"] == id_list[i]) & (
                        data["T_long"].isin(list(times_i))))]

    return data, time_dep_features_col_name, time_indep_features_col_name

def run():

    # loading data
    test_size = .3
    data, time_dep_feat, time_indep_feat = get_NASAdata(1)
    data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]

    # data preprocessing
    id_list = np.unique(data["id"])
    nb_test_sample = int(test_size * len(id_list))
    n_long_features = len(time_dep_feat)
    time_indep_feat
    id_test = np.random.choice(id_list, size=nb_test_sample, replace=False)
    data_train = data[~data.id.isin(id_test)]
    data_test = data[data.id.isin(id_test)]
    data_test_truncated = truncate_data(data_test)
    X_train, Y_train, T_train, delta_train = extract_flash_feat(data_train,
                                                                time_indep_feat,
                                                                time_dep_feat)
    X_test, Y_test, _, _ = extract_flash_feat(data_test_truncated,
                                                            time_indep_feat,
                                                            time_dep_feat)

    # Feature screening
    fc_parameters = ComprehensiveFCParameters()
    ext_feat = extract_rep_features(Y_train, column_id="id",
                                    column_sort="T_long",
                                    default_fc_parameters=fc_parameters,
                                    impute_function=None,
                                    disable_progressbar=True
                                    )
    y = np.zeros(len(T_train),
                 dtype={'names': ('indicator', 'time'), 'formats': ('?', 'f8')})
    y['indicator'] = delta_train
    y['time'] = T_train

    first_scores = {}
    idx = 0
    asso_feat_names = ext_feat.columns
    n_asso_feat = len(asso_feat_names) // n_long_features
    for key, val in fc_parameters.items():
        if val is not None:
            tmp = fc_parameters[key]
            score_tmp = []
            for val in tmp:
                feat = {key: [val]}
                asso_feat = ext_feat[asso_feat_names[idx:-1:n_asso_feat]]
                try:
                    model = CoxPHSurvivalAnalysis().fit(asso_feat, y)
                except:
                    score_tmp.append(0.5)
                else:
                    score_tmp_ = model.score(asso_feat, y)
                    score_tmp.append(max(score_tmp_, 1 - score_tmp_))
                idx += 1
            first_scores[key] = score_tmp
        else:
            feat = {key: val}
            asso_feat = ext_feat[asso_feat_names[idx:-1:n_asso_feat]]
            try:
                model = CoxPHSurvivalAnalysis().fit(asso_feat, y)
            except:
                first_scores[key] = 0.5
            else:
                score_tmp_ = model.score(asso_feat, y)
                first_scores[key] = max(score_tmp_, 1 - score_tmp_)
            idx += 1

    flat_first_score = []
    for item in list(first_scores.values()):
        if isinstance(item, list):
            flat_first_score += item
        else:
            flat_first_score += [item]

    sorted_first_score = -np.sort(-np.array(flat_first_score))
    first_score_threshold = sorted_first_score[20]
    first_selected_fc_parameters = {}
    second_scores = {}
    for key, val in fc_parameters.items():
        score = first_scores[key]
        if val is not None:
            tmp = []
            score_tmp = []
            for idx in range(len(score)):
                if score[idx] >= first_score_threshold:
                    tmp.append(val[idx])
                    score_tmp.append(score[idx])
            if tmp:
                first_selected_fc_parameters[key] = tmp
                second_scores[key] = score_tmp

        else:
            if score >= first_score_threshold:
                first_selected_fc_parameters[key] = val
                second_scores[key] = score

    # Remove correlation
    ext_feat = extract_rep_features(Y_train, column_id="id",
                                    column_sort="T_long",
                                    default_fc_parameters=first_selected_fc_parameters,
                                    impute_function=None,
                                    disable_progressbar=True
                                    )
    nb_extracted_feat = len(ext_feat.columns) // n_long_features
    n_train_sample = ext_feat.shape[0]
    ext_feat_df = ext_feat.values.reshape(n_train_sample, n_long_features,
                                          nb_extracted_feat)
    ext_feat_df = ext_feat_df.T.reshape(nb_extracted_feat, -1).T
    cor_matrix = pd.DataFrame(ext_feat_df).corr().abs()

    upper_tri = cor_matrix.where(
        np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_columns = [column for column in upper_tri.columns if
                    any(upper_tri[column] > 0.9)]

    col_idx = 0
    second_selected_fc_parameters = {}
    third_scores = {}
    for key, val in first_selected_fc_parameters.items():
        score = second_scores[key]
        if val is not None:
            tmp = []
            score_tmp = []
            for idx in range(len(val)):
                if col_idx not in drop_columns:
                    tmp.append(val[idx])
                    score_tmp.append(score[idx])
                col_idx += 1
            if tmp:
                second_selected_fc_parameters[key] = tmp
                third_scores[key] = score_tmp
        else:
            if col_idx not in drop_columns:
                second_selected_fc_parameters[key] = val
                third_scores[key] = score
            col_idx += 1
    flat_third_score = []
    for item in list(third_scores.values()):
        if isinstance(item, list):
            flat_third_score += item
        else:
            flat_third_score += [item]

    sorted_third_score = -np.sort(-np.array(flat_third_score))
    third_score_threshold = sorted_third_score[8]
    final_selected_fc_parameters = {}
    final_scores = {}
    for key, val in second_selected_fc_parameters.items():
        score = third_scores[key]
        if val is not None:
            tmp = []
            score_tmp = []
            for idx in range(len(score)):
                if score[idx] >= third_score_threshold:
                    tmp.append(val[idx])
                    score_tmp.append(score[idx])
            if tmp:
                final_selected_fc_parameters[key] = tmp
                final_scores[key] = score_tmp

        else:
            if score >= third_score_threshold:
                final_selected_fc_parameters[key] = val
                final_scores[key] = score

    # Model fitting
    l_pen_EN, l_pen_SGL = 2e-2, 2e-1
    learner = ext_EM(max_iter=10, print_every=1, l_pen_SGL=l_pen_SGL,
                     l_pen_EN=l_pen_EN, initialize=False,
                     fc_parameters=final_selected_fc_parameters, verbose=False)
    learner.fit(X_train, Y_train, T_train, delta_train, K=2)

    # visualization
    marker_pred = learner.predict_marker_sample(X_test, Y_test)
    fontsize=10

    mosaic = """
        .a...
        .a.d.
        .b.d.
        .c.d.
        .c...
        """
    fig = plt.figure(figsize=(16,5))
    axs = fig.subplot_mosaic(mosaic,
                             gridspec_kw={
                                     # set the height ratios between the rows
        "height_ratios": [.2, .8, 1, .8, .2],
        # set the width ratios between the columns
        "width_ratios":[.2, 6., .2, 5., .2],
        })

    sns.set(style='whitegrid',font="STIXGeneral",context='talk',palette='colorblind')
    [x.set_linewidth(2) for x in axs["a"].spines.values()]
    [x.set_edgecolor('black') for x in axs["a"].spines.values()]
    [x.set_linewidth(2) for x in axs["b"].spines.values()]
    [x.set_edgecolor('black') for x in axs["b"].spines.values()]
    [x.set_linewidth(2) for x in axs["c"].spines.values()]
    [x.set_edgecolor('black') for x in axs["c"].spines.values()]
    [x.set_linewidth(2) for x in axs["d"].spines.values()]
    [x.set_edgecolor('black') for x in axs["d"].spines.values()]


    xi_est = learner.theta["xi"][:, 1]
    xi_est_support = (xi_est != 0).astype(int)
    ax = axs['a']
    markerline, stemline, _, = ax.stem((np.arange(len(xi_est)) + .5).tolist(), xi_est_support,
                                              linefmt='r-', markerfmt='ro', label= r"$\hat \xi$")
    plt.setp(stemline, linewidth = 2, color="red")
    plt.setp(markerline, markersize = 4, color="red")
    ax.set_xlim([-1, len(xi_est) + 1])
    ax.legend(fontsize=fontsize)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim(-0.1, 3)
    plt.tick_params(axis='x', bottom=False, labelbottom=False)

    L = n_long_features
    gamma_1_est_support = (learner.theta["gamma_0"].copy() != 0).astype(int)
    ax = axs['b']
    markerline, stemline, _, = ax.stem((np.arange(len(gamma_1_est_support))).tolist(), gamma_1_est_support,
                                              linefmt='r-', markerfmt='ro', label= r"$\hat \gamma_1$")
    plt.setp(stemline, linewidth = 2, color="red")
    plt.setp(markerline, markersize = 4, color="red")
    ax.set_xlim([-5, len(gamma_1_est_support) + 5])
    ax.legend(fontsize=fontsize)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim(-0.05, 3)
    plt.tick_params(axis='x', bottom=False, labelbottom=False)
    visualize_vect_per_group(gamma_1_est_support, L, axs['b'], plot_area=False)
    axs['b'].legend(prop={'size': 10})

    gamma_2_est_support = (learner.theta["gamma_1"].copy() != 0).astype(int)
    ax = axs['c']
    markerline, stemline, baseline, = ax.stem((np.arange(len(gamma_2_est_support))).tolist(), gamma_2_est_support,
                                              linefmt='r-', markerfmt='ro', label= r"$\hat \gamma_2$")
    plt.setp(stemline, linewidth = 2, color="red")
    plt.setp(markerline, markersize = 4, color="red")
    ax.set_xlim([-5, len(gamma_1_est_support) + 5])
    ax.legend(fontsize=fontsize)
    ax.set_yticks([])
    ax.set_ylim(-0.05, 3)
    axs['c'].grid(False)
    plt.tick_params(axis='x', bottom=False, labelbottom=False)
    visualize_vect_per_group(gamma_1_est_support, L, axs['c'], plot_area=False)
    axs['c'].legend(prop={'size': 10})

    id_list = list(np.unique(Y_test.id.values))
    for i in range(len(marker_pred)):
        t_ = Y_test[(Y_test["id"] == id_list[i])]["T_long"].values
        if marker_pred[i][-1] < .5:
            axs['d'].plot(t_, marker_pred[i], 'blue', alpha=1.)
        else:
            axs['d'].plot(t_, marker_pred[i], 'red', alpha=1.)
    axs['d'].set_xlabel('Time $t$', size=15)
    axs['d'].set_ylabel('Predictive marker $\hat\mathcal{R}_i(t)$', size=15)
    axs['d'].set_title('Evolution curve of predictive marker', size=15)
    axs['d'].legend(['High-risk', 'Low-risk'])
    leg = axs['d'].get_legend()
    leg.legendHandles[0].set_color('red')
    leg.legendHandles[1].set_color('blue')
    axs['d'].axhline(y = 0.5, color = 'grey', linestyle = '-.', linewidth = 2)
    axs['d'].grid(False)
    if not os.path.exists("results"):
        os.mkdir("results")
    plt.savefig('results/NASA_perf.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run()