# Library setup
import os
import pandas as pd
import numpy as np
from tsfresh import extract_features as extract_rep_features
from sksurv.linear_model import CoxPHSurvivalAnalysis
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.model_selection import ShuffleSplit
from prettytable import PrettyTable

from flash.base.base import normalize
from flash.inference import ext_EM
from flash.base.base import feat_representation_extraction
from competing_methods.all_model import load_data, extract_flash_feat

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', ConvergenceWarning)
np.random.seed(0)

def run():
	# loading data
	test_size = .3
	params_path = "flash/params/"
	dataset = "PBCseq"
	data, time_dep_feat, time_indep_feat = load_data(data_name=dataset)
	data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
	X_first = data.drop_duplicates(subset = ['id'], keep = 'first')[["id"] + time_dep_feat]
	time_indep_feat_sub = [x + "_1st" for x in time_dep_feat]
	X_first.columns = ["id"] + time_indep_feat_sub
	time_indep_feat = time_indep_feat + time_indep_feat_sub
	data = data.merge(X_first, how='right', on='id')
	id_list = np.unique(data["id"])
	nb_test_sample = int(test_size * len(id_list))
	id_test = np.random.choice(id_list, size=nb_test_sample, replace=False)
	data_train = data[~data.id.isin(id_test)]


	# data preprocessing
	X, Y, T, delta = extract_flash_feat(data, time_indep_feat, time_dep_feat)
	X_train, Y_train, T_train, delta_train = extract_flash_feat(data_train, time_indep_feat, time_dep_feat)
	n_long_features = len(time_dep_feat)
	features_timedep_names = time_dep_feat
	X_train = normalize(X_train)
	Y_train[features_timedep_names] = normalize(Y_train[features_timedep_names].values)

	# feature screening
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
	third_score_threshold = sorted_third_score[9]
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


	# ectracted features
	T_u = np.unique(T_train[delta_train == 1])
	asso_feat_train, _ = feat_representation_extraction(Y_train, n_long_features, T_u, final_selected_fc_parameters)
	# training
	with open(params_path + dataset + '/Flash_penalties', 'rb') as f:
		flash_pens = np.load(f)
	l_pen_EN, l_pen_SGL = flash_pens[0], flash_pens[1]
	fixed_effect_time_order = 1
	learner = ext_EM(fixed_effect_time_order=fixed_effect_time_order,
	                 max_iter=80, print_every=1, l_pen_SGL=l_pen_SGL,
	                 l_pen_EN=l_pen_EN, initialize=True,
	                 fc_parameters=final_selected_fc_parameters)
	learner.fit(X_train, Y_train, T_train, delta_train, asso_feat_train, K=2)

	# run boostrap to get standard deviation
	gamma_0_est_support = (learner.theta["gamma_0"].copy() != 0).astype(int)
	gamma_1_est_support = (learner.theta["gamma_1"].copy() != 0).astype(int)
	xi_est_support = (learner.theta["xi"][:, 1] != 0).astype(int)

	gamma_CIs = []
	xi_CIs = []
	n_runs = 10
	for k in range(n_runs):
		# Split data into training and test sets
		test_size = .3  # proportion of data used for testing
		rs = ShuffleSplit(n_splits=1, test_size=test_size, random_state=0 + k)

		data_index = np.unique(Y.id.values)
		for train_index, test_index in rs.split(X):
			train_index = np.sort(train_index)
			data_index_train = data_index[train_index]
			X_train = X[train_index]
			Y_train = Y[Y.id.isin(data_index_train)]
			T_train = T[train_index]
			delta_train = delta[train_index]


		T_u = np.unique(T_train[delta_train == 1])
		n_long_features = len(time_dep_feat)
		asso_feat_train, _ = feat_representation_extraction(Y_train, n_long_features,
		                                                    T_u, final_selected_fc_parameters)

		l_pen_EN, l_pen_SGL = 5e-2, 2e-1
		learner = ext_EM(max_iter=40, print_every=1, eta_elastic_net=1.,
		                 eta_sp_gp_l1=1., l_pen_SGL=l_pen_SGL,
		                 l_pen_EN=l_pen_EN, initialize=True,
		                 fc_parameters=final_selected_fc_parameters)
		learner.fit(X_train, Y_train, T_train, delta_train, asso_feat_train, K=2,
		            xi_support=xi_est_support,
		            gamma_supports=[gamma_0_est_support.flatten(),
		                      gamma_1_est_support.flatten()])

		xi_CIs.append(learner.theta["xi_1"] * xi_est_support.flatten())
		gamma_CIs.append([learner.theta["gamma_0"].flatten() * gamma_0_est_support.flatten(),
		                  learner.theta["gamma_1"].flatten() * gamma_1_est_support.flatten()])

	gamma_0_cis = np.linalg.norm(np.array(gamma_CIs)[:, 0].reshape((n_runs, n_long_features, -1)), axis=2)
	gamma_1_cis = np.linalg.norm(np.array(gamma_CIs)[:, 1].reshape((n_runs, n_long_features, -1)), axis=2)

	# results
	mean_gamma_0 = gamma_0_cis.mean(axis=0)
	std_gamma_0 = gamma_0_cis.std(axis=0)
	mean_gamma_1 = gamma_1_cis.mean(axis=0)
	std_gamma_1 = gamma_1_cis.std(axis=0)
	mean_xi = np.array(xi_CIs).mean(axis=0)
	std_xi = np.array(xi_CIs).std(axis=0)

	coefs = PrettyTable(["Features", "Coefficient"])
	coefs.align["Features"] = "c"  # Center align feature names
	coefs.padding_width = 1  # One space between column edges and contents (default)
	for j in range(len(time_indep_feat)):
		if j != len(time_indep_feat) - 1:
			if mean_xi[j] != 0:
				coefs.add_row([time_indep_feat[j], str(round(mean_xi[j], 4)) + " ± " + str(round(std_xi[j], 4))])
			else:
				coefs.add_row([time_indep_feat[j], "0.0"])
		else:
			if mean_xi[j] != 0:
				coefs.add_row([time_indep_feat[j], str(round(mean_xi[j], 4))  + " ± " + str(round(std_xi[j], 4))], divider=True)
			else:
				coefs.add_row([time_indep_feat[j], "0.0"])

	for j in range(len(time_dep_feat)):
		if j != len(time_dep_feat) - 1:
			if mean_gamma_0[j] != 0:
				coefs.add_row([time_dep_feat[j], str(round(mean_gamma_0[j], 4)) + " ± " + str(round(std_gamma_0[j], 4))])
			else:
				coefs.add_row([time_dep_feat[j], "0.0"])
		else:
			if mean_gamma_0[j] != 0:
				coefs.add_row([time_dep_feat[j], str(round(mean_gamma_0[j], 4)) + " ± " + str(round(std_gamma_0[j], 4))], divider=True)
			else:
				coefs.add_row([time_dep_feat[j], "0.0"], divider=True)

	for j in range(len(time_dep_feat)):
		if j != len(time_dep_feat) - 1:
			if mean_gamma_1[j] != 0:
				coefs.add_row([time_dep_feat[j], str(round(mean_gamma_1[j], 4)) + " ± " + str(round(std_gamma_1[j], 4))])
			else:
				coefs.add_row([time_dep_feat[j], "0.0"])
		else:
			if mean_gamma_1[j] != 0:
				coefs.add_row([time_dep_feat[j], str(round(mean_gamma_1[j], 4)) + " ± " + str(round(std_gamma_1[j], 4))], divider=True)
			else:
				coefs.add_row([time_dep_feat[j], "0.0"], divider=True)

	coefs_ = coefs.get_string()
	if not os.path.exists("results"):
		os.mkdir("results")
	with open('results/PBC_coefs.txt', 'w') as f:
		f.write(coefs_)

if __name__ == "__main__":
    run()
