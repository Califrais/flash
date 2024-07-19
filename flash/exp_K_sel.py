# Library setup
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from flash.base.base import feat_representation_extraction
from flash.inference import ext_EM
from competing_methods.all_model import load_data, extract_flash_feat, truncate_data

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', ConvergenceWarning)

np.random.seed(0)

def run():
	# loading data
	test_size = .3
	dataset = "PBCseq"
	data, time_dep_feat, time_indep_feat = load_data(data_name=dataset)
	data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
	id_list = np.unique(data["id"])
	nb_test_sample = int(test_size * len(id_list))
	id_test = np.random.choice(id_list, size=nb_test_sample, replace=False)
	data_train = data[~data.id.isin(id_test)]
	data_test = data[data.id.isin(id_test)]
	data_test_truncated = truncate_data(data_test)
	n_long_features = len(time_dep_feat)

	# data preprocessing
	X_train, Y_train, T_train, delta_train = extract_flash_feat(data_train, time_indep_feat, time_dep_feat)
	final_selected_fc_parameters = {
	    'sum_values': None,
	    'abs_energy': None,
	    'standard_deviation': None,
	    'time_reversal_asymmetry_statistic': [{'lag': 1}]
	}
	T_u = np.unique(T_train[delta_train == 1])
	asso_feat_train, _ = feat_representation_extraction(Y_train, n_long_features, T_u, final_selected_fc_parameters)

	# Training with best cross-validation params
	l_pen_EN, l_pen_SGL = 1e-2, 2e-1
	fixed_effect_time_order = 1
	K_list = [2, 3, 4, 5, 6, 7, 8, 9]
	BIC_list = []
	N = X_train.shape[0]
	for k in K_list:
		learner = ext_EM(fixed_effect_time_order=fixed_effect_time_order,
		                 max_iter=60,
		                 print_every=1, l_pen_SGL=l_pen_SGL, l_pen_EN=l_pen_EN,
		                 initialize=True,
		                 fc_parameters=final_selected_fc_parameters)

		learner.fit(X_train, Y_train, T_train, delta_train, asso_feat_train, k)

		LL = -learner.history.values["obj"][-1]
		BIC = -2 * N * LL + np.log(N) * k
		BIC_list.append(BIC)

	# visualization
	sns.set(style='whitegrid',font="STIXGeneral",context='talk',palette='colorblind')
	f,ax=plt.subplots(figsize=(12,6))
	[x.set_linewidth(2) for x in ax.spines.values()]
	[x.set_edgecolor('black') for x in ax.spines.values()]
	ax.plot(K_list, BIC_list, linewidth=5, label='BIC',c='blue')
	plt.setp(ax.patches, linewidth=0)
	ax.set_ylim([13000, 13500])
	ax.set_xlabel("Number of latent groups (K)")
	ax.set_ylabel("BIC")

	plt.tight_layout()
	plt.savefig('./BIC_K_chosen.pdf')
	plt.show()

if __name__ == "__main__":
    run()