

def ret_feature_correlation():
	DIR_PATH = '/content/drive/MyDrive/ctr/data/feature_correlation/corr_check/'
	f_list = [
	    'site_id', 'site_domain', 'site_category',
	    'app_id', 'app_domain', 'app_category',
	    'device_model', 'device_type',
	    'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'
	]
	IMP_THRESHOLD_LIST = [1000, 2500, 5000, 10000, 20000]
	CTR_THRESHOLD_LIST = [1, 2.5, 5, 7.5, 10]


	feature_comb_list = []
	total_imp_list = []
	diff_above_num_list = []
	diff_below_num_list = []
	diff_none_num_list = []
	diff_above_ratio_list = []
	diff_below_ratio_list = []
	diff_none_ratio_list = []
	imp_threhold_list = []
	ctr_threshold_list = []

	f_num = len(f_list)
	for idx1 in tqdm(range(f_num)):
		for idx2 in range(idx1+1, f_num):
			f1, f2 = f_list[idx1], f_list[idx2]
			data = pd.read_csv(DIR_PATH + f1 + '_' + f2 + '.csv')
			for IMP_THRESHOLD in IMP_THRESHOLD_LIST:
				for CTR_THRESHOLD in CTR_THRESHOLD_LIST:
					target = data.loc[data['impression'] >= IMP_THRESHOLD]
					total_imp = len(target)
					diff_above = target.loc[(target[f1 + ' CTR Diff'] >= CTR_THRESHOLD) & (target[f2 + ' CTR Diff'] >= CTR_THRESHOLD)]
					diff_above_num = len(diff_above)
					diff_below = target.loc[(target[f1 + ' CTR Diff'] <= -CTR_THRESHOLD) & (target[f2 + ' CTR Diff'] <= -CTR_THRESHOLD)]
					diff_below_num = len(diff_below)
					diff_none = target.loc[
					    (target[f1 + ' CTR Diff'] > -CTR_THRESHOLD) & (target[f1 + ' CTR Diff'] < CTR_THRESHOLD) &
					    (target[f2 + ' CTR Diff'] > -CTR_THRESHOLD) & (target[f2 + ' CTR Diff'] < CTR_THRESHOLD)    
					]
					diff_none_num = len(diff_none)

					feature_comb_list.append(f1 + '-' + f2)
					total_imp_list.append(total_imp)
					diff_above_num_list.append(diff_above_num)
					diff_below_num_list.append(diff_below_num)
					diff_none_num_list.append(diff_none_num)
					if total_imp != 0:
						diff_above_ratio_list.append(diff_above_num / total_imp * 100)
						diff_below_ratio_list.append(diff_below_num / total_imp * 100)
						diff_none_ratio_list.append(diff_none_num / total_imp * 100)
					else:
						diff_above_ratio_list.append(0)
						diff_below_ratio_list.append(0)
						diff_none_ratio_list.append(0)
					imp_threhold_list.append(IMP_THRESHOLD)
					ctr_threshold_list.append(CTR_THRESHOLD)

	df = pd.DataFrame(
	    { 'feature' : feature_comb_list,
	     'total imp' : total_imp_list,
	      'diff above num' : diff_above_num_list,
	      'diff below num' : diff_below_num_list,
	      'diff none num' : diff_none_num_list,
	      'diff above ratio' : diff_above_ratio_list,
	      'diff below ratio' : diff_below_ratio_list,
	      'diff none ratio' : diff_none_ratio_list,
	      'imp threshold' : imp_threhold_list,
	      'ctr threshold' : ctr_threshold_list
	      })

	return df