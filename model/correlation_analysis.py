def ret_corr_check(data, f1, f2, dir_path, THRESHOLD=1000):
    click = data.groupby(by=[f1, f2])['click'].sum()
    imp = data.groupby(by=[f1, f2])['id'].count()

    corr_check = pd.concat({'click':click, 'impression':imp}, axis=1)
    corr_check['CTR'] = click / imp * 100

    comb_dict = {}
    for c1, c2 in data.groupby(by=[f1, f2]).count().index:
        if c1 not in comb_dict:
            comb_dict[c1] = [c2]
        else:
            comb_dict[c1].append(c2)

    f1_list, cate_num_list = [], []
    for c1, c2_list in comb_dict.items():
        f1_list.append(c1)
        cate_num_list.append(len(c2_list))

    ctr_list = list(data.groupby([f1])['click'].sum() / data.groupby([f1])['id'].count() * 100)

    cate_num_full_list = []
    for cate_num in cate_num_list:
        cate_num_full_list += [cate_num] * cate_num

    f1_ctr = data.groupby([f1])['click'].sum() / data.groupby([f1])['id'].count() * 100
    f2_ctr = data.groupby([f2])['click'].sum() / data.groupby([f2])['id'].count() * 100

    f1_ctr_list = []
    f2_ctr_list = []
    for c1, c2 in corr_check.index:
        f1_ctr_list.append(f1_ctr.loc[c1])
        f2_ctr_list.append(f2_ctr.loc[c2])

    corr_check[f1 + ' CTR'], corr_check[f2 + ' CTR'] = f1_ctr_list, f2_ctr_list
    corr_check['cate num'] = cate_num_full_list
    corr_check[f1 + ' CTR Diff'] = corr_check['CTR'] - corr_check[f1 + ' CTR']
    corr_check[f2 + ' CTR Diff'] = corr_check['CTR'] - corr_check[f2 + ' CTR']
    corr_check['CTR Diff Sum'] = corr_check[f1 + ' CTR Diff'] + corr_check[f2 + ' CTR Diff']

    THRESHOLD = 1000
    corr_check_filtered = corr_check.loc[corr_check['impression'] > THRESHOLD]

    corr_check_filtered = corr_check_filtered[['cate num', 'click', 'impression', 'CTR', f1 + ' CTR', f2 + ' CTR', f1 + ' CTR Diff', f2 + ' CTR Diff', 'CTR Diff Sum']]
    corr_check_filtered = corr_check_filtered.round({
    	'CTR' : 2,
    	f1 + ' CTR' : 2,
    	f2 + ' CTR' : 2,
    	f1 + ' CTR Diff' : 2, 
    	f2 + ' CTR Diff' : 2, 
    	'CTR Diff Sum' : 2
    	})
    corr_check_filtered.to_csv(dir_path + f1 + '_' + f2 + '.csv')
    return corr_check_filtered