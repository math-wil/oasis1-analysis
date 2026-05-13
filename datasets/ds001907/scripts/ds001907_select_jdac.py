import pandas as pd

df = pd.read_csv('datasets/ds001907/results/ds001907_scores_ses1.csv', index_col=0)[['sub','motion']].drop_duplicates('sub').sort_values('motion').reset_index(drop=True)
n = len(df)
idx = sorted(set(list(range(9)) + list(range(n//2-3, n//2+4)) + list(range(n-9, n))))
sel = df.loc[idx].copy()
sel['stratum'] = 'mid'
sel.loc[sel.index[:9], 'stratum'] = 'low'
sel.loc[sel.index[-9:], 'stratum'] = 'high'
sel['t1w_path'] = sel['sub'].apply(lambda s: f'/home/av62870@ens.ad.etsmtl.ca/Documents/ds001907/derivatives/{s}/ses-1/anat/{s}_ses-1_T1wbrain.nii.gz')
sel.to_csv('datasets/ds001907/results/ds001907_jdac_subjects.csv', index=False)
print(sel[['sub','motion','stratum']].to_string(index=False))
print(f'Total : {len(sel)} sujets')
