## Yoni Schirris, 8 June 2020

## Script to compute patient-level AUC from a .csv file with binary predictions per tile

import pandas as pd
import numpy as np
from sklearn import metrics


filepath = ''

df = pd.read_csv(filepath)

dfgroup = df.groupby(['patient']).mean()

dfgroup['count'] = df.groupby('patient').count()['labels'] # can be used to discard patients with very few tiles

labels = dfgroup['labels'].values
preds = dfgroup['preds'].values

binary_preds = np.zeros(preds.shape)

binary_preds[preds > 0.5] = 1

df_msi = dfgroup[dfgroup['labels']==0]
df_mss = dfgroup[dfgroup['labels']==1]

min_size = min(len(df_msi.index), len(df_mss.index))

df_msi_sub = df_msi.sample(min_size)
df_mss_sub = df_mss.sample(min_size)

dfgroup_balanced = df_msi_sub.append(df_mss)

balanced_labels = dfgroup_balanced['labels'].values
balanced_preds = dfgroup_balanced['preds'].values

prec, rec, thresholds = metrics.precision_recall_curve(y_true=labels, y_true=preds)

print(f'Unbalanced ROC AUC  for {filepath} is {metrics.roc_auc_score(y_true=labels, y_score=preds)}')
print(f'Balanced AUC for {filepath} is {metrics.roc_auc_score(y_true=balanced_labels, y_score=balanced_preds)}')
print(f"Unbalanced PR AUC for {filepath} is {metric.auc(prec, rec)}")
print(f'F1 score (balanced) for {filepath} is {metrics.f1_score(y_true=labels, y_pred=binary_preds)}')


