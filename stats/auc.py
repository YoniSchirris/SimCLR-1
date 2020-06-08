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

print(f'Unbalanced AUC for {filepath} is {metrics.roc_auc_score(y_true=labels, y_score=preds)}')
print(f'F1 score (balanced) for {filepath} is {metrics.f1_score(y_true=labels, y_pred=binary_preds)}')


