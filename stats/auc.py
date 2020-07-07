## Yoni Schirris, 8 June 2020

## Script to compute patient-level AUC from a .csv file with binary predictions per tile

import pandas as pd
import numpy as np
from sklearn import metrics
import glob
import time

# filepaths = [f'../logs/pretrain/{i}' for i in range(142,142+27+1)] # directories to loop over
dirs = ['../logs/pretrain/142'] # no trailing /

data = {
    'filepath': [],
    'rocauc': [],
    'brocauc': [],
    'prauc': [],
    'bprauc': [],
    'f1': [],
    'tn': [],
    'fp': [],
    'fn': [],
    'tp': []
}

# filepath = '../logs/pretrain/105/regression_output_epoch_40_2020-07-01-15-17-44.csv'
for dir in dirs:
    print(dir)
    filepaths = glob.glob(f'{dir}/*output*.csv')
    for filepath in filepaths:
        print(filepath)
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


        tn, fp, fn, tp = metrics.confusion_matrix(labels, binary_preds).ravel()

        rocauc=metrics.roc_auc_score(y_true=labels, y_score=preds)
        brocauc=metrics.roc_auc_score(y_true=balanced_labels, y_score=balanced_preds)
        prauc=metrics.average_precision_score(labels, preds)
        bprauc=metrics.average_precision_score(balanced_labels, balanced_preds)
        f1=metrics.f1_score(y_true=labels, y_pred=binary_preds)
        

        data['filepath'].append(filepath)
        data['rocauc'].append(rocauc)
        data['brocauc'].append(brocauc)
        data['prauc'].append(prauc)
        data['bprauc'].append(bprauc)
        data['f1'].append(f1)
        data['tn'].append(tn)
        data['fp'].append(fp)
        data['fn'].append(fn)
        data['tp'].append(tp)

    # print(f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')
    # print(f'Unbalanced ROC AUC  for {filepath} is {metrics.roc_auc_score(y_true=labels, y_score=preds)}')
    # print(f'Balanced AUC for {filepath} is {metrics.roc_auc_score(y_true=balanced_labels, y_score=balanced_preds)}')
    # print(f'F1 score (balanced) for {filepath} is {metrics.f1_score(y_true=labels, y_pred=binary_preds)}')

    # print(f"Unbalanced PR AUC for {filepath} is {metrics.average_precision_score(labels, preds)}")
    # print(f'Balanced PR AUC is {metrics.average_precision_score(balanced_labels, balanced_preds)}')



pd.DataFrame(data=data).to_csv(f"metrics_{time.strftime('%B-%d-%H:%M:%S')}.csv")

