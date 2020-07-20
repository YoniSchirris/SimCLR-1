import numpy as np

from msidata.dataset_msi_features_with_patients import PreProcessedMSIFeatureDataset
from msidata.dataset_msi import PreProcessedMSIDataset
from msidata.dataset_tcga_tiles import TiledTCGADataset

def split_indices_by_patient(labels, val_split, label_col, patient_col):
    labels_msi = labels[labels[label_col]==1]
    labels_mss = labels[labels[label_col]==0]

    msi_val_patients = labels_msi.groupby(patient_col).mean().sample(frac=val_split).index
    val_msi_idc = labels_msi[labels_msi[patient_col].isin(msi_val_patients)].index.tolist()
    train_msi_idc = labels_msi[~labels_msi[patient_col].isin(msi_val_patients)].index.tolist()

    mss_val_patients = labels_mss.groupby(patient_col).mean().sample(frac=val_split).index
    val_mss_idc = labels_mss[labels_mss[patient_col].isin(mss_val_patients)].index.tolist()
    train_mss_idc = labels_mss[~labels_mss[patient_col].isin(mss_val_patients)].index.tolist()

    val_indices = val_msi_idc + val_mss_idc
    train_indices = train_msi_idc + train_mss_idc

    return train_indices, val_indices


def split_indices(labels, val_split, label_col):
    labels_msi = labels[labels[label_col]==1]
    val_msi_idc = labels_msi.sample(frac=val_split).index.tolist()
    train_msi_idc = labels_msi[~labels_msi.index.isin(val_msi_idc)].index.tolist()
    
    labels_mss = labels[labels[label_col]==0]
    val_mss_idc = labels_mss.sample(frac=val_split).index.tolist()
    train_mss_idc = labels_mss[~labels_mss.index.isin(val_mss_idc)].index.tolist()

    val_indices = val_mss_idc + val_msi_idc
    train_indices = train_mss_idc + train_msi_idc

    return train_indices, val_indices


def get_train_val_indices(train_dataset, val_split):
    assert (train_dataset.labels is not None), "The dataset has no labels."
    labels = train_dataset.labels

    if isinstance(train_dataset, PreProcessedMSIFeatureDataset) or isinstance(train_dataset, PreProcessedMSIDataset):
        label_col = 'label'
        patient_col = 'patient_id'
    elif isinstance(train_dataset, TiledTCGADataset):
        label_col = 'msi'
        patient_col = 'case'

    if isinstance(train_dataset, PreProcessedMSIFeatureDataset) or isinstance(train_dataset, TiledTCGADataset):
        if train_dataset.sampling_strategy == 'patient' and train_dataset.tensor_per_patient:
            train_indices, val_indices = split_indices(labels, val_split, label_col)
        else:
            train_indices, val_indices = split_indices_by_patient(labels, val_split, label_col, patient_col)
    
    elif isinstance(train_dataset, PreProcessedMSIDataset):
        train_indices, val_indices = split_indices_by_patient(labels, val_split, label_col, patient_col)
    else:
        raise NotImplementedError

    train_patients = len(labels.loc[train_indices].groupby(patient_col).mean().index)
    val_patients = len(labels.loc[val_indices].groupby(patient_col).mean().index)
    all_patients = len(labels.groupby(patient_col).mean().index)

    assert (abs(val_patients - (val_split*all_patients)) <= 1), f"Something went wrong in val (n={val_patients} patients)-train (n={train_patients} patients) split"
    assert (abs(train_patients - ((1-val_split)*all_patients)) <= 1), f"Something went wrong in val (n={val_patients} patients)-train (n={train_patients} patients) split"
    assert (len(set(labels.loc[train_indices][patient_col].unique().tolist() + labels.loc[val_indices][patient_col].unique().tolist())) == len(labels.loc[train_indices][patient_col].unique().tolist() + labels.loc[val_indices][patient_col].unique().tolist())), "There's a patient leak!"

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    return train_indices, val_indices
    # get the .csv file
    # split by patient
    # split with similar distribution of the label we look at
    # then get all the indices of all the tiles of the patients
    # create a dataloader with that split

