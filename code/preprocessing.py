import pandas as pd
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from sklearn.preprocessing import StandardScaler
import random

import cliffs as cliffs_van_tilborg_et_al
import data_prep as data_prep_van_tilborg_et_al

np.random.seed(12)
random.seed(12)
torch.manual_seed(12)
torch.random.manual_seed(12)


def smiles_to_ecfp(smiles, rad, nB):
    """
    Computes the Extended Connectivity Fingerprint (ECFP) representation of a molecule given its SMILES string.

    Parameters:
        smiles (string): SMILES string of a molecule.
        rad (int): The radius used for the ECFP computation (determines the size of the "neighborhood" considered).
        nB (int): Length (number of Bits) the resulting ECFP should have.

    Returns:
        np.ndarray: Extended Connectivity Fingerprint representation.
    """

    mol = Chem.MolFromSmiles(smiles)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=rad, nBits=nB)
    return np.array(list(ecfp), dtype=np.float32)


def initial_preprocessing(path, threshold, radius, nBits):
    """
    Reads the dataframe of the specified dataset, performs binarization on the activity labels, relabels activity cliff molecules and 
        computes the Extended Connectivity Fingerprint representation of the molecules.

    Parameters:
        path (string): Path to the dataset.
        threshold (int or float): Threshold for binarization of activity levels.
        rad (int): The radius used for the ECFP computation (determines the size of the "neighborhood" considered).
        nB (int): Length (number of Bits) the resulting ECFP should have.

    Returns:
        pandas.DataFrame: Preprocessed dataframe.
    """

    df = pd.read_csv(path)
    df['active'] = df['y [pEC50/pKi]'].apply(
        lambda x: 1 if x > threshold else 0)

    smiles_list = list(df['smiles'])
    activity_list = list(df['active'])
    cliffs = cliffs_van_tilborg_et_al.ActivityCliffs(
        smiles=smiles_list, bioactivity=activity_list)
    cliffs_binary = cliffs.get_cliff_molecules(return_smiles=False)
    df['cliff_mol_binary'] = cliffs_binary

    df['ecfp'] = df['smiles'].apply(
        lambda x: smiles_to_ecfp(x, rad=radius, nB=nBits))

    return df


def split_data(df):
    """
    Splits the dataframe into a training, validation and test set.

    Parameters:
        df (pandas.DataFrame): Preprocessed dataframe.

    Returns:
        tuple:
            A tuple containing:
                - pandas.DataFrame: Training set.
                - pandas.DataFrame: Validation set.
                - pandas.DataFrame: Test set.
    """
    df_train_val = df[df['split'] == 'train']
    df_test = df[df['split'] == 'test']

    df_train_val_split = data_prep_van_tilborg_et_al.split_data(smiles=list(df_train_val['smiles']),
                                                                bioactivity=np.array(
                                                                    df_train_val['exp_mean [nM]']),
                                                                test_size=0.2)

    df_train_val['split'] = list(df_train_val_split['split'])

    df_train = df_train_val[df_train_val['split'] == 'train']
    df_val = df_train_val[df_train_val['split'] == 'test']

    return df_train, df_val, df_test


def reset_indices(df_train, df_val, df_test):
    """
    Resets the indices of the given training, validation and test sets to go from 0 to n respectively.

    Parameters:
        tuple:
            A tuple containing:
                - pandas.DataFrame: Train set.
                - pandas.DataFrame: Validation set.
                - pandas.DataFrame: Test set.

    Returns:
        tuple:
            A tuple containing:
                - pandas.DataFrame: Train set (reindexed).
                - pandas.DataFrame: Validation set (reindexed).
                - pandas.DataFrame: Test set (reindexed).
    """
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    return df_train, df_val, df_test


def normalize_ecfps(df_train, df_val, df_test):
    """
    Normalizes the ECFPs with Z-score normalization.

    Parameters:
        tuple:
            A tuple containing:
                - pandas.DataFrame: Train set.
                - pandas.DataFrame: Validation set.
                - pandas.DataFrame: Test set.

    Returns:
        tuple:
            A tuple containing:
                - pandas.DataFrame: Train set (normalized).
                - pandas.DataFrame: Validation set (normalized).
                - pandas.DataFrame: Test set (normalized).
    """

    train_ecfps = np.vstack(df_train['ecfp'].values)
    scaler = StandardScaler()
    scaler.fit(train_ecfps)

    def normalize_array(ecfp_array):
        return scaler.transform([ecfp_array])[0]

    df_train['ecfp'] = df_train['ecfp'].apply(normalize_array)
    df_val['ecfp'] = df_val['ecfp'].apply(normalize_array)
    df_test['ecfp'] = df_test['ecfp'].apply(normalize_array)

    return df_train, df_val, df_test


def preprocess_data(perform_add_preprocessing, path):
    """
    Performs preprocessing on the dataset.

    Parameters:
        perform_add_preprocessing (bool): True if dataset should be preprocessed (binarization etc.), False if 
            already preprocessed dataframes should be loaded from the data folder.
        path (string): Path to the dataset.
        rad (int): The radius used for the ECFP computation (determines the size of the "neighborhood" considered).
        nB (int): Length (number of Bits) the resulting ECFP should have.

    Returns:
        tuple:
            A tuple containing:
                - pandas.DataFrame: Train set.
                - pandas.DataFrame: Validation set.
                - pandas.DataFrame: Test set.
    """

    if perform_add_preprocessing:
        df = initial_preprocessing(
            path=path, threshold=7, radius=3, nBits=2048)
        df_train, df_val, df_test = split_data(df)
        df_train, df_val, df_test = normalize_ecfps(df_train, df_val, df_test)
        df_train, df_val, df_test = reset_indices(df_train, df_val, df_test)

        similarities = cliffs_van_tilborg_et_al.moleculeace_similarity(
            df_train['smiles'])  # no self-similarity, i. e. similarity[i, i] == 0

        df_train['similar_molecules'] = [
            np.where(similarities[i] == 1)[0] for i in range(similarities.shape[0])
        ]

        df_save = df.copy()
        df_train_save = df_train.copy()
        df_val_save = df_val.copy()
        df_test_save = df_test.copy()

        df_save['ecfp'] = df_save['ecfp'].apply(
            lambda x: json.dumps(x.tolist()))
        df_train_save['ecfp'] = df_train_save['ecfp'].apply(
            lambda x: json.dumps(x.tolist()))
        df_train_save['similar_molecules'] = df_train_save['similar_molecules'].apply(
            lambda x: json.dumps(x.tolist() if isinstance(x, np.ndarray) else x))
        df_val_save['ecfp'] = df_val_save['ecfp'].apply(
            lambda x: json.dumps(x.tolist()))
        df_test_save['ecfp'] = df_test_save['ecfp'].apply(
            lambda x: json.dumps(x.tolist()))

        df_save.to_csv('data/df.csv', index=False)
        df_train_save.to_csv('data/df_train.csv', index=False)
        df_val_save.to_csv('data/df_val.csv', index=False)
        df_test_save.to_csv('data/df_test.csv', index=False)

    else:

        def str_to_array(x):
            return np.array(json.loads(x), dtype=np.float32)

        df = pd.read_csv('data/df.csv')
        df_train = pd.read_csv('data/df_train.csv')
        df_val = pd.read_csv('data/df_val.csv')
        df_test = pd.read_csv('data/df_test.csv')

        df['ecfp'] = df['ecfp'].apply(str_to_array)
        df_train['ecfp'] = df_train['ecfp'].apply(str_to_array)
        df_train['similar_molecules'] = df_train['similar_molecules'].apply(
            str_to_array)
        df_val['ecfp'] = df_val['ecfp'].apply(str_to_array)
        df_test['ecfp'] = df_test['ecfp'].apply(str_to_array)

    return df, df_train, df_val, df_test
