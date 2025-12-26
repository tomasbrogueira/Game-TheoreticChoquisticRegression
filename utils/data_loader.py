"""
Data loading and processing utilities.

This module provides functions for loading and preprocessing datasets
used in the project.
"""

import os
import numpy as np
import pandas as pd


def func_read_data(data_imp):
    """
    Load a dataset by name.
    
    Parameters
    ----------
    data_imp : str
        Name of the dataset to load. Options include:
        - "banknotes": Banknote authentication dataset
        - "transfusion": Blood Transfusion Service Center Data Set
        - "mammographic": Mammographic mass dataset
        - "raisin": Raisin dataset
        - "rice": Rice (Commeo and Osmancik) dataset
        - "diabetes": Diabetes (PIMA) dataset
        - "skin": Skin segmentation dataset
        
    Returns
    -------
    X : pandas.DataFrame
        Feature matrix
    y : numpy.ndarray
        Target values
    """
    # Base directory for datasets
    data_dir = "data"

    if data_imp == "banknotes":
        # Banknote authentication dataset - UCI
        file_path = os.path.join(data_dir, "data_banknotes.csv")
        dataset = pd.read_csv(file_path, header=None)
        vals = dataset.values
        X = dataset.iloc[:, :-1]
        y = vals[:, -1]

    elif data_imp == "transfusion":
        # Blood Transfusion Service Center Data Set dataset - UCI
        file_path = os.path.join(data_dir, "transfusion.csv")
        dataset = pd.read_csv(file_path)
        vals = dataset.values
        X = dataset.iloc[:, :-1]
        y = vals[:, -1]

    elif data_imp == "mammographic":
        # Mammographic mass dataset - UCI
        file_path = os.path.join(data_dir, "data_mammographic.data")
        dataset = pd.read_csv(file_path, header=None)
        vals = dataset.values
        X = dataset.iloc[:, 1:-1]
        y = vals[:, -1]
        for ii in range(X.shape[1]):
            y = y[X.iloc[:, ii] != "?"]
            X = X.loc[X.iloc[:, ii] != "?", :]
        X = X.astype(float)
        y = y.astype(float)

    elif data_imp == "raisin":
        # Raisin dataset - UCI
        file_path = os.path.join(data_dir, "data_raisin.xlsx")
        dataset = pd.read_excel(file_path)
        vals = dataset.values
        X = dataset.iloc[:, 0:-1]
        y = np.array(vals[:, -1] == "Kecimen").astype(float)

    elif data_imp == "rice":
        # Rice (Commeo - 1 and Osmancik - 0) dataset - UCI
        file_path = os.path.join(data_dir, "data_rice.xlsx")
        dataset = pd.read_excel(file_path)
        X = dataset.loc[:, dataset.columns != "Class"]
        vals = dataset.values
        y = vals[:, -1]

    elif data_imp == "diabetes":
        # Diabetes (PIMA) dataset
        file_path = os.path.join(data_dir, "diabetes.csv")
        dataset = pd.read_csv(file_path)
        vals = dataset.values
        X = dataset.drop("Outcome", axis=1)
        y = dataset["Outcome"]

    elif data_imp == "skin":
        # Skin segmentation dataset - UCI
        file_path = os.path.join(data_dir, "data_skin.csv")
        dataset = pd.read_csv(file_path)
        X = dataset.loc[:, dataset.columns != "Class"]
        vals = dataset.values
        y = vals[:, -1]
    
    elif data_imp == "pure_pairwise_interaction":
        # Pure pairwise interaction dataset
        file_path = os.path.join(data_dir, "pure_pairwise_interaction_dataset.csv")
        dataset = pd.read_csv(file_path, skiprows=2)
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1].values
    
    else:
        raise ValueError(f"Unknown dataset: {data_imp}")

    return X, y
