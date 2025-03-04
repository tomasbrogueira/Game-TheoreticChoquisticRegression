"""Modules used in Fair Supervised PCA analysis"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler


def func_read_data(data_imp):
    # Base directory for datasets
    data_dir = "data"

    if data_imp == "banknotes":
        "Banknote authentication dataset - UCI"
        file_path = os.path.join(data_dir, "data_banknotes.csv")
        dataset = pd.read_csv(file_path)
        vals = dataset.values
        X = dataset.loc[:, dataset.columns != "authentic"]
        y = vals[:, 4]

    elif data_imp == "transfusion":
        "Blood Transfusion Service Center Data Set dataset - UCI"
        file_path = os.path.join(data_dir, "transfusion.csv")
        dataset = pd.read_csv(file_path)
        vals = dataset.values
        X = dataset.iloc[:, 0:-1]
        y = vals[:, -1]

    elif data_imp == "mammographic":
        "Mammographic mass dataset - UCI"
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
        "Raisin dataset - UCI"
        file_path = os.path.join(data_dir, "data_raisin.xlsx")
        dataset = pd.read_excel(file_path)
        vals = dataset.values
        X = dataset.iloc[:, 0:-1]
        y = np.array(vals[:, -1] == "Kecimen").astype(float)

    elif data_imp == "rice":
        "Rice (Commeo - 1 and Osmancik - 0) dataset - UCI"
        file_path = os.path.join(data_dir, "data_rice.xlsx")
        dataset = pd.read_excel(file_path)
        X = dataset.loc[:, dataset.columns != "Class"]
        vals = dataset.values
        y = vals[:, -1]

    elif data_imp == "diabetes":
        "Diabetes (PIMA) dataset"
        file_path = os.path.join(data_dir, "diabetes.csv")
        dataset = pd.read_csv(file_path)
        vals = dataset.values
        X = dataset.drop("Outcome", axis=1)
        y = dataset["Outcome"]

    elif data_imp == "wine":
        "Red wine quality dataset"
        file_path = os.path.join(data_dir, "data_wine_quality_red.csv")
        dataset = pd.read_csv(file_path)
        X = dataset.drop("quality", axis=1)
        y = dataset["quality"]
        y = (y > 5) * 1

    elif data_imp == "compas":
        # Compas dataset
        file_path = os.path.join(data_dir, "data_compas.xlsx")
        dataset = pd.read_excel(file_path)
        X = dataset.iloc[:, 0:8]
        X["race"] = np.where(X["race"] == "Hispanic", "Other", X["race"])
        X["race"] = np.where(X["race"] == "Asian", "Other", X["race"])
        X["race"] = np.where(X["race"] == "Native American", "Other", X["race"])
        y = dataset.loc[:, "score_risk"]

    elif data_imp == "lsac_new":
        # LSAC dataset (new)
        file_path = os.path.join(data_dir, "data_lsac_new.csv")
        dataset = pd.read_csv(file_path)
        dataset[["fulltime", "male", "race"]] = dataset[
            ["fulltime", "male", "race"]
        ].astype(str)
        X = dataset.iloc[:, 0:-1]
        y = dataset.loc[:, "pass_bar"]

    elif data_imp == "skin":
        "Skin segmentation dataset - UCI"
        file_path = os.path.join(data_dir, "data_skin.csv")
        dataset = pd.read_csv(file_path)
        X = dataset.loc[:, dataset.columns != "Class"]
        vals = dataset.values
        y = vals[:, -1] - 1

    elif data_imp == "seeds":
        "Seeds dataset - UCI"
        file_path = os.path.join(data_dir, "data_seeds.xlsx")
        dataset = pd.read_excel(file_path)
        X = dataset.loc[:, dataset.columns != "Class"]
        vals = dataset.values
        y = vals[:, -1]

    elif data_imp == "iris":
        "Iris dataset"
        from sklearn import datasets

        iris = datasets.load_iris()
        X = iris.data  # we only take the first two features.
        X = pd.DataFrame(X, columns=iris.feature_names)
        y = iris.target

    elif data_imp == "user_knowledge":
        "User knowledge modeling dataset - UCI"
        file_path = os.path.join(data_dir, "data_user_knowledge.xls")
        dataset = pd.read_excel(file_path)
        X = dataset.loc[:, :]
        uns1, uns2, uns3, uns4, uns5 = (
            X.UNS == "very_low",
            X.UNS == "Very Low",
            X.UNS == "Low",
            X.UNS == "Middle",
            X.UNS == "High",
        )
        X["UNS"] = np.select(
            [uns1, uns2, uns3, uns4, uns5], [1, 1, 2, 3, 4], default=None
        )
        vals = dataset.values
        y = vals[:, -1].astype(float)
        X = dataset.loc[:, dataset.columns != "UNS"]

    elif data_imp == "covid_gamma":
        # COVID Gamma dataset
        file_path = os.path.join(data_dir, "covid_gamma.csv")
        dados = pd.read_csv(file_path, index_col=0)
        dados = dados.dropna()
        for ii in range(dados.shape[1]):
            dados = dados[dados.iloc[:, ii] != 9]
        dados = dados.dropna()
        y = 2 * dados["CLASSI_FIN"].astype(int) - 1
        X = dados.drop(columns=["CLASSI_FIN", "VACINA_COV"])
        X = X - 1

    elif data_imp == "covid_delta":
        # COVID Delta dataset
        file_path = os.path.join(data_dir, "covid_delta.csv")
        dados = pd.read_csv(file_path)
        dados = dados.dropna()
        for ii in range(dados.shape[1]):
            dados = dados[dados.iloc[:, ii] != 9]
        dados = dados.dropna()
        y = 2 * dados["CLASSI_FIN"].astype(int) - 1
        X = dados.drop(columns=["CLASSI_FIN", "VACINA_COV"])
        X = X - 1

    elif data_imp == "covid_omicron":
        # COVID Omicron dataset
        file_path = os.path.join(data_dir, "covid_omicron.csv")
        dados = pd.read_csv(file_path, index_col=0)
        dados = dados.dropna()
        for ii in range(dados.shape[1]):
            dados = dados[dados.iloc[:, ii] != 9]
        dados = dados.dropna()
        y = 2 * dados["CLASSI_FIN"].astype(int) - 1
        X = dados.drop(columns=["CLASSI_FIN", "VACINA_COV"])
        X = X - 1

    elif data_imp == "dados_covid_sbpo_atual":
        # COVID SBPO dataset
        file_path = os.path.join(data_dir, "dados_covid_sbpo_atual.csv")
        dados = pd.read_csv(file_path)
        dados = dados.dropna()
        for ii in range(dados.shape[1]):
            dados = dados[dados.iloc[:, ii] != "Inconclusivo"]
        aux = []
        for ii in range(dados.shape[0]):
            if np.sum(dados.iloc[ii, 1:]) != 0:
                aux.append(ii)
        dados = dados.iloc[aux, :]
        y = pd.to_numeric(dados["test_result"].replace({"Negativo": -1, "Positivo": 1}))
        X = dados.drop(columns=["test_result"]) * 1

    else:
        raise ValueError(
            f"Invalid dataset name: {data_imp}. Please provide a valid dataset name."
        )

    # Over-sampling
    ros = RandomOverSampler(random_state=42)
    X, y = ros.fit_resample(X, y)

    # One-hot encoding for categorical features
    features_to_encode = X.columns[X.dtypes == object].tolist()
    if features_to_encode:
        ohe = OneHotEncoder(handle_unknown="ignore")
        encoder_X = pd.DataFrame(ohe.fit_transform(X[features_to_encode]).toarray())
        X = X.join(encoder_X)
        X.drop(features_to_encode, axis=1, inplace=True)
    else:
        print("No categorical features to encode.")

    return X, y


def func_kernel(Y, param_hsic):
    "This function calculates the centered kernel of Y"
    n = len(Y)
    Y = np.reshape(np.array(Y), (n, 1))
    H = np.eye(n) - (1 / n) * np.ones((n, n))

    if param_hsic == "linear":
        Hsic = Y @ Y.T  # + np.eye(n)

    elif param_hsic == "delta":
        A = np.array(Y @ Y.T)
        rows, cols = np.where(A == -1)
        A[rows, cols] = 0
        Hsic = A + np.eye(n)

    else:
        Hsic = Y @ Y.T + np.eye(n)

    return H @ Hsic @ H


def func_kernel_matrix(Y, param_hsic):
    "This function calculates the centered kernel of Y"
    n = Y.shape[0]
    H = np.eye(n) - (1 / n) * np.ones((n, n))

    if param_hsic == "linear":
        Hsic = Y @ Y.T  # + np.eye(n)

    elif param_hsic == "delta":
        A = np.array(Y @ Y.T)
        rows, cols = np.where(A == -1)
        A[rows, cols] = 0
        Hsic = A + np.eye(n)

    else:
        Hsic = Y @ Y.T + np.eye(n)

    return H @ Hsic @ H


def func_metrics(tp_g, fp_g, tn_g, fn_g, tp_gout, fp_gout, tn_gout, fn_gout):

    tpr_g, fpr_g = tp_g / (tp_g + fn_g), fp_g / (fp_g + tn_g)
    tpr_g[np.isnan(tpr_g)], fpr_g[np.isnan(fpr_g)] = 0, 0
    tpr_g_mean, tpr_g_std, fpr_g_mean, fpr_g_std = (
        np.mean(tpr_g, axis=0),
        np.std(tpr_g, axis=0),
        np.mean(fpr_g, axis=0),
        np.std(fpr_g, axis=0),
    )

    tpr_gout, fpr_gout = tp_gout / (tp_gout + fn_gout), fp_gout / (fp_gout + tn_gout)
    tpr_gout[np.isnan(tpr_gout)], fpr_gout[np.isnan(fpr_gout)] = 0, 0
    tpr_gout_mean, tpr_gout_std, fpr_gout_mean, fpr_gout_std = (
        np.mean(tpr_gout, axis=0),
        np.std(tpr_gout, axis=0),
        np.mean(fpr_gout, axis=0),
        np.std(fpr_gout, axis=0),
    )

    accur_g = (tp_g + tn_g) / (tp_g + tn_g + fp_g + fn_g)
    accur_g[np.isnan(accur_g)] = 0
    accur_g_mean, accur_g_std = np.mean(accur_g, axis=0), np.std(accur_g, axis=0)

    accur_gout = (tp_gout + tn_gout) / (tp_gout + tn_gout + fp_gout + fn_gout)
    accur_gout[np.isnan(accur_gout)] = 0
    accur_gout_mean, accur_gout_std = np.mean(accur_gout, axis=0), np.std(
        accur_gout, axis=0
    )

    eq_op = np.abs(tpr_g - tpr_gout)
    eq_op_mean, eq_op_std = np.mean(eq_op, axis=0), np.std(eq_op, axis=0)

    pr_pa = np.abs(fpr_g - fpr_gout)
    pr_pa_mean, pr_pa_std = np.mean(pr_pa, axis=0), np.std(pr_pa, axis=0)

    eq_od = np.abs(tpr_g - tpr_gout) + np.abs(fpr_g - fpr_gout)
    eq_od_mean, eq_od_std = np.mean(eq_od, axis=0), np.std(eq_od, axis=0)

    ov_ac = np.abs(accur_g - accur_gout)
    ov_ac_mean, ov_ac_std = np.mean(ov_ac, axis=0), np.std(ov_ac, axis=0)

    di_im = ((tp_gout + fp_gout) / (tp_gout + tn_gout + fp_gout + fn_gout)) / (
        (tp_g + fp_g) / (tp_g + tn_g + fp_g + fn_g)
    )
    di_im_mean, di_im_std = np.mean(di_im, axis=0), np.std(di_im, axis=0)

    return (
        tpr_g_mean,
        tpr_g_std,
        fpr_g_mean,
        fpr_g_std,
        tpr_gout_mean,
        tpr_gout_std,
        fpr_gout_mean,
        accur_g_mean,
        accur_g_std,
        accur_gout_mean,
        accur_gout_std,
        eq_op_mean,
        eq_op_std,
        pr_pa_mean,
        pr_pa_std,
        eq_od_mean,
        eq_od_std,
        ov_ac_mean,
        ov_ac_std,
        di_im_mean,
        di_im_std,
    )
