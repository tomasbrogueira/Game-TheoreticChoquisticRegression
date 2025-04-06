from regression_classes import ChoquisticRegression_Composition, ChoquisticRegression_Inheritance, choquet_k_additive_game, choquet_k_additive_mobius

import os
os.environ["SCIPY_ARRAY_API"] = "1"

import itertools
import numpy as np
from math import factorial, comb
from itertools import chain, combinations

import matplotlib.pyplot as plt
import mod_GenFuzzyRegression as mGFR

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Define functions from original.py here instead of importing
def powerset(iterable,k_add):
    '''Return the powerset (for coalitions until k_add players) of a set of m attributes
    powerset([1,2,..., m],m) --> () (1,) (2,) (3,) ... (m,) (1,2) (1,3) ... (1,m) ... (m-1,m) ... (1, ..., m)
    powerset([1,2,..., m],2) --> () (1,) (2,) (3,) ... (m,) (1,2) (1,3) ... (1,m) ... (m-1,m)
    '''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(k_add+1))

def nParam_kAdd(kAdd,nAttr):
    '''Return the number of parameters in a k-additive model'''
    aux_numb = 1
    for ii in range(kAdd):
        aux_numb += comb(nAttr,ii+1)
    return aux_numb



#data_imp = ['dados_covid_sbpo_atual', 'banknotes', 'transfusion', 'mammographic', 'raisin', 'rice', 'diabetes', 'skin']
data_imp = ['banknotes', 'transfusion']

for ll in range(len(data_imp)):
    X, y = mGFR.func_read_data(data_imp[ll])

    # Data parameters
    nSamp,nAttr = X.shape
    print("Verifying Choquet model equivalence across different domains")
    print("=" * 70)

    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale data to [0,1] range (important for both approaches)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    for k in range(1, nAttr+1):
        
        X_choquet_game = choquet_k_additive_game(X_train_scaled, k_add=k)
        model_game = LogisticRegression(max_iter=1000, random_state=0, solver='newton-cg')
        model_game.fit(X_choquet_game, y_train)
        model_game_coefficients = model_game.coef_[0]
        model_game_intercept = model_game.intercept_[0]

        inheritance_model_game = ChoquisticRegression_Inheritance(
            method="choquet",
            representation="game",
            k_add=k,
            scale_data=True,
            logistic_params={"max_iter": 1000, "random_state": 0, "solver": "newton-cg"}
        )

        inheritance_model_game.fit(X_train, y_train)
        inheritance_model_game_coefficients = inheritance_model_game.coef_[0]
        inheritance_model_game_intercept = inheritance_model_game.intercept_[0]

        composition_model_game = ChoquisticRegression_Composition(
            method="choquet",
            representation="game",
            k_add=k,
            scale_data=True,
            logistic_params={"max_iter": 1000, "random_state": 0, "solver": "newton-cg"}
        )
        composition_model_game.fit(X_train, y_train)
        composition_model_game_coefficients = composition_model_game.coef_[0]
        composition_model_game_intercept = composition_model_game.intercept_[0]

        # comparison of model coefficients using np allclose
        is_equal_inheritance = np.allclose(model_game_coefficients, inheritance_model_game_coefficients) and np.allclose(model_game_intercept, inheritance_model_game_intercept) 
        is_equal_composition = np.allclose(model_game_coefficients, composition_model_game_coefficients) and np.allclose(model_game_intercept, composition_model_game_intercept)
        is_equal_classes = np.allclose(inheritance_model_game_coefficients, composition_model_game_coefficients) and np.allclose(inheritance_model_game_intercept, composition_model_game_intercept)
        print(f"{data_imp[ll]} is_equal_inheritance (game): {is_equal_inheritance}")
        print(f"{data_imp[ll]} is_equal_composition (game): {is_equal_composition}")
        print(f"{data_imp[ll]} is_equal_classes (game): {is_equal_classes}")
        print("=" * 70)

        X_choquet_mobius = choquet_k_additive_mobius(X_train_scaled, k_add=k)
        model_mobius = LogisticRegression(max_iter=1000, random_state=0, solver='newton-cg')
        model_mobius.fit(X_choquet_mobius, y_train)
        model_mobius_coefficients = model_mobius.coef_[0]
        model_mobius_intercept = model_mobius.intercept_[0]

        inheritance_model_mobius = ChoquisticRegression_Inheritance(
            method="choquet",
            representation="mobius",
            k_add=k,
            scale_data=True,
            logistic_params={"max_iter": 1000, "random_state": 0}
        )
        inheritance_model_mobius.fit(X_train, y_train)
        inheritance_model_mobius_coefficients = inheritance_model_mobius.coef_[0]
        inheritance_model_mobius_intercept = inheritance_model_mobius.intercept_[0]

        composition_model_mobius = ChoquisticRegression_Composition(
            method="choquet",
            representation="mobius",
            k_add=k,
            scale_data=True,
            logistic_params={"max_iter": 1000, "random_state": 0}
        )
        composition_model_mobius.fit(X_train, y_train)
        composition_model_mobius_coefficients = composition_model_mobius.coef_[0]
        composition_model_mobius_intercept = composition_model_mobius.intercept_[0]

        # comparison of model coefficients using np allclose for mobius representation
        is_equal_inheritance_mobius = np.allclose(model_mobius_coefficients, inheritance_model_mobius_coefficients) and np.allclose(model_mobius_intercept, inheritance_model_mobius_intercept) 
        is_equal_composition_mobius = np.allclose(model_mobius_coefficients, composition_model_mobius_coefficients) and np.allclose(model_mobius_intercept, composition_model_mobius_intercept)
        is_equal_classes_mobius = np.allclose(inheritance_model_mobius_coefficients, composition_model_mobius_coefficients) and np.allclose(inheritance_model_mobius_intercept, composition_model_mobius_intercept)
        print(f"{data_imp[ll]} is_equal_inheritance (mobius): {is_equal_inheritance_mobius}")
        print(f"{data_imp[ll]} is_equal_composition (mobius): {is_equal_composition_mobius}")
        print(f"{data_imp[ll]} is_equal_classes (mobius): {is_equal_classes_mobius}")
        print("=" * 70)
    

