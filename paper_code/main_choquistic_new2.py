import os
os.environ["SCIPY_ARRAY_API"] = "1"  # Add this line before any other imports

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:56:41 2024

@author: guipe
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from itertools import chain, combinations
import itertools
from math import comb
from scipy.special import bernoulli
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from scipy.special import expit
from matplotlib.colors import BoundaryNorm, ListedColormap
import mod_GenFuzzyRegression

# from sklearn.datasets import make_classification
# X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2,n_clusters_per_class=1)
# plot_decision_boundary(X, y, clf)


class LogisticRegression_new(object):
    """
    Logistic Regression Classifier
    Parameters
    ----------
    learning_rate : int or float, default=0.1
        The tuning parameter for the optimization algorithm (here, Gradient Descent) 
        that determines the step size at each iteration while moving toward a minimum 
        of the cost function.
    max_iter : int, default=100
        Maximum number of iterations taken for the optimization algorithm to converge
    
    penalty : None or 'l2', default='l2'.
        Option to perform L2 regularization.
    C : float, default=0.1
        Inverse of regularization strength; must be a positive float. 
        Smaller values specify stronger regularization. 
    tolerance : float, optional, default=1e-4
        Value indicating the weight change between epochs in which
        gradient descent should terminated. 
    """

    def __init__(self, learning_rate=0.1, max_iter=10000, regularization='l2', tolerance = 1e-4):
        self.learning_rate  = learning_rate
        self.max_iter       = max_iter
        self.regularization = regularization
        self.tolerance      = tolerance
    
    def fit(self, X, y, C):
        """
        Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        Returns
        -------
        self : object
        """
        self.theta = np.zeros(X.shape[1] + 1)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        for _ in range(self.max_iter):
        
            errors = (self.__sigmoid(X @ self.theta)) - y
            N = X.shape[1]

            if self.regularization is not None:
                delta_grad = self.learning_rate * ((C * (X.T @ errors)) + np.sum(self.theta))
            else:
                delta_grad = self.learning_rate * (X.T @ errors)

            if np.all(abs(delta_grad) >= self.tolerance):
                self.theta -= delta_grad / N
            else:
                break
                
        return self

    def predict_proba(self, X):
        """
        Probability estimates for samples in X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Returns
        -------
        probs : array-like of shape (n_samples,)
            Returns the probability of each sample.
        """
        return self.__sigmoid((X @ self.theta[1:]) + self.theta[0])
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        labels : array, shape [n_samples]
            Predicted class label per sample.
        """
        return np.round(self.predict_proba(X))
        
    def __sigmoid(self, z):
        """
        The sigmoid function.
        Parameters
        ------------
        z : float
            linear combinations of weights and sample features
            z = w_0 + w_1*x_1 + ... + w_n*x_n
        Returns
        ---------
        Value of logistic function at z
        """
        return 1 / (1 + expit(-z))

    def get_params(self):
        """
        Get method for models coeffients and intercept.
        Returns
        -------
        params : dict
        """
        try:
            params = dict()
            params['intercept'] = self.theta[0]
            params['coef'] = self.theta[1:]
            return params
        except:
            raise Exception('Fit the model first!')
            
def plot_decision_boundary(X, y, model):
    cMap = ListedColormap(["#6b76e8", "#c775d1"])
    cMapa = ListedColormap(["#c775d1", "#6b76e8"])

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .02  # step size in the mesh

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.column_stack((xx.ravel(), yy.ravel())))
    Z = Z.reshape(xx.shape)

    plt.figure(1, figsize=(8, 6), frameon=True)
    plt.axis('off')
    plt.pcolormesh(xx, yy, Z, cmap=cMap)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker = "o", edgecolors='k', cmap=cMapa)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

def nParam_kAdd(kAdd, nAttr):
    '''Return the number of parameters in a k-additive model'''
    aux_numb = 1
    for ii in range(kAdd):
        aux_numb += comb(nAttr, ii+1)
    return aux_numb

def powerset(iterable, k_add):
    '''Return the powerset (for coalitions until k_add players) of a set of m attributes'''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(k_add+1))

def choquet_matrix_mobius(X_orig, kadd):
    nSamp, nAttr = X_orig.shape  # Number of samples and attributes
    k_add_numb = nParam_kAdd(kadd, nAttr)
    data_opt = np.zeros((nSamp, k_add_numb-1))
    for i, s in enumerate(powerset(range(nAttr), kadd)):
        s = list(s)
        if len(s) > 0:
            data_opt[:, i-1] = np.min(X_orig.iloc[:, s], axis=1)
    return data_opt

def choquet_matrix(X_orig):
    X_orig_sort = np.sort(X_orig)
    X_orig_sort_ind = np.array(np.argsort(X_orig))
    nSamp, nAttr = X_orig.shape
    X_orig_sort_ext = np.concatenate((np.zeros((nSamp, 1)), X_orig_sort), axis=1)
    
    sequence = np.arange(nAttr)
    combin = (99)*np.ones((2**nAttr-1, nAttr))
    count = 0
    for ii in range(nAttr):
        combin[count:count+comb(nAttr, ii+1), 0:ii+1] = np.array(list(itertools.combinations(sequence, ii+1)))
        count += comb(nAttr, ii+1)
    
    data_opt = np.zeros((nSamp, 2**nAttr-1))
    for ii in range(nAttr):
        for jj in range(nSamp):
            list1 = combin.tolist()
            aux = list1.index(np.concatenate((np.sort(X_orig_sort_ind[jj, ii:]), 99*np.ones((ii,))), axis=0).tolist())
            data_opt[jj, aux] = X_orig_sort_ext[jj, ii+1] - X_orig_sort_ext[jj, ii]
    return data_opt

def choquet_matrix_2add(X_orig):
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    k_add = 2
    k_add_numb = nParam_kAdd(k_add, nAttr)
    coalit = np.zeros((k_add_numb, nAttr))
    for i, s in enumerate(powerset(range(nAttr), k_add)):
        s = list(s)
        coalit[i, s] = 1
    data_opt = np.zeros((nSamp, k_add_numb))
    for i in range(nAttr):
        data_opt[:, i+1] = data_opt[:, i+1] + X_orig[:, i]
        for i2 in range(i+1, nAttr):
            data_opt[:, (coalit[:, [i, i2]]==1).all(axis=1)] = (np.min([X_orig[:, i], X_orig[:, i2]], axis=0)).reshape(nSamp, 1)
        for ii in range(nAttr+1, len(coalit)):
            if coalit[ii, i] == 1:
                data_opt[:, ii] = data_opt[:, ii] + (-1/2)*X_orig[:, i]
    return data_opt[:, 1:]

def mlm_matrix(X_orig):
    nSamp, nAttr = X_orig.shape
    X_orig = np.array(X_orig)
    data_opt = np.zeros((nSamp, 2**nAttr))
    for i, s in enumerate(powerset(range(nAttr), nAttr)):
        s = list(s)
        data_opt[:, i] = math.prod(X_orig[:, [s]].T) * math.prod(1-X_orig[:, [diff(np.arange(nAttr), s)]].T)
    return data_opt[:, 1:]

def mlm_matrix_2add(X_orig):
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    k_add = 2
    k_add_numb = nParam_kAdd(k_add, nAttr)
    coalit = np.zeros((k_add_numb, nAttr))
    for i, s in enumerate(powerset(range(nAttr), k_add)):
        s = list(s)
        coalit[i, s] = 1
    data_opt = np.zeros((nSamp, k_add_numb))
    for i in range(nAttr):
        data_opt[:, i+1] = data_opt[:, i+1] + X_orig[:, i]
        for i2 in range(i+1, nAttr):
            data_opt[:, (coalit[:, [i, i2]]==1).all(axis=1)] = (X_orig[:, i]* X_orig[:, i2]).reshape(nSamp, 1)
        for ii in range(nAttr+1, len(coalit)):
            if coalit[ii, i] == 1:
                data_opt[:, ii] = data_opt[:, ii] + (-1/2)*X_orig[:, i]
    return data_opt[:, 1:]

def tr_shap2game(nAttr, k_add):
    '''Return the transformation matrix from Shapley interaction indices to game, given a k-additive model'''
    nBern = bernoulli(k_add)  # Bernoulli numbers
    k_add_numb = nParam_kAdd(k_add, nAttr)
    coalit = np.zeros((k_add_numb, nAttr))
    for i, s in enumerate(powerset(range(nAttr), k_add)):
        s = list(s)
        coalit[i, s] = 1
    matrix_shap2game = np.zeros((k_add_numb, k_add_numb))
    for i in range(coalit.shape[0]):
        for i2 in range(k_add_numb):
            aux2 = int(sum(coalit[i2, :]))
            aux3 = int(sum(coalit[i, :] * coalit[i2, :]))
            aux4 = 0
            for i3 in range(int(aux3+1)):
                aux4 += comb(aux3, i3) * nBern[aux2-i3]
            matrix_shap2game[i, i2] = aux4
    return matrix_shap2game

def tr_banz2game(nAttr, k_add):
    '''Return the transformation matrix from Banzhaf interaction indices, given a k-additive model, to game'''
    nBern = bernoulli(k_add)
    k_add_numb = nParam_kAdd(k_add, nAttr)
    coalit = np.zeros((k_add_numb, nAttr))
    for i, s in enumerate(powerset(range(nAttr), k_add)):
        s = list(s)
        coalit[i, s] = 1
    matrix_banz2game = np.zeros((k_add_numb, k_add_numb))
    for i in range(coalit.shape[0]):
        A = coalit[i, :]
        cardA = int(sum(A))
        for i2 in range(k_add_numb):
            B = coalit[i2, :]
            cardB = int(sum(B))
            cardBminusA = sum((B - A) > 0)
            matrix_banz2game[i, i2] = ((1/2)**cardB) * ((-1)**cardBminusA)
    return matrix_banz2game

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

''' Importing dataset '''
# data_imp is the dataset to import (banknotes, transfusion, mammographic, raisin, rice, diabetes, skin, dados_covid_sbpo_atual)
#data_imp = list(['banknotes','transfusion','mammographic','raisin','rice','diabetes','skin','dados_covid_sbpo_atual'])
# data_imp = list(['covid_gamma','covid_delta','covid_omicron','dados_covid_sbpo'])
data_imp = list(['dados_covid_sbpo_atual'])

attr = ('LR', 'CR', 'CR2add', 'MLMR', 'MLMR2add')
# solver_lr = ('lbfgs', 'newton-cg', 'sag','saga')
solver_lr = ('newton-cg',)

nSimul = 1  # 50 simulations

accuracy_linear_train = np.zeros((len(data_imp), len(solver_lr), nSimul))
accuracy_linear_test = np.zeros((len(data_imp), len(solver_lr), nSimul))

accuracy_choquet_kadd_train = np.zeros((len(data_imp), len(solver_lr), nSimul))
accuracy_choquet_kadd_test = np.zeros((len(data_imp), len(solver_lr), nSimul))

accuracy_choquet_train = np.zeros((len(data_imp), len(solver_lr), nSimul))
accuracy_choquet_test = np.zeros((len(data_imp), len(solver_lr), nSimul))

accuracy_mlm_kadd_train = np.zeros((len(data_imp), len(solver_lr), nSimul))
accuracy_mlm_kadd_test = np.zeros((len(data_imp), len(solver_lr), nSimul))

accuracy_mlm_train = np.zeros((len(data_imp), len(solver_lr), nSimul))
accuracy_mlm_test = np.zeros((len(data_imp), len(solver_lr), nSimul))

param_linear_train = []
param_choquet_train = []
param_choquet_kadd_train = []
param_mlm_train = []
param_mlm_kadd_train = []
n_iterations = {'LR': [], 'CR': [], 'CR2add': [], 'MLMR': [], 'MLMR2add': []}

for ll in range(len(data_imp)):
    X, y = mod_GenFuzzyRegression.func_read_data(data_imp[ll])
    
    # Data parameters
    nSamp, nAttr = X.shape
    
    # Compute transformation matrices for the full Choquet model.
    # These matrices allow you to map between the game parameters (regression coefficients)
    # and the Shapley or Banzhaf interaction indices.
    matrix_s2g = tr_shap2game(nAttr, nAttr)
    matrix_s2g = matrix_s2g[1:, :]  # Remove the empty coalition row
    matrix_b2g = tr_banz2game(nAttr, nAttr)
    matrix_b2g = matrix_b2g[1:, :]
    
    n_2add = nParam_kAdd(2, nAttr)
    n_2add = nAttr + 1
    
    # Normalization 0-1
    X = (X - X.min()) / (X.max() - X.min())
    
    # Choquet integral matrix
    X_choquet = choquet_matrix(X)
    X_choquet_2add = choquet_matrix_2add(X)

    # Multilinear model matrix
    X_mlm = mlm_matrix(X)
    X_mlm_2add = mlm_matrix_2add(X)
    
    for kk in range(nSimul):
        
        indices = np.arange(np.size(X, 0))
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.2, stratify=y)
        
        X_choquet_train = X_choquet[indices_train, :]
        X_choquet_kadd_train = X_choquet_2add[indices_train, :]
        X_choquet_test = X_choquet[indices_test, :]
        X_choquet_kadd_test = X_choquet_2add[indices_test, :]
        
        X_mlm_train = X_mlm[indices_train, :]
        X_mlm_kadd_train = X_mlm_2add[indices_train, :]
        X_mlm_test = X_mlm[indices_test, :]
        X_mlm_kadd_test = X_mlm_2add[indices_test, :]
        
        for ii in range(len(solver_lr)):
        
            log_reg = LogisticRegression(random_state=0, penalty=None, solver=solver_lr[ii], max_iter=10000)
            log_reg.fit(X_train, y_train)
            accuracy_linear_train[ll, ii, kk] = log_reg.score(X_train, y_train)
            accuracy_linear_test[ll, ii, kk] = log_reg.score(X_test, y_test)
            param_linear_train.append(log_reg.coef_)
            n_iterations['LR'].append(log_reg.n_iter_)
            
            log_reg = LogisticRegression(random_state=0, penalty=None, solver=solver_lr[ii], max_iter=10000)
            log_reg.fit(X_choquet_kadd_train, y_train)
            accuracy_choquet_kadd_train[ll, ii, kk] = log_reg.score(X_choquet_kadd_train, y_train)
            accuracy_choquet_kadd_test[ll, ii, kk] = log_reg.score(X_choquet_kadd_test, y_test)
            param_choquet_kadd_train.append(log_reg.coef_)
            n_iterations['CR2add'].append(log_reg.n_iter_)
            
            log_reg = LogisticRegression(random_state=0, penalty=None, solver=solver_lr[ii], max_iter=10000)
            log_reg.fit(X_choquet_train, y_train)
            accuracy_choquet_train[ll, ii, kk] = log_reg.score(X_choquet_train, y_train)
            accuracy_choquet_test[ll, ii, kk] = log_reg.score(X_choquet_test, y_test)
            param_choquet_train.append(log_reg.coef_)
            n_iterations['CR'].append(log_reg.n_iter_)
            
            log_reg = LogisticRegression(random_state=0, penalty=None, solver=solver_lr[ii], max_iter=10000)
            log_reg.fit(X_mlm_kadd_train, y_train)
            accuracy_mlm_kadd_train[ll, ii, kk] = log_reg.score(X_mlm_kadd_train, y_train)
            accuracy_mlm_kadd_test[ll, ii, kk] = log_reg.score(X_mlm_kadd_test, y_test)
            param_mlm_kadd_train.append(log_reg.coef_)
            n_iterations['MLMR2add'].append(log_reg.n_iter_)
            
            log_reg = LogisticRegression(random_state=0, penalty=None, solver=solver_lr[ii], max_iter=10000)
            log_reg.fit(X_mlm_train, y_train)
            accuracy_mlm_train[ll, ii, kk] = log_reg.score(X_mlm_train, y_train)
            accuracy_mlm_test[ll, ii, kk] = log_reg.score(X_mlm_test, y_test)
            param_mlm_train.append(log_reg.coef_)
            n_iterations['MLMR'].append(log_reg.n_iter_)
                      
            print(ll+1, '/', len(data_imp), '-', kk, '/', nSimul, '-', ii+1, '/', len(solver_lr))
            
# exit();

data_save = [accuracy_linear_train, accuracy_linear_test, accuracy_choquet_kadd_train, accuracy_choquet_kadd_test, 
             accuracy_choquet_train, accuracy_choquet_test, accuracy_mlm_kadd_train, accuracy_mlm_kadd_test, 
             accuracy_mlm_train, accuracy_mlm_test, data_imp, param_linear_train, param_choquet_train, 
             param_choquet_kadd_train, param_mlm_train, param_mlm_kadd_train, solver_lr]
#np.save('results_logistic_all_test.npy', np.array(data_save, dtype=object), allow_pickle=True)
# accuracy_linear_train, accuracy_linear_test, accuracy_choquet_kadd_train, accuracy_choquet_kadd_test, accuracy_choquet_train, accuracy_choquet_test, accuracy_mlm_kadd_train, accuracy_mlm_kadd_test, accuracy_mlm_train, accuracy_mlm_test, data_imp, param_linear_train, param_choquet_train, param_choquet_kadd_train, param_mlm_train, param_mlm_kadd_train, solver_lr = np.load('results_logistic_all.npy', allow_pickle=True)
    
accuracy_linear_train_mean = np.mean(accuracy_linear_train, axis=2)
accuracy_linear_test_mean = np.mean(accuracy_linear_test, axis=2)
    
accuracy_choquet_kadd_train_mean = np.mean(accuracy_choquet_kadd_train, axis=2)
accuracy_choquet_kadd_test_mean = np.mean(accuracy_choquet_kadd_test, axis=2)

accuracy_choquet_train_mean = np.mean(accuracy_choquet_train, axis=2)
accuracy_choquet_test_mean = np.mean(accuracy_choquet_test, axis=2)
    
accuracy_mlm_kadd_train_mean = np.mean(accuracy_mlm_kadd_train, axis=2)
accuracy_mlm_kadd_test_mean = np.mean(accuracy_mlm_kadd_test, axis=2)

accuracy_mlm_train_mean = np.mean(accuracy_mlm_train, axis=2)
accuracy_mlm_test_mean = np.mean(accuracy_mlm_test, axis=2)  

accuracy_linear_train_std = np.std(accuracy_linear_train, axis=2)
accuracy_linear_test_std = np.std(accuracy_linear_test, axis=2)
    
accuracy_choquet_kadd_train_std = np.std(accuracy_choquet_kadd_train, axis=2)
accuracy_choquet_kadd_test_std = np.std(accuracy_choquet_kadd_test, axis=2)

accuracy_choquet_train_std = np.std(accuracy_choquet_train, axis=2)
accuracy_choquet_test_std = np.std(accuracy_choquet_test, axis=2)
    
accuracy_mlm_kadd_train_std = np.std(accuracy_mlm_kadd_train, axis=2)
accuracy_mlm_kadd_test_std = np.std(accuracy_mlm_kadd_test, axis=2)

accuracy_mlm_train_std = np.std(accuracy_mlm_train, axis=2)
accuracy_mlm_test_std = np.std(accuracy_mlm_test, axis=2)

iterations_mean = {key: np.mean(val) for key, val in n_iterations.items()}
iterations_std = {key: np.std(val) for key, val in n_iterations.items()}
print(iterations_mean)
print(iterations_std)

print([accuracy_linear_train_mean, accuracy_choquet_train_mean, accuracy_choquet_kadd_train_mean, accuracy_mlm_train_mean, accuracy_mlm_kadd_train_mean])
print([accuracy_linear_train_std, accuracy_choquet_train_std, accuracy_choquet_kadd_train_std, accuracy_mlm_train_std, accuracy_mlm_kadd_train_std])

print([accuracy_linear_test_mean, accuracy_choquet_test_mean, accuracy_choquet_kadd_test_mean, accuracy_mlm_test_mean, accuracy_mlm_kadd_test_mean])
print([accuracy_linear_test_std, accuracy_choquet_test_std, accuracy_choquet_kadd_test_std, accuracy_mlm_test_std, accuracy_mlm_kadd_test_std])



if len(param_choquet_train) > 0:
    # Average the full-Choquet coefficients over all simulation runs
    game_params_avg = np.mean(np.vstack(param_choquet_train), axis=0)
    # Recover Shapley indices using the pseudo-inverse of the transformation matrix
    recovered_shapley = np.linalg.pinv(matrix_s2g) @ game_params_avg
    # Recover Banzhaf indices similarly (if needed)
    recovered_banzhaf = np.linalg.pinv(matrix_b2g) @ game_params_avg

    # We assume that the first nAttr entries in recovered_shapley are the singleton contributions.
    singleton_shapley = recovered_shapley[:nAttr]
    
    # Define feature names (adjust as needed)
    feature_names = ['Fever', 'Cough', 'Sore throat', 'Runny nose', 'Myalgia', 
                     'Nausea', 'Diarrhea', 'Loss of smell', 'Shortness of breath']
    
    # Order features by descending Shapley value (largest first)
    ordered_indices = np.argsort(singleton_shapley)[::-1]
    names_ordered = np.array(feature_names)[ordered_indices]
    values_ordered = singleton_shapley[ordered_indices]
    
    # Create the horizontal bar chart with descending values
    plt.figure(figsize=(10, 8))
    plt.barh(names_ordered, values_ordered, color='green', edgecolor='black')
    plt.xlabel('Shapley Value', fontsize=24)
    plt.title('Shapely Values for Features', fontsize=28)
    plt.tight_layout()

    # Set up x-axis ticks and vertical grid lines at intervals of 0.25
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
    
    # Save the plot
    os.makedirs("plots", exist_ok=True)
    shapely_plot_path = os.path.join("plots", "shapely_values.png")
    plt.savefig(shapely_plot_path)
    plt.close()
    print("Saved Shapley values plot to:", shapely_plot_path)


#covid_param = param_choquet_kadd_train[700:]
covid_param = param_choquet_kadd_train

lista = np.arange(0, len(covid_param), 2)  # Only use valid indices
covid_param_aux = []
for ii in lista:
    covid_param_aux.append(covid_param[ii])

param_values = np.zeros((len(covid_param_aux), len(covid_param_aux[ii][0])))
for ii in range(len(covid_param_aux)):
    param_values[ii, :] = covid_param_aux[ii][0]
covid_param_mean = np.mean(param_values, axis=0)[0:9]
covid_param_std = np.std(param_values, axis=0)[0:9]

import matplotlib.pyplot as plt
import numpy as np
import itertools
import os


# Ensure the "plots" folder exists
os.makedirs("plots", exist_ok=True)

# Define the symptom names
names = ['Fever', 'Cough', 'Sore throat', 'Runny nose', 'Myalgia', 
         'Nausea', 'Diarrhea', 'Loss of smell', 'Shortness of breath']

# --- Dummy data for testing (remove if already defined) ---
# Uncomment these lines if your variables are not defined
# covid_param_mean = np.random.rand(len(names))
# covid_param_std = np.random.rand(len(names)) * 0.1
# nAttr = len(names)
# param_values = np.random.rand(20, nAttr)  # Adjust shape as needed
# ---------------------------------------------------------

# Ordering the data
ordered_indices = np.argsort(covid_param_mean)[::-1]
names_ordered = np.array(names)[ordered_indices]
ordered_values = np.array(covid_param_mean)[ordered_indices]
ordered_std = np.array(covid_param_std)[ordered_indices]

# --- First Plot: Marginal Contribution ---
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(names_ordered, ordered_values, xerr=ordered_std, color='blue', edgecolor='black')

# Adjusting the plot appearance
plt.yticks(fontsize=20)
plt.xticks(fontsize=15)
ax.set_xlabel('Marginal contribution', fontsize=24)
ax.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot instead of showing it
marginal_plot_path = os.path.join("plots", "marginal_contribution.png")
plt.savefig(marginal_plot_path)
plt.close()
print("Saved marginal contribution plot to:", marginal_plot_path)

# --- Second Plot: Interaction Effects ---
sequence = np.arange(nAttr)
combin = np.array(list(itertools.combinations(sequence, 2)))
covid_param_inter_mean = np.mean(param_values, axis=0)[9:]

plot_aux = np.zeros((nAttr, nAttr))
for ll in range(combin.shape[0]):
    plot_aux[combin[ll, 0], combin[ll, 1]] = covid_param_inter_mean[ll]
plot_aux = plot_aux + plot_aux.T

plt.figure(figsize=(8, 6))
plt.imshow(plot_aux)
plt.colorbar(orientation="vertical")
pos = np.arange(len(feature_names))
plt.yticks(pos, feature_names)
plt.xticks(pos, feature_names, rotation=90)
plt.title("Interaction effects among symptoms")
interaction_plot_path = os.path.join("plots", "interaction_effects.png")
plt.savefig(interaction_plot_path)
plt.close()
print("Saved interaction effects plot to:", interaction_plot_path)

# Interpreting odds change
# aa = pd.DataFrame([1,1,0,0,1,0,0,0,0]).T
# bb = choquet_matrix_2add(aa)
# log_reg.predict_proba(bb)
