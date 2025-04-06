"""
Regression models that extend logistic regression with aggregation functions.

This module implements regression models that extend logistic regression
by using aggregation functions like the Choquet integral and Multi-Linear Model
to capture non-linear interactions between features.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted, check_array
from .choquet import ChoquetTransformer
from .mlm import MLMTransformer


class ChoquisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic regression with Choquet integral or MLM feature transformation.
    
    This classifier extends logistic regression by first transforming the input
    features using either the Choquet integral or Multi-Linear Model (MLM),
    then applying logistic regression to the transformed features.
    
    Parameters
    ----------
    method : str, default="choquet_2add"
        The transformation method. Options:
        - "choquet": General Choquet integral
        - "choquet_2add": 2-additive Choquet integral with Shapley representation
        - "mlm": Full multilinear model
        - "mlm_2add": 2-additive multilinear model
    representation : str, default="game"
        For method="choquet", defines the representation to use:
        - "game": Uses game-based representation
        - "mobius": Uses Möbius representation
        Ignored for other methods.
    k_add : int or None, default=None
        Additivity level for k-additive models. If not specified and method is 
        "choquet" or "mlm", defaults to using all features. Ignored for methods 
        ending with "_2add" (where k_add=2 is implicit).
    scale_data : bool, default=True
        Whether to scale input data to [0,1] range before transformation.
    C : float, default=1.0
        Inverse of regularization strength for logistic regression.
    penalty : {'l1', 'l2', 'elasticnet', None}, default='l2'
        Penalty for logistic regression.
    solver : str, default='lbfgs'
        Solver for logistic regression.
    max_iter : int, default=1000
        Maximum number of iterations for logistic regression.
    random_state : int or None, default=None
        Random seed for logistic regression.
    """

    def __init__(self, method="choquet_2add", representation="game", k_add=None,
                 scale_data=True, C=1.0, penalty='l2', solver='lbfgs',
                 max_iter=1000, random_state=None):
        self.method = method
        self.representation = representation
        self.k_add = k_add
        self.scale_data = scale_data
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Validate method
        valid_methods = ["choquet", "choquet_2add", "mlm", "mlm_2add"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
            
    def fit(self, X, y):
        """
        Fit the model to the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Store original feature names if available
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"Feature_{i+1}" for i in range(X.shape[1])]
        
        # Scale data if requested
        if self.scale_data:
            self.scaler_ = MinMaxScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X
            
        # Create and fit the transformer
        if self.method.startswith("choquet"):
            self.transformer_ = ChoquetTransformer(
                method=self.method,
                representation=self.representation,
                k_add=self.k_add
            )
        elif self.method.startswith("mlm"):
            self.transformer_ = MLMTransformer(
                method=self.method,
                k_add=self.k_add
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        self.transformer_.fit(X_scaled)
        
        # Transform the data
        X_transformed = self.transformer_.transform(X_scaled)
        
        # Create and fit the logistic regression model
        self.model_ = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.model_.fit(X_transformed, y)
        
        return self
        
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        check_is_fitted(self, ["model_", "transformer_"])
        X = np.asarray(X)
        
        # Scale data if requested
        if self.scale_data:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        # Transform the data
        X_transformed = self.transformer_.transform(X_scaled)
        
        # Predict using the logistic regression model
        return self.model_.predict(X_transformed)
        
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        check_is_fitted(self, ["model_", "transformer_"])
        X = np.asarray(X)
        
        # Scale data if requested
        if self.scale_data:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        # Transform the data
        X_transformed = self.transformer_.transform(X_scaled)
        
        # Predict probabilities using the logistic regression model
        return self.model_.predict_proba(X_transformed)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            "method": self.method,
            "representation": self.representation,
            "k_add": self.k_add,
            "scale_data": self.scale_data,
            "C": self.C,
            "penalty": self.penalty,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "random_state": self.random_state
        }
    
    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
