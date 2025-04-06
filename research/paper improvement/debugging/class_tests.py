import os
os.environ["SCIPY_ARRAY_API"] = "1"

import numpy as np
import math
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

# filepath: c:\Users\Tomas\OneDrive - Universidade de Lisboa\3ºano_LEFT\PIC-I\paper improvement\test_class_tests.py


from regression_classes import (
    ChoquetTransformer,
    ChoquisticRegression_Composition,
    ChoquisticRegression_Inheritance,
    nParam_kAdd,
    powerset
)

# Constants for test data
N_SAMPLES = 100
N_FEATURES = 5
RANDOM_STATE = 0


@pytest.fixture
def classification_data():
    """Generate a small classification dataset for testing."""
    X, y = make_classification(
        n_samples=N_SAMPLES, 
        n_features=N_FEATURES,
        n_informative=3, 
        n_redundant=1,
        random_state=RANDOM_STATE
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Scale the data to [0,1] which is required for Choquet transformations
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


# Utility function tests
def test_nParam_kAdd():
    """Test the nParam_kAdd function."""
    # Test for a simple case: k=1, n=3
    result = nParam_kAdd(1, 3)
    # We expect 1 + 3 = 4 parameters (empty set + 3 singletons)
    assert result == 4
    
    # Test for k=2, n=3
    result = nParam_kAdd(2, 3)
    # We expect 1 + 3 + 3 = 7 parameters (empty set + 3 singletons + 3 pairs)
    assert result == 7
    
    # Test for full model k=n=3
    result = nParam_kAdd(3, 3)
    # We expect 1 + 3 + 3 + 1 = 8 parameters (empty + singletons + pairs + triplet)
    assert result == 8


def test_powerset():
    """Test the powerset function."""
    # Test with a small set and k_add=2
    result = list(powerset(range(3), 2))
    # We expect: (), (0,), (1,), (2,), (0,1), (0,2), (1,2)
    expected = [(), (0,), (1,), (2,), (0,1), (0,2), (1,2)]
    assert result == expected
    
    # Test with k_add=1
    result = list(powerset(range(3), 1))
    # We expect: (), (0,), (1,), (2,)
    expected = [(), (0,), (1,), (2,)]
    assert result == expected


# ChoquetTransformer tests
def test_choquet_transformer_init():
    """Test initialization of ChoquetTransformer with different parameters."""
    # Test default initialization
    transformer = ChoquetTransformer()
    assert transformer.method == "choquet_2add"
    assert transformer.representation == "game"
    assert transformer.k_add is None
    
    # Test custom initialization
    transformer = ChoquetTransformer(method="mlm", representation="mobius", k_add=2)
    assert transformer.method == "mlm"
    assert transformer.representation == "mobius"
    assert transformer.k_add == 2


def test_choquet_transformer_fit_transform(classification_data):
    """Test fit and transform methods of ChoquetTransformer."""
    X_train, _, y_train, _ = classification_data
    
    # Test with default parameters
    transformer = ChoquetTransformer()
    transformer.fit(X_train, y_train)
    X_transformed = transformer.transform(X_train)
    
    # Check that the transformed data has the right shape
    # For choquet_2add, we expect n_features + n_features*(n_features-1)/2 output features
    expected_features = N_FEATURES + N_FEATURES * (N_FEATURES - 1) // 2
    assert X_transformed.shape == (X_train.shape[0], expected_features)
    
    # Test with different method - full choquet with k_add=2
    transformer = ChoquetTransformer(method="choquet", k_add=2)
    transformer.fit(X_train, y_train)
    X_transformed = transformer.transform(X_train)
    
    # Calculate expected number of features for k=2
    # We expect all coalitions up to size 2
    expected_features = sum(math.comb(N_FEATURES, i) for i in range(1, 3))
    assert X_transformed.shape == (X_train.shape[0], expected_features)
    
    # Test with MLM method
    transformer = ChoquetTransformer(method="mlm", k_add=2)
    transformer.fit(X_train, y_train)
    X_transformed = transformer.transform(X_train)
    
    # MLM with k=2 should have the same number of features as choquet with k=2
    assert X_transformed.shape == (X_train.shape[0], expected_features)


def test_choquet_transformer_feature_names(classification_data):
    """Test the get_feature_names_out method."""
    X_train, _, _, _ = classification_data
    
    # Create transformer and fit
    transformer = ChoquetTransformer(method="choquet_2add")
    transformer.fit(X_train)
    
    # Get feature names
    feature_names = transformer.get_feature_names_out()
    
    # For choquet_2add, we expect n_features + n_features*(n_features-1)/2 feature names
    expected_count = N_FEATURES + N_FEATURES * (N_FEATURES - 1) // 2
    assert len(feature_names) == expected_count
    
    # The names should be strings
    assert all(isinstance(name, str) for name in feature_names)


# ChoquisticRegression tests
@pytest.mark.parametrize("model_class", [
    ChoquisticRegression_Composition,
    ChoquisticRegression_Inheritance
])
def test_choquistic_regression_init(model_class):
    """Test initialization of ChoquisticRegression classes."""
    # Test default initialization
    model = model_class()
    # Unified check for transformer_
    assert hasattr(model, 'transformer_')

    # Test with custom parameters
    model = model_class(method="mlm", k_add=2, C=0.5)
    # Check that transformer_ is set as expected
    assert model.transformer_.method == "mlm"
    assert model.transformer_.k_add == 2
    # Also check the LogisticRegression param
    assert model.C == 0.5


@pytest.mark.parametrize("model_class", [
    ChoquisticRegression_Composition,
    ChoquisticRegression_Inheritance
])
def test_choquistic_regression_fit_predict(classification_data, model_class):
    """Test fitting and prediction with ChoquisticRegression classes."""
    X_train, X_test, y_train, y_test = classification_data
    
    # Create and fit the model
    model = model_class(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    # Check predictions
    y_pred = model.predict(X_test)
    assert y_pred.shape == y_test.shape
    assert set(np.unique(y_pred)).issubset({0, 1})  # Binary classification
    
    # Check probability predictions
    y_proba = model.predict_proba(X_test)
    assert y_proba.shape == (len(y_test), 2)  # Binary classification: probs for 0 and 1
    assert np.all(y_proba >= 0) and np.all(y_proba <= 1)  # Valid probabilities
    
    # Basic accuracy check - should be better than random
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.5  # Better than random for this synthetic dataset


@pytest.mark.parametrize("model_class", [
    ChoquisticRegression_Composition, 
    ChoquisticRegression_Inheritance
])
def test_choquistic_regression_pipeline(classification_data, model_class):
    """Test ChoquisticRegression classes in a scikit-learn pipeline."""
    X_train, X_test, y_train, y_test = classification_data
    
    # Create a pipeline with the model
    pipeline = Pipeline([
        ('model', model_class(random_state=RANDOM_STATE))
    ])
    
    # Fit and predict
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Check predictions
    assert y_pred.shape == y_test.shape
    assert set(np.unique(y_pred)).issubset({0, 1})
    
    # Check that we can get the underlying model
    model = pipeline.named_steps['model']
    assert isinstance(model, model_class)


@pytest.mark.parametrize("model_class", [
    ChoquisticRegression_Composition,
    ChoquisticRegression_Inheritance
])
def test_choquistic_regression_methods(classification_data, model_class):
    """Test various methods of ChoquisticRegression classes."""
    X_train, X_test, y_train, _ = classification_data
    
    # Create and fit the model
    model = model_class(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    # Test score method
    score = model.score(X_train, y_train)
    assert 0 <= score <= 1  # Score should be between 0 and 1
    
    # Test get_params and set_params
    params = model.get_params()
    assert isinstance(params, dict)
    
    # Set a parameter and check it was updated
    if model_class == ChoquisticRegression_Composition:
        model.set_params(regressor__C=0.5)  # Use nested parameter syntax
    else:  # Inheritance version
        model.set_params(C=0.5)  # If it inherits from LogisticRegression
    if model_class == ChoquisticRegression_Composition:
        assert model.regressor.C == 0.5
    else:  # Inheritance version
        assert model.C == 0.5


# Edge cases and error handling
def test_choquet_transformer_edge_cases():
    """Test ChoquetTransformer with edge cases."""
    # Test with a single feature
    X = np.random.random((10, 1))
    transformer = ChoquetTransformer()
    transformer.fit(X)
    X_transformed = transformer.transform(X)
    assert X_transformed.shape == (10, 1)  # Should keep the single feature
    
    # Test with invalid method
    with pytest.raises(ValueError):
        ChoquetTransformer(method="invalid_method")
    
    # Test with invalid representation
    with pytest.raises(ValueError):
        ChoquetTransformer(representation="invalid_repr")
    
    # Test with k_add > n_features
    X = np.random.random((10, 3))
    transformer = ChoquetTransformer(method="choquet", k_add=5)
    # Should warn or adjust k_add automatically
    transformer.fit(X)
    assert transformer.k_add <= 3


@pytest.mark.parametrize("model_class", [
    ChoquisticRegression_Composition,
    ChoquisticRegression_Inheritance
])
def test_model_performance_comparison(classification_data, model_class):
    """Compare model performance with different parameters."""
    X_train, X_test, y_train, y_test = classification_data
    
    results = []
    
    # Test different methods and k_add values
    for method in ["choquet", "choquet_2add", "mlm", "mlm_2add"]:
        for k_add in [1, 2, None]:
            if method.endswith("_2add") and k_add is not None:
                continue  # Skip k_add parameter for _2add methods
                
            # Create and fit the model
            model = model_class(
                method=method,
                k_add=k_add,
                random_state=RANDOM_STATE
            )
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Try to get ROC AUC if predict_proba is available
                auc = None
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                except:
                    pass
                
                results.append({
                    'model_class': model_class.__name__,
                    'method': method,
                    'k_add': k_add,
                    'accuracy': accuracy,
                    'auc': auc
                })
            except Exception as e:
                # Log any errors but continue testing
                results.append({
                    'model_class': model_class.__name__,
                    'method': method,
                    'k_add': k_add,
                    'error': str(e)
                })
    
    # Verify we have results
    assert len(results) > 0
    
    # Check that at least one configuration works well
    accuracies = [r.get('accuracy', 0) for r in results if 'accuracy' in r]
    assert len(accuracies) > 0
    assert max(accuracies) > 0.5  # Should be better than random guessing


if __name__ == "__main__":
    pytest.main()