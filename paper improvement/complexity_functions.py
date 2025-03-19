import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from regression_classes import ChoquisticRegression
import psutil
import os
import matplotlib.pyplot as plt



def clone_model(model):
    if isinstance(model, ChoquisticRegression):
        return ChoquisticRegression(**model.get_params())
    elif isinstance(model, LogisticRegression):
        return LogisticRegression(**model.get_params())

def measure_training_time(model, X, y, n_runs=5):
    """
    Measure the average time it takes to train a model.
    
    Parameters:
        model: Model instance with fit method
        X: Input features
        y: Target values
        n_runs: Number of runs to average over
        
    Returns:
        float: Average training time in seconds
    """
    times = []
    for _ in range(n_runs):
        model_copy = clone_model(model)
        start_time = time.time()
        model_copy.fit(X, y)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)

def measure_prediction_time(model, X, n_runs=5):
    """
    Measure the average time it takes to make predictions.
    
    Parameters:
        model: Trained model instance with predict method
        X: Input features for prediction
        n_runs: Number of runs to average over
        
    Returns:
        float: Average prediction time in seconds
    """
    times = []
    for _ in range(n_runs):
        start_time = time.time()
        _ = model.predict(X)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)


def estimate_memory_usage(model):
    """
    Estimate the memory usage of a trained model.
    
    Parameters:
        model: Trained model instance
        
    Returns:
        float: Memory usage in MB
    """
    # Store the model in memory using pickle serialization
    import pickle
    model_bytes = pickle.dumps(model)
    model_size = len(model_bytes) / (1024 * 1024)  # Size in MB
    
    return model_size
    
def compute_model_complexities(models, X_train, y_train, X_test, labels=None):
    """
    Compare the time and space complexity of different models.
    
    Parameters:
        models: List of model instances
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        labels: List of labels for the models
        
    Returns:
        dict: Complexity metrics for each model
    """
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(models))]
    
    results = {}
    
    for model, label in zip(models, labels):
        print(f"Analyzing {label}...")
        
        # Time complexity - training
        train_time, train_std = measure_training_time(model, X_train, y_train)
        
        # Fit the model for prediction tests
        model.fit(X_train, y_train)
        
        # Time complexity - prediction
        pred_time, pred_std = measure_prediction_time(model, X_test)
        
        # Space complexity
        memory_usage = estimate_memory_usage(model)

        # Analyse scaling behavior
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_train], axis=0)
        scaling_results = analyze_scaling_behavior(model, X, y)
        
        # Parameter count if available
        try:
            n_params = model.coef_.size
        except (AttributeError, ValueError):
            n_params = None
            
        results[label] = {
            "train_time": train_time,
            "train_std": train_std,
            "pred_time": pred_time,
            "pred_std": pred_std,
            "memory_mb": memory_usage,
            "n_params": n_params,
            "scaling_results": scaling_results
        }
        
    return results

def analyze_scaling_behavior(model, X, y, test_sizes=[0.2, 0.4, 0.6, 0.8]):
    """
    Analyze how model training time scales with dataset size.
    
    Parameters:
        model: Model instance
        X: Input features
        y: Target values
        test_sizes: List of dataset proportions to test
        
    Returns:
        dict: Training times for different dataset sizes
    """
    results = {}
    n_samples = X.shape[0]
    
    for size in test_sizes:
        n = int(n_samples * size)
        X_subset = X[:n]
        y_subset = y[:n]
        
        train_time, _ = measure_training_time(model, X_subset, y_subset, n_runs=3)
        results[n] = train_time
        
    return results

def plot_complexity_results(complexity_results, title="Model Complexity Comparison"):
    """
    Plot the complexity comparison results.
    
    Parameters:
        complexity_results: Dict returned by compare_model_complexities
        title: Plot title
    """
    labels = list(complexity_results.keys())
    train_times = [complexity_results[label]["train_time"] for label in labels]
    train_stds = [complexity_results[label]["train_std"] for label in labels]
    pred_times = [complexity_results[label]["pred_time"] for label in labels]
    pred_stds = [complexity_results[label]["pred_std"] for label in labels]
    memory = [complexity_results[label]["memory_mb"] for label in labels]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Training time
    ax1.bar(labels, train_times, yerr=train_stds)
    ax1.set_ylabel("Seconds")
    ax1.set_title("Training Time")
    ax1.tick_params(axis='x', rotation=45)
    
    # Prediction time
    ax2.bar(labels, pred_times, yerr=pred_stds)
    ax2.set_ylabel("Seconds")
    ax2.set_title("Prediction Time")
    ax2.tick_params(axis='x', rotation=45)
    
    # Memory usage
    ax3.bar(labels, memory)
    ax3.set_ylabel("MB")
    ax3.set_title("Model Size")
    ax3.tick_params(axis='x', rotation=45)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig

def plot_scaling_behavior(scaling_results, model_name="Model"):
    """
    Plot how training time scales with dataset size.
    
    Parameters:
        scaling_results: Dict with dataset sizes as keys and times as values
        model_name: Name of the model
    """
    sizes = list(scaling_results.keys())
    times = list(scaling_results.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, 'o-')
    plt.xlabel("Dataset Size (samples)")
    plt.ylabel("Training Time (seconds)")
    plt.title(f"Scaling Behavior: {model_name}")
    plt.grid(True)
    
    # Fit a polynomial regression to see the scaling pattern
    coeffs = np.polyfit(sizes, times, 2)
    poly = np.poly1d(coeffs)
    
    x_range = np.linspace(min(sizes), max(sizes), 100)
    plt.plot(x_range, poly(x_range), '--', label=f"Fitted curve (degree 2)")
    
    plt.legend()
    return plt.gcf()