import time
import numpy as np
import platform
import pickle
import warnings

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from regression_classes import ChoquisticRegression


# Track availability of measurement tools
ENERGY_AVAILABLE = False
FLOPS_AVAILABLE = False

# Try to load direct measurement tools with proper platform detection
IS_LINUX = platform.system() == 'Linux'

# Energy measurement (Linux + Intel only)
if IS_LINUX:
    try:
        import pyRAPL
        pyRAPL.setup()
        ENERGY_AVAILABLE = True
    except (ImportError, FileNotFoundError, Exception) as e:
        warnings.warn(f"Energy measurement unavailable: {e}")

# FLOPS measurement
try:
    if IS_LINUX:
        import papi.events as papi_events
        import papi.low as papi
        FLOPS_AVAILABLE = True
    else:
        # Windows-specific counter if available
        pass
except ImportError:
    warnings.warn("FLOPS direct measurement unavailable")


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
    model_bytes = pickle.dumps(model)
    model_size = len(model_bytes) / (1024 * 1024)  # Size in MB
    
    return model_size
    
def compute_model_complexities(models, X_train, y_train, X_test, labels=None):
    """Compare time and space complexity of different models."""
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

        # FLOPS and energy consumption
        flops, energy, runtime, is_estimated = measure_model_energy_and_flops(model, X_test)

        # Scaling behavior
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
            "scaling_results": scaling_results,
            "flops": flops,
            "energy_uj": energy,
            "pred_runtime": runtime,
            "is_estimated": is_estimated
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
    Plot the complexity comparison results including energy and FLOPS metrics.
    
    Parameters:
        complexity_results: Dict returned by compare_model_complexities
        title: Plot title
    
    Returns:
        fig: Matplotlib figure object
    """
    labels = list(complexity_results.keys())
    train_times = [complexity_results[label]["train_time"] for label in labels]
    train_stds = [complexity_results[label]["train_std"] for label in labels]
    pred_times = [complexity_results[label]["pred_time"] for label in labels]
    pred_stds = [complexity_results[label]["pred_std"] for label in labels]
    memory = [complexity_results[label]["memory_mb"] for label in labels]
    
    # Check if we have energy and FLOPS data
    has_energy = all("energy_uj" in complexity_results[label] for label in labels)
    has_flops = all("flops" in complexity_results[label] for label in labels)
    
    # Determine number of subplots based on available metrics
    n_plots = 3  # Default: train time, pred time, memory
    if has_energy:
        n_plots += 1
    if has_flops:
        n_plots += 1
    
    # Create figure with appropriate number of subplots
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots*5, 5))
    
    # Training time
    axes[0].bar(labels, train_times, yerr=train_stds)
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Training Time")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Prediction time
    axes[1].bar(labels, pred_times, yerr=pred_stds)
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Prediction Time")
    axes[1].tick_params(axis='x', rotation=45)
    
    # Memory usage
    axes[2].bar(labels, memory)
    axes[2].set_ylabel("MB")
    axes[2].set_title("Model Size")
    axes[2].tick_params(axis='x', rotation=45)
    
    # Add FLOPS plot if available
    plot_idx = 3
    if has_flops:
        flops_values = [complexity_results[label]["flops"] for label in labels]
        # Convert to millions of FLOPS for better readability
        flops_values_m = [flop/1e6 for flop in flops_values]
        axes[plot_idx].bar(labels, flops_values_m)
        axes[plot_idx].set_ylabel("MFLOPS")
        axes[plot_idx].set_title("Computational Complexity")
        axes[plot_idx].tick_params(axis='x', rotation=45)
        plot_idx += 1
    
    # Add energy consumption plot if available
    if has_energy:
        energy_values = []
        for label in labels:
            # Handle None values for systems without energy measurement
            energy = complexity_results[label].get("energy_uj")
            energy_values.append(energy if energy is not None else 0)
            
        # Convert to millijoules for better readability if values are large
        if any(e > 1000 for e in energy_values if e is not None):
            energy_values_mj = [e/1000 for e in energy_values]
            energy_unit = "mJ"
        else:
            energy_values_mj = energy_values
            energy_unit = "μJ"
            
        axes[plot_idx].bar(labels, energy_values_mj)
        axes[plot_idx].set_ylabel(energy_unit)
        axes[plot_idx].set_title("Energy Consumption")
        axes[plot_idx].tick_params(axis='x', rotation=45)
    
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

def theoretical_flops_estimation(model, X):
    """
    Theoretical estimation of FLOPS for prediction.
    This is used when hardware counters aren't available.
    """
    n_samples, n_features = X.shape
    
    # Base matrix operations common to all models
    base_ops = n_samples * n_features  
    
    if hasattr(model, 'method'):  # ChoquisticRegression
        method = model.method.lower()  # Normalize to lowercase for consistent checking
        
        # Full Choquet models have exponential complexity
        if 'choquet' in method and '2add' not in method:
            # Exponential complexity capped to prevent overflow
            # For full Choquet integral without 2-additivity restriction
            n_feat_capped = min(n_features, 20)  # Reasonable cap
            ops = base_ops + n_samples * (2**n_feat_capped * 10)
            
        # 2-additive Choquet models have quadratic complexity
        elif 'choquet' in method and '2add' in method:
            # Quadratic complexity in number of features
            ops = base_ops + n_samples * (n_features**2 * 5)
            
        # MLM full models
        elif 'mlm' in method and '2add' not in method:
            # Full MLM also has exponential operations
            n_feat_capped = min(n_features, 20)
            ops = base_ops + n_samples * (2**n_feat_capped * 8)
            
        # MLM 2-additive models
        elif 'mlm' in method and '2add' in method:
            # 2-additive MLM has quadratic complexity
            ops = base_ops + n_samples * (n_features**2 * 4)
            
        # Default/unknown model
        else:
            ops = base_ops * 5
            
    else:  # Standard LogisticRegression
        # Matrix multiply + activation function
        ops = 2 * base_ops + 10 * n_samples
    
    return ops

def measure_model_energy_and_flops(model, X):
    """
    Measure energy usage and FLOPS with proper platform detection.
    Falls back to theoretical estimation when direct measurement unavailable.
    
    Returns:
        flops: Floating point operations (measured or estimated)
        energy: Energy consumption in microjoules (None if unavailable)
        runtime: Execution time in seconds
        is_estimated: Dictionary indicating which metrics are estimates
    """
    is_estimated = {
        "flops": True,
        "energy": True
    }
    
    # Direct FLOPS measurement (if available)
    flops = None
    if FLOPS_AVAILABLE and IS_LINUX:
        try:
            papi.library_init()
            eventset = papi.create_eventset()
            papi.add_event(eventset, papi_events.PAPI_FP_OPS)
            
            papi.start(eventset)
            model.predict(X)
            counts = papi.stop(eventset)
            flops = counts[0]
            papi.cleanup_eventset(eventset)
            papi.destroy_eventset(eventset)
            
            is_estimated["flops"] = False
        except Exception as e:
            warnings.warn(f"FLOPS measurement failed: {e}")
    
    # Fall back to theoretical estimation
    if flops is None:
        flops = theoretical_flops_estimation(model, X)
        # Print for debugging
        model_type = getattr(model, 'method', 'LogisticRegression')
        print(f"Estimated FLOPS for {model_type}: {flops:,}")
    
    # Always measure runtime (this is reliable across platforms)
    start_time = time.time()
    _ = model.predict(X)
    runtime = time.time() - start_time

    # Energy measurement (Linux + Intel only)
    energy = None
    if ENERGY_AVAILABLE:
        try:
            # Create measurement wrapper function
            @pyRAPL.Measurement
            def energy_run():
                return model.predict(X)
                
            # Run measurement
            measurement = energy_run()
            # Convert joules to microjoules
            energy = measurement.result * 1e6
            is_estimated["energy"] = False
        except Exception as e:
            warnings.warn(f"Energy measurement failed: {e}")
    
    return flops, energy, runtime, is_estimated
