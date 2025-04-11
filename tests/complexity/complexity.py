import time
import numpy as np
import platform
import pickle
import warnings

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from core.models.regression import ChoquisticRegression 


# Track tool availability
ENERGY_AVAILABLE = False
FLOPS_AVAILABLE = False

# Platform detection
IS_LINUX = platform.system() == 'Linux'

# Try to set up energy measurement on Linux
if IS_LINUX:
    try:
        import pyRAPL
        pyRAPL.setup()
        ENERGY_AVAILABLE = True
    except (ImportError, FileNotFoundError, Exception) as e:
        warnings.warn(f"Energy measurement unavailable: {e}")

# Try to set up FLOPS counting
try:
    if IS_LINUX:
        import papi.events as papi_events
        import papi.low as papi
        FLOPS_AVAILABLE = True
    else:
        # No Windows counter available yet
        pass
except ImportError:
    warnings.warn("FLOPS direct measurement unavailable")


def clone_model(model):
    if isinstance(model, ChoquisticRegression):
        return ChoquisticRegression(**model.get_params())
    elif isinstance(model, LogisticRegression):
        return LogisticRegression(**model.get_params())

def measure_training_time(model, X, y, n_runs=5):
    """Measure average training time across multiple runs"""
    times = []
    for _ in range(n_runs):
        model_copy = clone_model(model)
        start_time = time.time()
        model_copy.fit(X, y)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)

def measure_prediction_time(model, X, n_runs=5):
    """Measure average prediction time across multiple runs"""
    times = []
    for _ in range(n_runs):
        start_time = time.time()
        _ = model.predict(X)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)


def estimate_memory_usage(model):
    """Estimate model size in MB using pickle serialization"""
    model_bytes = pickle.dumps(model)
    model_size = len(model_bytes) / (1024 * 1024)  # Convert to MB
    
    return model_size
    
def compute_model_complexities(models, X_train, y_train, X_test, labels=None):
    """Compare runtime, memory and computational complexity of models"""
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(models))]
    
    results = {}
    
    for model, label in zip(models, labels):
        print(f"Analyzing {label}...")
        
        # Measure training time
        train_time, train_std = measure_training_time(model, X_train, y_train)
        
        # Fit model for further tests
        model.fit(X_train, y_train)
        
        # Measure prediction time
        pred_time, pred_std = measure_prediction_time(model, X_test)
        
        # Measure memory usage
        memory_usage = estimate_memory_usage(model)

        # Get FLOPS and energy metrics
        flops, energy, runtime, is_estimated = measure_model_energy_and_flops(model, X_test)

        # Measure scaling behavior with dataset size
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_train], axis=0)
        scaling_results = analyze_scaling_behavior(model, X, y)
        
        # Get parameter count if available
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
    """Analyze how training time scales with increasing dataset size"""
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
    """Plot complexity metrics including time, memory, FLOPS and energy"""
    labels = list(complexity_results.keys())
    train_times = [complexity_results[label]["train_time"] for label in labels]
    train_stds = [complexity_results[label]["train_std"] for label in labels]
    pred_times = [complexity_results[label]["pred_time"] for label in labels]
    pred_stds = [complexity_results[label]["pred_std"] for label in labels]
    memory = [complexity_results[label]["memory_mb"] for label in labels]
    
    # Check available metrics
    has_energy = all("energy_uj" in complexity_results[label] for label in labels)
    has_flops = all("flops" in complexity_results[label] for label in labels)
    
    # Determine number of plots needed
    n_plots = 3  # Base: train time, pred time, memory
    if has_energy:
        n_plots += 1
    if has_flops:
        n_plots += 1
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots*5, 5))
    
    # Training time plot
    axes[0].bar(labels, train_times, yerr=train_stds)
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Training Time")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Prediction time plot
    axes[1].bar(labels, pred_times, yerr=pred_stds)
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Prediction Time")
    axes[1].tick_params(axis='x', rotation=45)
    
    # Memory usage plot
    axes[2].bar(labels, memory)
    axes[2].set_ylabel("MB")
    axes[2].set_title("Model Size")
    axes[2].tick_params(axis='x', rotation=45)
    
    # Add FLOPS plot if data available
    plot_idx = 3
    if has_flops:
        flops_values = [complexity_results[label]["flops"] for label in labels]
        # Convert to millions for readability
        flops_values_m = [flop/1e6 for flop in flops_values]
        axes[plot_idx].bar(labels, flops_values_m)
        axes[plot_idx].set_ylabel("MFLOPS")
        axes[plot_idx].set_title("Computational Complexity")
        axes[plot_idx].tick_params(axis='x', rotation=45)
        plot_idx += 1
    
    # Add energy plot if data available
    if has_energy:
        energy_values = []
        for label in labels:
            energy = complexity_results[label].get("energy_uj")
            energy_values.append(energy if energy is not None else 0)
            
        # Use appropriate units based on magnitude
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
    """Plot how training time scales with dataset size"""
    sizes = list(scaling_results.keys())
    times = list(scaling_results.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, 'o-')
    plt.xlabel("Dataset Size (samples)")
    plt.ylabel("Training Time (seconds)")
    plt.title(f"Scaling Behavior: {model_name}")
    plt.grid(True)
    
    # Fit curve to visualize trend
    coeffs = np.polyfit(sizes, times, 2)
    poly = np.poly1d(coeffs)
    
    x_range = np.linspace(min(sizes), max(sizes), 100)
    plt.plot(x_range, poly(x_range), '--', label=f"Fitted curve (degree 2)")
    
    plt.legend()
    return plt.gcf()

def theoretical_flops_estimation(model, X):
    """Estimate computational complexity when direct measurement isn't available"""
    n_samples, n_features = X.shape
    
    # Base operations for all models
    base_ops = n_samples * n_features  
    
    if hasattr(model, 'method'):  # ChoquisticRegression
        method = model.method.lower()
        
        # Full Choquet (exponential complexity)
        if 'choquet' in method and '2add' not in method:
            n_feat_capped = min(n_features, 20)  # Cap to avoid overflow
            ops = base_ops + n_samples * (2**n_feat_capped * 10)
            
        # 2-additive Choquet (quadratic complexity)
        elif 'choquet' in method and '2add' in method:
            ops = base_ops + n_samples * (n_features**2 * 5)
            
        # Full MLM (exponential complexity)
        elif 'mlm' in method and '2add' not in method:
            n_feat_capped = min(n_features, 20)
            ops = base_ops + n_samples * (2**n_feat_capped * 8)
            
        # 2-additive MLM (quadratic complexity)
        elif 'mlm' in method and '2add' in method:
            ops = base_ops + n_samples * (n_features**2 * 4)
            
        # Fallback
        else:
            ops = base_ops * 5
            
    else:  # Standard LogisticRegression
        ops = 2 * base_ops + 10 * n_samples
    
    return ops

def measure_model_energy_and_flops(model, X):
    """Measure or estimate computational resources for prediction"""
    is_estimated = {
        "flops": True,
        "energy": True
    }
    
    # Try direct FLOPS measurement if available
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
    
    # Fall back to estimation if needed
    if flops is None:
        flops = theoretical_flops_estimation(model, X)
        model_type = getattr(model, 'method', 'LogisticRegression')
        print(f"FLOPS for {model_type}: {flops:,}")
    
    # Measure runtime (works on all platforms)
    start_time = time.time()
    _ = model.predict(X)
    runtime = time.time() - start_time

    # Try energy measurement if available
    energy = None
    if ENERGY_AVAILABLE:
        try:
            @pyRAPL.Measurement
            def energy_run():
                return model.predict(X)
                
            measurement = energy_run()
            energy = measurement.result * 1e6  # Convert to microjoules
            is_estimated["energy"] = False
        except Exception as e:
            warnings.warn(f"Energy measurement failed: {e}")
    
    return flops, energy, runtime, is_estimated