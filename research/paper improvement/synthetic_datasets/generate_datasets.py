import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import os

# Reproducible results
np.random.seed(42)

def save_dataset(X, y, filename, description):
    """Save dataset to CSV with description as a comment"""
    df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(X.shape[1])])
    df['target'] = y
    
    with open(filename, 'w') as f:
        f.write(f"# {description}\n")
    df.to_csv(filename, mode='a', index=False)
    
    print(f"Saved to {filename}")
    return df

def plot_feature_correlations(df, filename):
    """Plot and save a correlation heatmap"""
    plt.figure(figsize=(12, 10))
    features = df.drop('target', axis=1)
    corr = features.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_class_distribution(df, filename):
    """Plot and save class distribution"""
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df)
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def analyze_dataset(df, name):
    """Analyze dataset and save visualizations and statistics"""
    os.makedirs('plots', exist_ok=True)
    
    plot_feature_correlations(df, f'plots/{name}_correlation.png')
    plot_class_distribution(df, f'plots/{name}_class_distribution.png')
    
    # Calculate statistics
    stats = {
        'n_samples': len(df),
        'n_features': len(df.columns) - 1,
        'class_balance': df['target'].value_counts(normalize=True).to_dict(),
        'feature_means': df.drop('target', axis=1).mean().to_dict(),
        'feature_stds': df.drop('target', axis=1).std().to_dict()
    }
    
    with open(f'{name}_stats.txt', 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Analysis completed for {name}")
    return stats

def enhance_class_separation(y_continuous, margin=0.0):
    """Create binary target with margin around decision boundary"""
    threshold = np.median(y_continuous)
    
    # Print diagnostic information to understand threshold effect
    print(f"Median threshold: {threshold:.4f}")
    print(f"Target min: {y_continuous.min():.4f}, max: {y_continuous.max():.4f}")
    print(f"Target std dev: {np.std(y_continuous):.4f}")
    
    y_binary = np.zeros(len(y_continuous))
    y_binary[y_continuous > threshold] = 1
    
    # Create mask to exclude samples near the boundary
    keep_mask = np.abs(y_continuous - threshold) > margin * np.std(y_continuous)
    
    # Print class balance information
    class_balance = np.mean(y_binary)
    print(f"Binary class balance: {class_balance:.2f} / {1-class_balance:.2f}")
    print(f"Samples kept: {np.sum(keep_mask)} out of {len(keep_mask)} ({np.mean(keep_mask)*100:.1f}%)")
    
    return y_binary.astype(int), keep_mask

# 1. Pairwise interactions dataset
def generate_pairwise_interaction_dataset(n_samples=1000, n_features=15, 
                                          feature_var=1.5, noise_level=0.0, margin=0.00):
    """Generate dataset with strong pairwise feature interactions"""
    X = np.random.normal(0, feature_var, size=(n_samples, n_features))
    y = np.zeros(n_samples)
    
    # Create pairwise interactions with strong coefficients
    pair_contributions = []
    for i in range(0, n_features-1, 2):
        interaction = X[:, i] * X[:, i+1]
        # Use absolute values to prevent cancellation effects
        contribution = 2.5 * np.abs(interaction)
        y += contribution
        
        # Track the mean contribution of each pair for diagnostics
        pair_contributions.append((i, i+1, np.mean(contribution)))
    
    # Print contribution of each pair to verify equal impact
    print("Pair contributions to target variable:")
    for i, j, contrib in pair_contributions:
        print(f"Pair X{i+1}-X{j+1}: {contrib:.4f}")
    
    y += np.random.normal(0, noise_level, size=n_samples)
    
    # Apply class separation with margin
    y_binary, keep_mask = enhance_class_separation(y, margin)
    
    if margin > 0:
        X = X[keep_mask]
        y_binary = y_binary[keep_mask]
        n_samples = len(y_binary)
    
    description = (
        f"Pairwise interaction dataset with {n_features} features and {n_samples} samples. "
        f"Feature variance={feature_var}, noise={noise_level}, margin={margin}. "
        f"Using absolute interaction values to prevent cancellation."
    )
    
    df = save_dataset(X, y_binary, "pairwise_interaction_dataset.csv", description)
    return X, y_binary, df

# 2. Triplet interactions dataset
def generate_triplet_interaction_dataset(n_samples=1000, n_features=15,
                                        feature_var=1.5, noise_level=0.3, margin=0.0):
    """Generate dataset with strong triplet feature interactions"""
    X = np.random.normal(0, feature_var, size=(n_samples, n_features))
    y = np.zeros(n_samples)
    
    # Create triplet interactions with strong coefficients
    for i in range(0, n_features-2, 3):
        if i+2 < n_features:
            y += 3.5 * (X[:, i] * X[:, i+1] * X[:, i+2])
    
    y += np.random.normal(0, noise_level, size=n_samples)
    
    # Apply class separation with margin
    y_binary, keep_mask = enhance_class_separation(y, margin)
    
    if margin > 0:
        X = X[keep_mask]
        y_binary = y_binary[keep_mask]
        n_samples = len(y_binary)
    
    description = (
        f"Triplet interaction dataset with {n_features} features and {n_samples} samples. "
        f"Feature variance={feature_var}, noise={noise_level}, margin={margin}."
    )
    
    df = save_dataset(X, y_binary, "triplet_interaction_dataset.csv", description)
    return X, y_binary, df

# 3. Mixed-order interactions dataset
def generate_mixed_interaction_dataset(n_samples=1000, n_features=15,
                                     feature_var=1.5, noise_level=0.3, margin=0.0):
    """Generate dataset with mixed individual, pair, and triplet interactions"""
    X = np.random.normal(0, feature_var, size=(n_samples, n_features))
    y = np.zeros(n_samples)
    
    # Individual effects
    for i in range(3):
        y += 1.0 * X[:, i]
    
    # Pairwise interactions
    for i in range(3, 9, 2):
        y += 2.5 * (X[:, i] * X[:, i+1])
    
    # Triplet interactions
    for i in range(9, 15, 3):
        if i+2 < n_features:
            y += 3.5 * (X[:, i] * X[:, i+1] * X[:, i+2])
    
    y += np.random.normal(0, noise_level, size=n_samples)
    
    # Apply class separation with margin
    y_binary, keep_mask = enhance_class_separation(y, margin)
    
    if margin > 0:
        X = X[keep_mask]
        y_binary = y_binary[keep_mask]
        n_samples = len(y_binary)
    
    description = (
        f"Mixed interaction dataset with {n_features} features and {n_samples} samples. "
        f"Feature variance={feature_var}, noise={noise_level}, margin={margin}."
    )
    
    df = save_dataset(X, y_binary, "mixed_interaction_dataset.csv", description)
    return X, y_binary, df

# 4. Non-linear transformations dataset
def generate_nonlinear_transformation_dataset(n_samples=1000, n_features=15,
                                             feature_var=1.5, noise_level=0.3, margin=0.0):
    """Generate dataset with various non-linear feature transformations"""
    X = np.random.normal(0, feature_var, size=(n_samples, n_features))
    y = np.zeros(n_samples)
    
    # Various non-linear transformations
    y += 2.0 * np.sin(X[:, 0] * X[:, 1])  # Sine
    y += 2.0 * np.exp(X[:, 2] * 0.5)      # Exponential
    y += 2.0 * np.log(np.abs(X[:, 3]) + 1)  # Log
    y += 2.0 * np.maximum(0, X[:, 4])     # ReLU
    y += 2.0 * X[:, 5]**2                 # Quadratic
    y += 2.0 * X[:, 6]**3                 # Cubic
    y += 2.0 * np.tanh(X[:, 7])           # Tanh
    y += 2.0 * np.sign(X[:, 8]) * np.sqrt(np.abs(X[:, 8]))  # Signed sqrt
    
    # Pairwise interactions for remaining features
    for i in range(9, n_features-1, 2):
        if i+1 < n_features:
            y += 2.5 * (X[:, i] * X[:, i+1])
    
    y += np.random.normal(0, noise_level, size=n_samples)
    
    # Apply class separation with margin
    y_binary, keep_mask = enhance_class_separation(y, margin)
    
    if margin > 0:
        X = X[keep_mask]
        y_binary = y_binary[keep_mask]
        n_samples = len(y_binary)
    
    description = (
        f"Non-linear transformation dataset with {n_features} features and {n_samples} samples. "
        f"Feature variance={feature_var}, noise={noise_level}, margin={margin}."
    )
    
    df = save_dataset(X, y_binary, "nonlinear_transformation_dataset.csv", description)
    return X, y_binary, df

# 5. Hierarchical interactions dataset
def generate_hierarchical_interaction_dataset(n_samples=1000, n_features=15,
                                             feature_var=1.5, noise_level=0.3, margin=0.0):
    """Generate dataset with nested conditional interactions"""
    X = np.random.normal(0, feature_var, size=(n_samples, n_features))
    y = np.zeros(n_samples)
    
    # Define conditions
    condition1 = X[:, 0] > 0
    condition2 = X[:, 1] > 0
    
    # Base effect
    y += 0.5 * X[:, 2]
    
    # First-level conditional interaction
    mask1 = condition1
    y[mask1] += X[mask1, 3] * X[mask1, 4]
    
    # Second-level conditional interaction
    mask2 = condition1 & condition2
    y[mask2] += X[mask2, 5] * X[mask2, 6] * X[mask2, 7]
    
    # Third-level conditional interaction
    mask3 = ~condition1 & ~condition2
    for i in range(8, n_features):
        y[mask3] += 0.5 * X[mask3, i]
    
    y += np.random.normal(0, noise_level, size=n_samples)
    
    # Apply class separation with margin
    y_binary, keep_mask = enhance_class_separation(y, margin)
    
    if margin > 0:
        X = X[keep_mask]
        y_binary = y_binary[keep_mask]
        n_samples = len(y_binary)
    
    description = (
        f"Hierarchical interaction dataset with {n_features} features and {n_samples} samples. "
        f"Feature variance={feature_var}, noise={noise_level}, margin={margin}."
    )
    
    df = save_dataset(X, y_binary, "hierarchical_interaction_dataset.csv", description)
    return X, y_binary, df

# 6. Sparse high-dimensional interactions dataset
def generate_sparse_interaction_dataset(n_samples=1000, n_features=20,
                                       feature_var=1.5, noise_level=0.3, margin=0.0):
    """Generate dataset with many features but only a few relevant interactions"""
    X = np.random.normal(0, feature_var, size=(n_samples, n_features))
    y = np.zeros(n_samples)
    
    # Only 6 features are relevant
    relevant_features = [0, 3, 7, 12, 15, 18]
    
    # Individual effects
    y += 0.5 * X[:, relevant_features[0]]
    y += 0.7 * X[:, relevant_features[1]]
    
    # Pairwise interaction
    y += 2.5 * (X[:, relevant_features[2]] * X[:, relevant_features[3]])
    
    # Triplet interaction
    y += 3.5 * (X[:, relevant_features[3]] * X[:, relevant_features[4]] * X[:, relevant_features[5]])
    
    y += np.random.normal(0, noise_level, size=n_samples)
    
    # Apply class separation with margin
    y_binary, keep_mask = enhance_class_separation(y, margin)
    
    if margin > 0:
        X = X[keep_mask]
        y_binary = y_binary[keep_mask]
        n_samples = len(y_binary)
    
    description = (
        f"Sparse interaction dataset with {n_features} features and {n_samples} samples. "
        f"Only 6 features are relevant. Feature variance={feature_var}, noise={noise_level}, margin={margin}."
    )
    
    df = save_dataset(X, y_binary, "sparse_interaction_dataset.csv", description)
    return X, y_binary, df

# 7. Threshold effects dataset
def generate_threshold_interaction_dataset(n_samples=1000, n_features=15,
                                          feature_var=1.5, noise_level=0.3, margin=0.0):
    """Generate dataset with interactions that only occur above thresholds"""
    X = np.random.normal(0, feature_var, size=(n_samples, n_features))
    y = np.zeros(n_samples)
    
    # Define random thresholds
    thresholds = np.random.uniform(-0.5, 0.5, size=n_features)
    
    # Add threshold-dependent interactions
    for i in range(0, n_features-1, 2):
        if i+1 < n_features:
            # Interaction only matters when both features exceed their thresholds
            mask = (X[:, i] > thresholds[i]) & (X[:, i+1] > thresholds[i+1])
            y[mask] += 2.5 * (X[mask, i] * X[mask, i+1])
    
    # Individual effects for first 5 features
    for i in range(0, 5):
        y += 0.3 * X[:, i]
    
    y += np.random.normal(0, noise_level, size=n_samples)
    
    # Apply class separation with margin
    y_binary, keep_mask = enhance_class_separation(y, margin)
    
    if margin > 0:
        X = X[keep_mask]
        y_binary = y_binary[keep_mask]
        n_samples = len(y_binary)
    
    description = (
        f"Threshold interaction dataset with {n_features} features and {n_samples} samples. "
        f"Feature variance={feature_var}, noise={noise_level}, margin={margin}."
    )
    
    df = save_dataset(X, y_binary, "threshold_interaction_dataset.csv", description)
    return X, y_binary, df

def generate_all_datasets():
    """Generate all synthetic datasets"""
    os.makedirs('plots', exist_ok=True)
    
    print("Generating pairwise interaction dataset...")
    _, _, df1 = generate_pairwise_interaction_dataset(feature_var=1.5, noise_level=0.3, margin=0.0)
    analyze_dataset(df1, "pairwise_interaction")
    
    print("\nGenerating triplet interaction dataset...")
    _, _, df2 = generate_triplet_interaction_dataset(feature_var=1.5, noise_level=0.3, margin=0.0)
    analyze_dataset(df2, "triplet_interaction")
    
    print("\nGenerating mixed interaction dataset...")
    _, _, df3 = generate_mixed_interaction_dataset(feature_var=1.5, noise_level=0.3, margin=0.0)
    analyze_dataset(df3, "mixed_interaction")
    
    print("\nGenerating non-linear transformation dataset...")
    _, _, df4 = generate_nonlinear_transformation_dataset(feature_var=1.5, noise_level=0.3, margin=0.0)
    analyze_dataset(df4, "nonlinear_transformation")
    
    print("\nGenerating hierarchical interaction dataset...")
    _, _, df5 = generate_hierarchical_interaction_dataset(feature_var=1.5, noise_level=0.3, margin=0.0)
    analyze_dataset(df5, "hierarchical_interaction")
    
    print("\nGenerating sparse interaction dataset...")
    _, _, df6 = generate_sparse_interaction_dataset(n_features=20, feature_var=1.5, noise_level=0.3, margin=0.0)
    analyze_dataset(df6, "sparse_interaction")
    
    print("\nGenerating threshold interaction dataset...")
    _, _, df7 = generate_threshold_interaction_dataset(feature_var=1.5, noise_level=0.3, margin=0.0)
    analyze_dataset(df7, "threshold_interaction")
    
    print("\nAll datasets generated successfully!")

if __name__ == "__main__":
    generate_all_datasets()
