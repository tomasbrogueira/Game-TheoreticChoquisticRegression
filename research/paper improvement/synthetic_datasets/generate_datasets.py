import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import os

# Set random seed for reproducibility
np.random.seed(42)

def save_dataset(X, y, filename, description):
    """Save dataset to CSV with description in header"""
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(X.shape[1])])
    df['target'] = y
    
    # Save to CSV with description as comment
    with open(filename, 'w') as f:
        f.write(f"# {description}\n")
    df.to_csv(filename, mode='a', index=False)
    
    print(f"Dataset saved to {filename}")
    return df

def plot_feature_correlations(df, filename):
    """Plot correlation heatmap for features"""
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
    print(f"Correlation plot saved to {filename}")

def plot_class_distribution(df, filename):
    """Plot class distribution"""
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df)
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Class distribution plot saved to {filename}")

def analyze_dataset(df, name):
    """Analyze dataset and save plots"""
    os.makedirs('plots', exist_ok=True)
    
    # Plot correlation heatmap
    plot_feature_correlations(df, f'plots/{name}_correlation.png')
    
    # Plot class distribution
    plot_class_distribution(df, f'plots/{name}_class_distribution.png')
    
    # Basic statistics
    stats = {
        'n_samples': len(df),
        'n_features': len(df.columns) - 1,
        'class_balance': df['target'].value_counts(normalize=True).to_dict(),
        'feature_means': df.drop('target', axis=1).mean().to_dict(),
        'feature_stds': df.drop('target', axis=1).std().to_dict()
    }
    
    # Save statistics
    with open(f'{name}_stats.txt', 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Analysis completed for {name}")
    return stats

# 1. Dataset with pairwise interactions
def generate_pairwise_interaction_dataset(n_samples=1000, n_features=15):
    """
    Generate dataset with pairwise interactions between features.
    This dataset has strong interactions between pairs of features.
    """
    # Generate base features
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Create binary outcome based on pairwise interactions
    y = np.zeros(n_samples)
    
    # Add pairwise interactions that influence the outcome
    for i in range(0, n_features-1, 2):
        # Create interaction terms between consecutive features
        interaction = X[:, i] * X[:, i+1]
        y += interaction
    
    # Add some noise
    y += np.random.normal(0, 0.5, size=n_samples)
    
    # Convert to binary classification
    y = (y > np.median(y)).astype(int)
    
    description = (
        "Pairwise Interaction Dataset: Features exhibit strong pairwise interactions. "
        f"Contains {n_features} features with {n_samples} samples. "
        "The target variable is determined by products of consecutive feature pairs."
    )
    
    df = save_dataset(X, y, "pairwise_interaction_dataset.csv", description)
    return X, y, df

# 2. Dataset with higher-order interactions (triplets)
def generate_triplet_interaction_dataset(n_samples=1000, n_features=15):
    """
    Generate dataset with triplet interactions between features.
    This dataset has strong interactions between groups of three features.
    """
    # Generate base features
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Create binary outcome based on triplet interactions
    y = np.zeros(n_samples)
    
    # Add triplet interactions that influence the outcome
    for i in range(0, n_features-2, 3):
        if i+2 < n_features:
            # Create interaction terms between triplets of features
            interaction = X[:, i] * X[:, i+1] * X[:, i+2]
            y += interaction
    
    # Add some noise
    y += np.random.normal(0, 0.5, size=n_samples)
    
    # Convert to binary classification
    y = (y > np.median(y)).astype(int)
    
    description = (
        "Triplet Interaction Dataset: Features exhibit strong triplet interactions. "
        f"Contains {n_features} features with {n_samples} samples. "
        "The target variable is determined by products of feature triplets."
    )
    
    df = save_dataset(X, y, "triplet_interaction_dataset.csv", description)
    return X, y, df

# 3. Dataset with mixed-order interactions
def generate_mixed_interaction_dataset(n_samples=1000, n_features=15):
    """
    Generate dataset with mixed-order interactions (pairs, triplets, and individual).
    This dataset has interactions of different orders to test the model's ability to
    capture different coalition sizes.
    """
    # Generate base features
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Create binary outcome based on mixed interactions
    y = np.zeros(n_samples)
    
    # Add individual feature effects (first 3 features)
    for i in range(3):
        y += 0.5 * X[:, i]
    
    # Add pairwise interactions (next 6 features)
    for i in range(3, 9, 2):
        interaction = X[:, i] * X[:, i+1]
        y += interaction
    
    # Add triplet interactions (next 6 features)
    for i in range(9, 15, 3):
        if i+2 < n_features:
            interaction = X[:, i] * X[:, i+1] * X[:, i+2]
            y += 1.5 * interaction
    
    # Add some noise
    y += np.random.normal(0, 0.5, size=n_samples)
    
    # Convert to binary classification
    y = (y > np.median(y)).astype(int)
    
    description = (
        "Mixed Interaction Dataset: Features exhibit interactions of different orders. "
        f"Contains {n_features} features with {n_samples} samples. "
        "The target variable is determined by individual features, pairs, and triplets."
    )
    
    df = save_dataset(X, y, "mixed_interaction_dataset.csv", description)
    return X, y, df

# 4. Dataset with non-linear transformations
def generate_nonlinear_transformation_dataset(n_samples=1000, n_features=15):
    """
    Generate dataset with non-linear transformations of features.
    This dataset applies various non-linear functions to features.
    """
    # Generate base features
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Create binary outcome based on non-linear transformations
    y = np.zeros(n_samples)
    
    # Add non-linear transformations
    y += np.sin(X[:, 0] * X[:, 1])  # Sine of product
    y += np.exp(X[:, 2] * 0.5)  # Exponential
    y += np.log(np.abs(X[:, 3]) + 1)  # Logarithm
    y += np.maximum(0, X[:, 4])  # ReLU-like
    y += X[:, 5]**2  # Quadratic
    y += X[:, 6]**3  # Cubic
    y += np.tanh(X[:, 7])  # Hyperbolic tangent
    y += np.sign(X[:, 8]) * np.sqrt(np.abs(X[:, 8]))  # Signed square root
    
    # Add pairwise interactions for remaining features
    for i in range(9, n_features-1, 2):
        if i+1 < n_features:
            y += X[:, i] * X[:, i+1]
    
    # Add some noise
    y += np.random.normal(0, 0.5, size=n_samples)
    
    # Convert to binary classification
    y = (y > np.median(y)).astype(int)
    
    description = (
        "Non-linear Transformation Dataset: Features undergo various non-linear transformations. "
        f"Contains {n_features} features with {n_samples} samples. "
        "The target variable is determined by non-linear functions of features including "
        "sine, exponential, logarithm, ReLU, polynomial, and hyperbolic tangent."
    )
    
    df = save_dataset(X, y, "nonlinear_transformation_dataset.csv", description)
    return X, y, df

# 5. Dataset with hierarchical interactions
def generate_hierarchical_interaction_dataset(n_samples=1000, n_features=15):
    """
    Generate dataset with hierarchical interactions between features.
    This dataset has nested interactions where some interactions only matter
    when other conditions are met.
    """
    # Generate base features
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Create binary outcome based on hierarchical interactions
    y = np.zeros(n_samples)
    
    # Create hierarchical conditions
    condition1 = X[:, 0] > 0  # First condition
    condition2 = X[:, 1] > 0  # Second condition
    
    # Base effect
    y += 0.5 * X[:, 2]
    
    # First-level interaction: only applies when condition1 is true
    mask1 = condition1
    y[mask1] += X[mask1, 3] * X[mask1, 4]
    
    # Second-level interaction: only applies when both conditions are true
    mask2 = condition1 & condition2
    y[mask2] += X[mask2, 5] * X[mask2, 6] * X[mask2, 7]
    
    # Third-level interaction: applies when neither condition is true
    mask3 = ~condition1 & ~condition2
    for i in range(8, n_features):
        y[mask3] += 0.5 * X[mask3, i]
    
    # Add some noise
    y += np.random.normal(0, 0.5, size=n_samples)
    
    # Convert to binary classification
    y = (y > np.median(y)).astype(int)
    
    description = (
        "Hierarchical Interaction Dataset: Features exhibit nested conditional interactions. "
        f"Contains {n_features} features with {n_samples} samples. "
        "The target variable is determined by hierarchical conditions where some "
        "interactions only matter when other conditions are met."
    )
    
    df = save_dataset(X, y, "hierarchical_interaction_dataset.csv", description)
    return X, y, df

# 6. Dataset with high-dimensional sparse interactions
def generate_sparse_interaction_dataset(n_samples=1000, n_features=20):
    """
    Generate dataset with high-dimensional sparse interactions.
    This dataset has many features but only a few are relevant, with complex interactions.
    """
    # Generate base features (more features for this dataset)
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Create binary outcome based on sparse interactions
    y = np.zeros(n_samples)
    
    # Only a few features are relevant
    relevant_features = [0, 3, 7, 12, 15, 18]
    
    # Add individual effects for some relevant features
    y += 0.5 * X[:, relevant_features[0]]
    y += 0.7 * X[:, relevant_features[1]]
    
    # Add pairwise interaction
    y += X[:, relevant_features[2]] * X[:, relevant_features[3]]
    
    # Add triplet interaction
    y += X[:, relevant_features[3]] * X[:, relevant_features[4]] * X[:, relevant_features[5]]
    
    # Add some noise
    y += np.random.normal(0, 0.5, size=n_samples)
    
    # Convert to binary classification
    y = (y > np.median(y)).astype(int)
    
    description = (
        "Sparse Interaction Dataset: High-dimensional with sparse relevant interactions. "
        f"Contains {n_features} features with {n_samples} samples. "
        "Only 6 features are truly relevant, with the rest being noise. "
        "The target variable is determined by individual effects and interactions "
        "among the relevant features."
    )
    
    df = save_dataset(X, y, "sparse_interaction_dataset.csv", description)
    return X, y, df

# 7. Dataset with threshold effects
def generate_threshold_interaction_dataset(n_samples=1000, n_features=15):
    """
    Generate dataset with threshold effects in feature interactions.
    This dataset has interactions that only become relevant when features cross thresholds.
    """
    # Generate base features
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Create binary outcome based on threshold interactions
    y = np.zeros(n_samples)
    
    # Define thresholds
    thresholds = np.random.uniform(-0.5, 0.5, size=n_features)
    
    # Add threshold effects
    for i in range(0, n_features-1, 2):
        if i+1 < n_features:
            # Interaction only matters when both features exceed their thresholds
            mask = (X[:, i] > thresholds[i]) & (X[:, i+1] > thresholds[i+1])
            y[mask] += X[mask, i] * X[mask, i+1]
    
    # Add some individual effects
    for i in range(0, 5):
        y += 0.3 * X[:, i]
    
    # Add some noise
    y += np.random.normal(0, 0.5, size=n_samples)
    
    # Convert to binary classification
    y = (y > np.median(y)).astype(int)
    
    description = (
        "Threshold Interaction Dataset: Features interact only when crossing thresholds. "
        f"Contains {n_features} features with {n_samples} samples. "
        "The target variable is determined by interactions that only become relevant "
        "when features exceed certain threshold values."
    )
    
    df = save_dataset(X, y, "threshold_interaction_dataset.csv", description)
    return X, y, df

# Main function to generate all datasets
def generate_all_datasets():
    """Generate all synthetic datasets"""
    os.makedirs('plots', exist_ok=True)
    
    # Generate datasets
    print("Generating pairwise interaction dataset...")
    _, _, df1 = generate_pairwise_interaction_dataset()
    analyze_dataset(df1, "pairwise_interaction")
    
    print("\nGenerating triplet interaction dataset...")
    _, _, df2 = generate_triplet_interaction_dataset()
    analyze_dataset(df2, "triplet_interaction")
    
    print("\nGenerating mixed interaction dataset...")
    _, _, df3 = generate_mixed_interaction_dataset()
    analyze_dataset(df3, "mixed_interaction")
    
    print("\nGenerating non-linear transformation dataset...")
    _, _, df4 = generate_nonlinear_transformation_dataset()
    analyze_dataset(df4, "nonlinear_transformation")
    
    print("\nGenerating hierarchical interaction dataset...")
    _, _, df5 = generate_hierarchical_interaction_dataset()
    analyze_dataset(df5, "hierarchical_interaction")
    
    print("\nGenerating sparse interaction dataset...")
    _, _, df6 = generate_sparse_interaction_dataset(n_features=20)
    analyze_dataset(df6, "sparse_interaction")
    
    print("\nGenerating threshold interaction dataset...")
    _, _, df7 = generate_threshold_interaction_dataset()
    analyze_dataset(df7, "threshold_interaction")
    
    print("\nAll datasets generated successfully!")

if __name__ == "__main__":
    generate_all_datasets()
