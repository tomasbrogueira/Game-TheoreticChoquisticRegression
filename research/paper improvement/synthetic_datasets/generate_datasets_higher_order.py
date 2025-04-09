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

def enhance_class_separation(y_continuous, margin=0.1):
    """Create binary target with margin around decision boundary"""
    threshold = np.median(y_continuous)
    
    y_binary = np.zeros(len(y_continuous))
    y_binary[y_continuous > threshold] = 1
    
    # Create mask to exclude samples near the boundary
    keep_mask = np.abs(y_continuous - threshold) > margin * np.std(y_continuous)
    
    return y_binary.astype(int), keep_mask

# 1. Quadruplet interactions dataset
def generate_quadruplet_interaction_dataset(n_samples=1000, n_features=20,
                                           feature_var=1.5, noise_level=0.25, margin=0.1):
    """Generate dataset with strong four-way feature interactions"""
    X = np.random.normal(0, feature_var, size=(n_samples, n_features))
    y = np.zeros(n_samples)
    
    # Create quadruplet interactions with strong coefficients
    for i in range(0, n_features-3, 4):
        if i+3 < n_features:
            y += 5.0 * (X[:, i] * X[:, i+1] * X[:, i+2] * X[:, i+3])
    
    y += np.random.normal(0, noise_level, size=n_samples)
    
    # Apply class separation with margin
    y_binary, keep_mask = enhance_class_separation(y, margin)
    
    if margin > 0:
        X = X[keep_mask]
        y_binary = y_binary[keep_mask]
        n_samples = len(y_binary)
    
    description = (
        f"Quadruplet interaction dataset with {n_features} features and {n_samples} samples. "
        f"Feature variance={feature_var}, noise={noise_level}, margin={margin}."
    )
    
    df = save_dataset(X, y_binary, "quadruplet_interaction_dataset.csv", description)
    return X, y_binary, df

# 2. Higher-order interactions dataset
def generate_higher_order_interaction_dataset(n_samples=1000, n_features=25, order=5,
                                            feature_var=1.5, noise_level=0.2, margin=0.1):
    """Generate dataset with strong feature interactions of specified order"""
    X = np.random.normal(0, feature_var, size=(n_samples, n_features))
    y = np.zeros(n_samples)
    
    # Scale coefficient based on interaction order
    interaction_coefficient = 2.0 * order
    
    # Create higher-order interactions
    for i in range(0, n_features-order+1, order):
        if i+order-1 < n_features:
            interaction = np.ones(n_samples)
            for j in range(order):
                interaction *= X[:, i+j]
            y += interaction_coefficient * interaction
    
    y += np.random.normal(0, noise_level, size=n_samples)
    
    # Apply class separation with margin
    y_binary, keep_mask = enhance_class_separation(y, margin)
    
    if margin > 0:
        X = X[keep_mask]
        y_binary = y_binary[keep_mask]
        n_samples = len(y_binary)
    
    description = (
        f"{order}-way interaction dataset with {n_features} features and {n_samples} samples. "
        f"Feature variance={feature_var}, noise={noise_level}, margin={margin}."
    )
    
    df = save_dataset(X, y_binary, f"higher_order_{order}_interaction_dataset.csv", description)
    return X, y_binary, df

# 3. Mixed higher-order interactions dataset
def generate_mixed_higher_order_dataset(n_samples=1000, n_features=30,
                                      feature_var=1.5, noise_level=0.25, margin=0.1):
    """Generate dataset with a mix of interactions from order 2 up to order 6"""
    X = np.random.normal(0, feature_var, size=(n_samples, n_features))
    y = np.zeros(n_samples)
    
    # Pairwise interactions (first 4 features)
    if n_features >= 4:
        y += 2.0 * X[:, 0] * X[:, 1]
        y += 2.0 * X[:, 2] * X[:, 3]
    
    # Triplet interactions (next 6 features)
    if n_features >= 10:
        y += 3.0 * X[:, 4] * X[:, 5] * X[:, 6]
        y += 3.0 * X[:, 7] * X[:, 8] * X[:, 9]
    
    # Quadruplet interactions (next 8 features)
    if n_features >= 18:
        y += 4.0 * X[:, 10] * X[:, 11] * X[:, 12] * X[:, 13]
        y += 4.0 * X[:, 14] * X[:, 15] * X[:, 16] * X[:, 17]
    
    # 5-way interactions (next 5 features)
    if n_features >= 23:
        interaction = np.ones(n_samples)
        for j in range(18, 23):
            interaction *= X[:, j]
        y += 5.0 * interaction
    
    # 6-way interactions (next 6 features)
    if n_features >= 29:
        interaction = np.ones(n_samples)
        for j in range(23, 29):
            interaction *= X[:, j]
        y += 6.0 * interaction
    
    y += np.random.normal(0, noise_level, size=n_samples)
    
    # Apply class separation with margin
    y_binary, keep_mask = enhance_class_separation(y, margin)
    
    if margin > 0:
        X = X[keep_mask]
        y_binary = y_binary[keep_mask]
        n_samples = len(y_binary)
    
    description = (
        f"Mixed higher-order interaction dataset with {n_features} features and {n_samples} samples. "
        f"Feature variance={feature_var}, noise={noise_level}, margin={margin}."
    )
    
    df = save_dataset(X, y_binary, "mixed_higher_order_interaction_dataset.csv", description)
    return X, y_binary, df

# 4. Exponentially weighted interactions dataset
def generate_exponentially_weighted_dataset(n_samples=1000, n_features=25,
                                          feature_var=1.5, noise_level=0.25, margin=0.1):
    """Generate dataset with interactions weighted by 2^(order)"""
    X = np.random.normal(0, feature_var, size=(n_samples, n_features))
    y = np.zeros(n_samples)
    
    # Individual effects (weight = 1)
    for i in range(5):
        y += 1.0 * X[:, i]
    
    # Pairwise interactions (weight = 2)
    for i in range(5, 9, 2):
        y += 2.0 * X[:, i] * X[:, i+1]
    
    # Triplet interactions (weight = 4)
    for i in range(9, 18, 3):
        if i+2 < n_features:
            y += 4.0 * X[:, i] * X[:, i+1] * X[:, i+2]
    
    # Quadruplet interactions (weight = 8)
    if n_features >= 22:
        y += 8.0 * X[:, 18] * X[:, 19] * X[:, 20] * X[:, 21]
    
    # 5-way interaction (weight = 16)
    if n_features >= 25:
        y += 16.0 * X[:, 20] * X[:, 21] * X[:, 22] * X[:, 23] * X[:, 24]
    
    y += np.random.normal(0, noise_level, size=n_samples)
    
    # Apply class separation with margin
    y_binary, keep_mask = enhance_class_separation(y, margin)
    
    if margin > 0:
        X = X[keep_mask]
        y_binary = y_binary[keep_mask]
        n_samples = len(y_binary)
    
    description = (
        f"Exponentially weighted interaction dataset with {n_features} features and {n_samples} samples. "
        f"Feature variance={feature_var}, noise={noise_level}, margin={margin}."
    )
    
    df = save_dataset(X, y_binary, "exponentially_weighted_interaction_dataset.csv", description)
    return X, y_binary, df

# 5. High-dimensional complex interactions dataset
def generate_high_dimensional_dataset(n_samples=1000, n_features=50,
                                    feature_var=1.5, noise_level=0.2, margin=0.1):
    """Generate high-dimensional dataset with complex interactions up to 8-way"""
    X = np.random.normal(0, feature_var, size=(n_samples, n_features))
    y = np.zeros(n_samples)
    
    # Individual effects for first 5 features
    for i in range(5):
        y += 0.5 * X[:, i]
    
    # Pairwise interactions for next 10 features
    for i in range(5, 15, 2):
        y += 2.0 * X[:, i] * X[:, i+1]
    
    # Triplet interactions for next 9 features
    for i in range(15, 24, 3):
        y += 3.0 * X[:, i] * X[:, i+1] * X[:, i+2]
    
    # Quadruplet interactions for next 8 features
    for i in range(24, 32, 4):
        y += 4.0 * X[:, i] * X[:, i+1] * X[:, i+2] * X[:, i+3]
    
    # 5-way interactions for next 10 features
    for i in range(32, 42, 5):
        if i+4 < n_features:
            interaction = np.ones(n_samples)
            for j in range(5):
                interaction *= X[:, i+j]
            y += 5.0 * interaction
    
    # 8-way interaction for remaining features
    if n_features >= 50:
        interaction = np.ones(n_samples)
        for j in range(42, 50):
            interaction *= X[:, j]
        y += 8.0 * interaction
    
    y += np.random.normal(0, noise_level, size=n_samples)
    
    # Apply class separation with margin
    y_binary, keep_mask = enhance_class_separation(y, margin)
    
    if margin > 0:
        X = X[keep_mask]
        y_binary = y_binary[keep_mask]
        n_samples = len(y_binary)
    
    description = (
        f"High-dimensional complex interaction dataset with {n_features} features and {n_samples} samples. "
        f"Feature variance={feature_var}, noise={noise_level}, margin={margin}."
    )
    
    df = save_dataset(X, y_binary, "high_dimensional_complex_dataset.csv", description)
    return X, y_binary, df

# 6. Nested interactions dataset
def generate_nested_interaction_dataset(n_samples=1000, n_features=25,
                                      feature_var=1.5, noise_level=0.25, margin=0.1):
    """Generate dataset with interactions-of-interactions patterns"""
    X = np.random.normal(0, feature_var, size=(n_samples, n_features))
    y = np.zeros(n_samples)
    
    # Level 1: Pairwise products
    p1 = X[:, 0] * X[:, 1]
    p2 = X[:, 2] * X[:, 3]
    p3 = X[:, 4] * X[:, 5]
    p4 = X[:, 6] * X[:, 7]
    
    # Level 2: Interactions between pairwise products
    i1 = p1 * p2
    i2 = p3 * p4
    
    # Level 3: Interaction between level 2 interactions
    y += 3.0 * i1 * i2
    
    # Another nested structure
    t1 = X[:, 8] * X[:, 9] * X[:, 10]
    t2 = X[:, 11] * X[:, 12] * X[:, 13]
    y += 3.0 * t1 * t2
    
    # Cross-level interactions
    f1 = X[:, 14]
    f2 = X[:, 15]
    y += 2.0 * f1 * p1
    y += 2.5 * f2 * t1
    
    # Deep nested structure (depth 3)
    d1 = X[:, 16] * X[:, 17]
    d2 = X[:, 18] * X[:, 19]
    d3 = d1 * d2
    d4 = X[:, 20] * X[:, 21]
    d5 = d3 * d4
    y += 4.0 * d5
    
    # Individual effects for remaining features
    for i in range(22, n_features):
        y += 0.3 * X[:, i]
    
    y += np.random.normal(0, noise_level, size=n_samples)
    
    # Apply class separation with margin
    y_binary, keep_mask = enhance_class_separation(y, margin)
    
    if margin > 0:
        X = X[keep_mask]
        y_binary = y_binary[keep_mask]
        n_samples = len(y_binary)
    
    description = (
        f"Nested interaction dataset with {n_features} features and {n_samples} samples. "
        f"Feature variance={feature_var}, noise={noise_level}, margin={margin}."
    )
    
    df = save_dataset(X, y_binary, "nested_interaction_dataset.csv", description)
    return X, y_binary, df

# 7. Extreme coalition dataset
def generate_extreme_coalition_dataset(n_samples=1000, n_features=40,
                                      feature_var=1.5, noise_level=0.2, margin=0.1):
    """Generate dataset with one extreme interaction involving all features"""
    X = np.random.normal(0, feature_var, size=(n_samples, n_features))
    y = np.zeros(n_samples)
    
    # Create the all-feature interaction (scaled to prevent overflow)
    extreme_interaction = np.ones(n_samples)
    for i in range(n_features):
        extreme_interaction *= (X[:, i] * 0.2)
    
    # Add the extreme interaction with a weight
    y += 5.0 * extreme_interaction
    
    # Add some individual effects for balance
    for i in range(10):
        y += 0.5 * X[:, i]
    
    # Add some pairwise effects
    for i in range(10, 20, 2):
        y += 0.8 * X[:, i] * X[:, i+1]
    
    y += np.random.normal(0, noise_level, size=n_samples)
    
    # Apply class separation with margin
    y_binary, keep_mask = enhance_class_separation(y, margin)
    
    if margin > 0:
        X = X[keep_mask]
        y_binary = y_binary[keep_mask]
        n_samples = len(y_binary)
    
    description = (
        f"Extreme coalition dataset with {n_features} features and {n_samples} samples. "
        f"Feature variance={feature_var}, noise={noise_level}, margin={margin}."
    )
    
    df = save_dataset(X, y_binary, "extreme_coalition_dataset.csv", description)
    return X, y_binary, df

def generate_enhanced_datasets():
    """Generate all higher-order synthetic datasets"""
    os.makedirs('plots', exist_ok=True)
    
    print("Generating quadruplet interaction dataset...")
    _, _, df1 = generate_quadruplet_interaction_dataset(feature_var=1.5, noise_level=0.25, margin=0.1)
    analyze_dataset(df1, "quadruplet_interaction")
    
    print("\nGenerating 5-way interaction dataset...")
    _, _, df2 = generate_higher_order_interaction_dataset(order=5, feature_var=1.5, noise_level=0.2, margin=0.1)
    analyze_dataset(df2, "higher_order_5_interaction")
    
    print("\nGenerating 6-way interaction dataset...")
    _, _, df3 = generate_higher_order_interaction_dataset(n_features=30, order=6, feature_var=1.5, noise_level=0.15, margin=0.1)
    analyze_dataset(df3, "higher_order_6_interaction")
    
    print("\nGenerating mixed higher-order interaction dataset...")
    _, _, df4 = generate_mixed_higher_order_dataset(feature_var=1.5, noise_level=0.25, margin=0.1)
    analyze_dataset(df4, "mixed_higher_order_interaction")
    
    print("\nGenerating exponentially weighted interaction dataset...")
    _, _, df5 = generate_exponentially_weighted_dataset(feature_var=1.5, noise_level=0.25, margin=0.1)
    analyze_dataset(df5, "exponentially_weighted_interaction")
    
    print("\nGenerating high-dimensional complex interaction dataset...")
    _, _, df6 = generate_high_dimensional_dataset(feature_var=1.5, noise_level=0.2, margin=0.1)
    analyze_dataset(df6, "high_dimensional_complex")
    
    print("\nGenerating nested interaction dataset...")
    _, _, df7 = generate_nested_interaction_dataset(feature_var=1.5, noise_level=0.25, margin=0.1)
    analyze_dataset(df7, "nested_interaction")
    
    print("\nGenerating extreme coalition dataset...")
    _, _, df8 = generate_extreme_coalition_dataset(feature_var=1.5, noise_level=0.2, margin=0.1)
    analyze_dataset(df8, "extreme_coalition")
    
    print("\nAll higher-order datasets generated successfully!")

if __name__ == "__main__":
    generate_enhanced_datasets()
