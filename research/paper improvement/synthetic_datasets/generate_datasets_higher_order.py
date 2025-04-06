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

# 1. Quadruplet Interaction Dataset
def generate_quadruplet_interaction_dataset(n_samples=1000, n_features=20):
    """
    Generate dataset with quadruplet interactions between features.
    This dataset has strong interactions between groups of four features.
    """
    # Generate base features
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Create binary outcome based on quadruplet interactions
    y = np.zeros(n_samples)
    
    # Add quadruplet interactions that influence the outcome
    for i in range(0, n_features-3, 4):
        if i+3 < n_features:
            # Create interaction terms between quadruplets of features
            interaction = X[:, i] * X[:, i+1] * X[:, i+2] * X[:, i+3]
            y += interaction
    
    # Add some noise
    y += np.random.normal(0, 0.5, size=n_samples)
    
    # Convert to binary classification
    y = (y > np.median(y)).astype(int)
    
    description = (
        "Quadruplet Interaction Dataset: Features exhibit strong interactions in groups of four. "
        f"Contains {n_features} features with {n_samples} samples. "
        "The target variable is determined by products of feature quadruplets."
    )
    
    df = save_dataset(X, y, "quadruplet_interaction_dataset.csv", description)
    return X, y, df

# 2. Higher-Order Interaction Dataset (5+ parameters)
def generate_higher_order_interaction_dataset(n_samples=1000, n_features=25, order=5):
    """
    Generate dataset with higher-order interactions between features.
    This dataset has strong interactions between groups of five or more features.
    """
    # Generate base features
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Create binary outcome based on higher-order interactions
    y = np.zeros(n_samples)
    
    # Add higher-order interactions that influence the outcome
    for i in range(0, n_features-order+1, order):
        if i+order-1 < n_features:
            # Create interaction terms between groups of 'order' features
            interaction = np.ones(n_samples)
            for j in range(order):
                interaction *= X[:, i+j]
            y += interaction
    
    # Add some noise
    y += np.random.normal(0, 0.5, size=n_samples)
    
    # Convert to binary classification
    y = (y > np.median(y)).astype(int)
    
    description = (
        f"Higher-Order Interaction Dataset: Features exhibit strong interactions in groups of {order}. "
        f"Contains {n_features} features with {n_samples} samples. "
        f"The target variable is determined by products of feature groups of size {order}."
    )
    
    df = save_dataset(X, y, f"higher_order_{order}_interaction_dataset.csv", description)
    return X, y, df

# 3. Mixed Higher-Order Interaction Dataset
def generate_mixed_higher_order_dataset(n_samples=1000, n_features=30):
    """
    Generate dataset with mixed higher-order interactions.
    This dataset has interactions of various orders (2, 3, 4, 5, and 6).
    """
    # Generate base features
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Create binary outcome based on mixed higher-order interactions
    y = np.zeros(n_samples)
    
    # Add pairwise interactions (first 4 features)
    if n_features >= 4:
        y += X[:, 0] * X[:, 1]
        y += X[:, 2] * X[:, 3]
    
    # Add triplet interactions (next 6 features)
    if n_features >= 10:
        y += X[:, 4] * X[:, 5] * X[:, 6]
        y += X[:, 7] * X[:, 8] * X[:, 9]
    
    # Add quadruplet interactions (next 8 features)
    if n_features >= 18:
        y += X[:, 10] * X[:, 11] * X[:, 12] * X[:, 13]
        y += X[:, 14] * X[:, 15] * X[:, 16] * X[:, 17]
    
    # Add 5-way interactions (next 5 features)
    if n_features >= 23:
        interaction = np.ones(n_samples)
        for j in range(18, 23):
            interaction *= X[:, j]
        y += interaction
    
    # Add 6-way interactions (next 6 features)
    if n_features >= 29:
        interaction = np.ones(n_samples)
        for j in range(23, 29):
            interaction *= X[:, j]
        y += 1.5 * interaction  # Weighted more heavily
    
    # Add some noise
    y += np.random.normal(0, 0.5, size=n_samples)
    
    # Convert to binary classification
    y = (y > np.median(y)).astype(int)
    
    description = (
        "Mixed Higher-Order Interaction Dataset: Features exhibit interactions of various orders (2, 3, 4, 5, and 6). "
        f"Contains {n_features} features with {n_samples} samples. "
        "The target variable is determined by a combination of different interaction orders."
    )
    
    df = save_dataset(X, y, "mixed_higher_order_interaction_dataset.csv", description)
    return X, y, df

# 4. Exponentially Weighted Interaction Dataset
def generate_exponentially_weighted_dataset(n_samples=1000, n_features=25):
    """
    Generate dataset with exponentially weighted interactions.
    This dataset has interactions where importance grows with coalition size.
    """
    # Generate base features
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Create binary outcome based on exponentially weighted interactions
    y = np.zeros(n_samples)
    
    # Add individual effects (weight = 1)
    for i in range(5):
        y += 1.0 * X[:, i]
    
    # Add pairwise interactions (weight = 2)
    for i in range(5, 9, 2):
        y += 2.0 * X[:, i] * X[:, i+1]
    
    # Add triplet interactions (weight = 4)
    for i in range(9, 18, 3):
        if i+2 < n_features:
            y += 4.0 * X[:, i] * X[:, i+1] * X[:, i+2]
    
    # Add quadruplet interactions (weight = 8)
    if n_features >= 22:
        y += 8.0 * X[:, 18] * X[:, 19] * X[:, 20] * X[:, 21]
    
    # Add 5-way interaction (weight = 16)
    if n_features >= 25:
        y += 16.0 * X[:, 20] * X[:, 21] * X[:, 22] * X[:, 23] * X[:, 24]
    
    # Add some noise
    y += np.random.normal(0, 0.5, size=n_samples)
    
    # Convert to binary classification
    y = (y > np.median(y)).astype(int)
    
    description = (
        "Exponentially Weighted Interaction Dataset: Features exhibit interactions with exponentially increasing importance. "
        f"Contains {n_features} features with {n_samples} samples. "
        "The target variable is determined by interactions with weights that double with each increase in coalition size."
    )
    
    df = save_dataset(X, y, "exponentially_weighted_interaction_dataset.csv", description)
    return X, y, df

# 5. High-Dimensional Complex Interaction Dataset
def generate_high_dimensional_dataset(n_samples=1000, n_features=50):
    """
    Generate high-dimensional dataset with complex interactions.
    This dataset has many features with various interaction patterns.
    """
    # Generate base features
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Create binary outcome based on complex interactions
    y = np.zeros(n_samples)
    
    # Add individual effects for first 5 features
    for i in range(5):
        y += 0.5 * X[:, i]
    
    # Add pairwise interactions for next 10 features
    for i in range(5, 15, 2):
        y += X[:, i] * X[:, i+1]
    
    # Add triplet interactions for next 9 features
    for i in range(15, 24, 3):
        y += X[:, i] * X[:, i+1] * X[:, i+2]
    
    # Add quadruplet interactions for next 8 features
    for i in range(24, 32, 4):
        y += X[:, i] * X[:, i+1] * X[:, i+2] * X[:, i+3]
    
    # Add 5-way interactions for next 10 features
    for i in range(32, 42, 5):
        if i+4 < n_features:
            interaction = np.ones(n_samples)
            for j in range(5):
                interaction *= X[:, i+j]
            y += interaction
    
    # Add 8-way interaction for remaining features
    if n_features >= 50:
        interaction = np.ones(n_samples)
        for j in range(42, 50):
            interaction *= X[:, j]
        y += 2.0 * interaction
    
    # Add some noise
    y += np.random.normal(0, 0.5, size=n_samples)
    
    # Convert to binary classification
    y = (y > np.median(y)).astype(int)
    
    description = (
        "High-Dimensional Complex Interaction Dataset: Features exhibit various interaction patterns in a high-dimensional space. "
        f"Contains {n_features} features with {n_samples} samples. "
        "The target variable is determined by a mix of individual effects and interactions of different orders (up to 8-way)."
    )
    
    df = save_dataset(X, y, "high_dimensional_complex_dataset.csv", description)
    return X, y, df

# 6. Nested Interaction Dataset
def generate_nested_interaction_dataset(n_samples=1000, n_features=25):
    """
    Generate dataset with nested interactions.
    This dataset has interactions within interactions.
    """
    # Generate base features
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Create binary outcome based on nested interactions
    y = np.zeros(n_samples)
    
    # Create nested interactions
    # Level 1: Pairwise products
    p1 = X[:, 0] * X[:, 1]
    p2 = X[:, 2] * X[:, 3]
    p3 = X[:, 4] * X[:, 5]
    p4 = X[:, 6] * X[:, 7]
    
    # Level 2: Interactions between pairwise products
    i1 = p1 * p2
    i2 = p3 * p4
    
    # Level 3: Interaction between level 2 interactions
    y += i1 * i2
    
    # Add another nested structure
    # Level 1: Triplet products
    t1 = X[:, 8] * X[:, 9] * X[:, 10]
    t2 = X[:, 11] * X[:, 12] * X[:, 13]
    
    # Level 2: Interaction between triplet products
    y += t1 * t2
    
    # Add a complex nested structure
    # Individual features
    f1 = X[:, 14]
    f2 = X[:, 15]
    
    # Pair them with products
    y += f1 * p1
    y += f2 * t1
    
    # Add a deep nested structure (depth 3)
    d1 = X[:, 16] * X[:, 17]
    d2 = X[:, 18] * X[:, 19]
    d3 = d1 * d2
    d4 = X[:, 20] * X[:, 21]
    d5 = d3 * d4
    y += d5
    
    # Add some individual effects for remaining features
    for i in range(22, n_features):
        y += 0.3 * X[:, i]
    
    # Add some noise
    y += np.random.normal(0, 0.5, size=n_samples)
    
    # Convert to binary classification
    y = (y > np.median(y)).astype(int)
    
    description = (
        "Nested Interaction Dataset: Features exhibit nested interactions (interactions within interactions). "
        f"Contains {n_features} features with {n_samples} samples. "
        "The target variable is determined by complex nested structures of interactions at multiple levels."
    )
    
    df = save_dataset(X, y, "nested_interaction_dataset.csv", description)
    return X, y, df

# 7. Extreme Coalition Dataset
def generate_extreme_coalition_dataset(n_samples=1000, n_features=40):
    """
    Generate dataset with an extreme coalition where all features interact.
    This dataset has one interaction term that includes all features.
    """
    # Generate base features
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Create binary outcome based on all-feature interaction
    y = np.zeros(n_samples)
    
    # Create the extreme coalition (all features interact)
    extreme_interaction = np.ones(n_samples)
    for i in range(n_features):
        extreme_interaction *= (X[:, i] * 0.2)  # Scale down to prevent numerical issues
    
    # Add the extreme interaction with a weight
    y += 5.0 * extreme_interaction
    
    # Add some individual effects to balance
    for i in range(10):
        y += 0.5 * X[:, i]
    
    # Add some pairwise effects
    for i in range(10, 20, 2):
        y += 0.8 * X[:, i] * X[:, i+1]
    
    # Add some noise
    y += np.random.normal(0, 0.5, size=n_samples)
    
    # Convert to binary classification
    y = (y > np.median(y)).astype(int)
    
    description = (
        "Extreme Coalition Dataset: Contains one interaction term that includes all features. "
        f"Contains {n_features} features with {n_samples} samples. "
        "The target variable is heavily influenced by a single interaction involving all features, "
        "along with some individual and pairwise effects."
    )
    
    df = save_dataset(X, y, "extreme_coalition_dataset.csv", description)
    return X, y, df

# Main function to generate all datasets
def generate_enhanced_datasets():
    """Generate all enhanced synthetic datasets"""
    os.makedirs('plots', exist_ok=True)
    
    # Generate datasets
    print("Generating quadruplet interaction dataset...")
    _, _, df1 = generate_quadruplet_interaction_dataset()
    analyze_dataset(df1, "quadruplet_interaction")
    
    print("\nGenerating higher-order (5) interaction dataset...")
    _, _, df2 = generate_higher_order_interaction_dataset(order=5)
    analyze_dataset(df2, "higher_order_5_interaction")
    
    print("\nGenerating higher-order (6) interaction dataset...")
    _, _, df3 = generate_higher_order_interaction_dataset(n_features=30, order=6)
    analyze_dataset(df3, "higher_order_6_interaction")
    
    print("\nGenerating mixed higher-order interaction dataset...")
    _, _, df4 = generate_mixed_higher_order_dataset()
    analyze_dataset(df4, "mixed_higher_order_interaction")
    
    print("\nGenerating exponentially weighted interaction dataset...")
    _, _, df5 = generate_exponentially_weighted_dataset()
    analyze_dataset(df5, "exponentially_weighted_interaction")
    
    print("\nGenerating high-dimensional complex interaction dataset...")
    _, _, df6 = generate_high_dimensional_dataset()
    analyze_dataset(df6, "high_dimensional_complex")
    
    print("\nGenerating nested interaction dataset...")
    _, _, df7 = generate_nested_interaction_dataset()
    analyze_dataset(df7, "nested_interaction")
    
    print("\nGenerating extreme coalition dataset...")
    _, _, df8 = generate_extreme_coalition_dataset()
    analyze_dataset(df8, "extreme_coalition")
    
    print("\nAll enhanced datasets generated successfully!")

if __name__ == "__main__":
    generate_enhanced_datasets()
