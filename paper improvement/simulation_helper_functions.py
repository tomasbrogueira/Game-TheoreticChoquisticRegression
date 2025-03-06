import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_feature_names(X):
    """Extract feature names from DataFrame or create default names."""
    if isinstance(X, pd.DataFrame):
        return X.columns.tolist()
    else:
        return [f"F{i}" for i in range(X.shape[1])]

def plot_horizontal_bar(names, values, std=None, title="", xlabel="", filename="", color="steelblue"):
    """
    Plots a horizontal bar chart.
    
    Parameters:
      - names: list of feature names.
      - values: values to plot.
      - std: error bars (optional).
      - title: plot title.
      - xlabel: label for x-axis.
      - filename: path to save the figure.
      - color: bar color.
    """
    ordered_indices = np.argsort(values)[::-1]
    ordered_names = np.array(names)[ordered_indices]
    ordered_values = values[ordered_indices]
    ordered_std = std[ordered_indices] if std is not None else None

    plt.figure(figsize=(10, 8))
    plt.barh(ordered_names, ordered_values, xerr=ordered_std, color=color, edgecolor="black")
    plt.xlabel(xlabel, fontsize=16)
    plt.title(title, fontsize=18)
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print("Saved plot to:", filename)
