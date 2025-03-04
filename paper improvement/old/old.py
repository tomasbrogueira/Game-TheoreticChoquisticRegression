import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import comb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Set SCIPY_ARRAY_API and pandas downcasting option BEFORE any other imports
os.environ["SCIPY_ARRAY_API"] = "1"
pd.set_option('future.no_silent_downcasting', True)

# Make sure plots folder exists.
os.makedirs("plots", exist_ok=True)

# Import our organized Choquistic module and the data reader module.
# (Assume the organized module is saved in a file called "paper_improve.py".)
from paper_improve import ChoquisticRegression, ChoquetTransformer
import mod_GenFuzzyRegression as modGF

# ----------------------------
# 1. Load and normalize the data
# ----------------------------
data_imp = 'dados_covid_sbpo_atual'
X, y = modGF.func_read_data(data_imp)

# Normalize to [0,1]
if isinstance(X, pd.DataFrame):
    X = (X - X.min()) / (X.max() - X.min())
else:
    X = (X - X.min()) / (X.max() - X.min())

# ----------------------------
# 2. Split the data into train/test sets
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

# ----------------------------
# 3. Baseline Logistic Regression (on original features)
# ----------------------------
lr_baseline = LogisticRegression(random_state=0, penalty=None, solver='newton-cg', max_iter=10000)
lr_baseline.fit(X_train, y_train)
baseline_train_acc = lr_baseline.score(X_train, y_train)
baseline_test_acc  = lr_baseline.score(X_test, y_test)
print("Baseline LR Train Acc: {:.2%}, Test Acc: {:.2%}".format(baseline_train_acc, baseline_test_acc))

# Create a dictionary to store all results and extra data.
results = {
    'LR': {
        'train_acc': baseline_train_acc,
        'test_acc': baseline_test_acc,
        'coef': lr_baseline.coef_
    }
}

# ----------------------------
# 4. Choquistic Models using various transformations
# ----------------------------
# We try four methods: "choquet_2add", "choquet", "mlm", and "mlm_2add".
methods = ["choquet_2add", "choquet", "mlm", "mlm_2add"]

for method in methods:
    print("Processing method:", method)
    model = ChoquisticRegression(
        method=method,
        logistic_params={'penalty': None},
        scale_data=True,
        random_state=0
    )
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc  = model.score(X_test, y_test)
    
    # Store basic metrics.
    results[method] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'coef': model.classifier_.coef_
    }
    
    # For the full choquet and choquet_2add methods, compute Shapley values.
    if method in ["choquet", "choquet_2add"]:
        try:
            shapley_vals = model.compute_shapley_values()
            results[method]['shapley'] = shapley_vals
            print("Shapley values for method {}: {}".format(method, shapley_vals))
        except Exception as e:
            print("Could not compute shapley values for method {}: {}".format(method, e))
    
    print("Method: {:12s} | Train Acc: {:.2%} | Test Acc: {:.2%}".format(method, train_acc, test_acc))

# ----------------------------
# 5. Plot Marginal Contributions Histogram using Shapley values
# ----------------------------
# We choose to plot the marginal contributions for the 2-additive version if available.
if "choquet_2add" in results and 'shapley' in results["choquet_2add"]:
    shapley_vals = results["choquet_2add"]['shapley']
    # Determine feature names.
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        nAttr = X.shape[1]
        feature_names = [f"Feature {i}" for i in range(nAttr)]
    # Create a horizontal bar plot of the Shapley values (marginal contributions).
    ordered_indices = np.argsort(shapley_vals)[::-1]
    ordered_names = np.array(feature_names)[ordered_indices]
    ordered_values = shapley_vals[ordered_indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(ordered_names, ordered_values, color='skyblue', edgecolor='black')
    plt.xlabel("Shapley Value (Marginal Contribution)", fontsize=16)
    plt.title("Marginal Contributions from Choquet 2-additive Model", fontsize=18)
    plt.gca().invert_yaxis()  # Highest contributions at the top
    plt.tight_layout()
    marginal_plot_path = os.path.join("plots", "marginal_contribution_shapley.png")
    plt.savefig(marginal_plot_path)
    plt.close()
    print("Saved marginal contribution histogram (Shapley) to:", marginal_plot_path)
else:
    print("Shapley values not available for choquet_2add; skipping marginal contributions plot.")

# ----------------------------
# 6. Plot Interaction Effects for 2-additive model (using raw logistic coefficients)
# ----------------------------
# Use the choquet_2add model again for interpretation.
model_ch2add = ChoquisticRegression(method="choquet_2add", logistic_params={'penalty': None}, random_state=0)
model_ch2add.fit(X_train, y_train)
coef = model_ch2add.classifier_.coef_[0]  # shape: (n_transformed_features,)

# In the 2-additive transformation, the first nAttr coefficients correspond to singletons.
if isinstance(X, pd.DataFrame):
    feature_names = X.columns.tolist()
else:
    nAttr = X.shape[1]
    feature_names = [f"Feature {i}" for i in range(nAttr)]
nAttr = len(feature_names)
marginal_coef = coef[:nAttr]
interaction_coef = coef[nAttr:]  # remaining coefficients for pairwise interactions

# Map the remaining coefficients into an interaction matrix.
n_pairs = comb(nAttr, 2)
interaction_matrix = np.zeros((nAttr, nAttr))
idx = 0
for i in range(nAttr):
    for j in range(i+1, nAttr):
        interaction_matrix[i, j] = interaction_coef[idx]
        interaction_matrix[j, i] = interaction_coef[idx]
        idx += 1

plt.figure(figsize=(8, 6))
plt.imshow(interaction_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(orientation="vertical")
plt.xticks(range(nAttr), feature_names, rotation=90, fontsize=12)
plt.yticks(range(nAttr), feature_names, fontsize=12)
plt.title("Interaction Effects (log-odds) from Choquet 2-additive Model", fontsize=16)
plt.tight_layout()
interaction_plot_path = os.path.join("plots", "interaction_effects.png")
plt.savefig(interaction_plot_path)
plt.close()
print("Saved interaction effects plot to:", interaction_plot_path)

# ----------------------------
# 7. Compute and plot log-odds for test set samples
# ----------------------------
# We use the choquet_2add model to compute the decision function (log-odds) for the entire test set.
if model_ch2add.scale_data:
    X_test_scaled = model_ch2add.scaler_.transform(X_test)
else:
    X_test_scaled = X_test
X_test_transformed = model_ch2add.transformer_.transform(X_test_scaled)
log_odds_test = model_ch2add.classifier_.decision_function(X_test_transformed)
# Also store the predicted probabilities.
probs_test = model_ch2add.classifier_.predict_proba(X_test_transformed)

# Plot a histogram of the log-odds.
plt.figure(figsize=(10, 6))
plt.hist(log_odds_test, bins=30, color='lightgreen', edgecolor='black')
plt.xlabel("Log-Odds", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.title("Histogram of Log-Odds (Choquet 2-additive Model)", fontsize=18)
plt.tight_layout()
log_odds_plot_path = os.path.join("plots", "log_odds_histogram.png")
plt.savefig(log_odds_plot_path)
plt.close()
print("Saved log-odds histogram to:", log_odds_plot_path)

# Also, for illustration, plot a scatter of log-odds vs predicted probabilities.
plt.figure(figsize=(10, 6))
plt.scatter(log_odds_test, probs_test[:, 1], alpha=0.7, color='coral', edgecolor='k')
plt.xlabel("Log-Odds", fontsize=16)
plt.ylabel("Predicted Probability (Positive Class)", fontsize=16)
plt.title("Log-Odds vs. Predicted Probability", fontsize=18)
plt.tight_layout()
log_odds_prob_plot_path = os.path.join("plots", "log_odds_vs_prob.png")
plt.savefig(log_odds_prob_plot_path)
plt.close()
print("Saved log-odds vs predicted probability plot to:", log_odds_prob_plot_path)

# ----------------------------
# 8. (Optional) Plot decision boundaries for 2D data
# ----------------------------
def plot_decision_boundary(X, y, model, filename="plots/decision_boundary.png"):
    from matplotlib.colors import ListedColormap
    X = np.array(X)
    y = np.array(y)
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    
    if X.shape[1] != 2:
        print("Decision boundary plot only works for 2D data.")
        return
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision Boundary")
    plt.savefig(filename)
    plt.close()
    print("Saved decision boundary plot to:", filename)

# If the data are 2D, plot decision boundaries for baseline LR and choquistic model.
if X.shape[1] == 2:
    plot_decision_boundary(X_train, y_train, lr_baseline, filename="plots/decision_boundary_lr.png")
    plot_decision_boundary(X_train, y_train, model_ch2add, filename="plots/decision_boundary_choquistic.png")

# ----------------------------
# 9. Save overall results for later inspection.
# ----------------------------
# Store additional interesting information:
results['choquet_2add_extra'] = {
    'log_odds_test': log_odds_test,
    'predicted_probabilities_test': probs_test
}
# (You can expand this dictionary with any extra info you want to analyze later.)

with open("results.pkl", "wb") as f:
    pickle.dump(results, f)
print("Saved overall results to results.pkl")
