import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import comb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Make sure plots folder exists.
os.makedirs("plots", exist_ok=True)
os.environ["SCIPY_ARRAY_API"] = "1"

pd.set_option('future.no_silent_downcasting', True)

# Import our organized Choquistic module and the data reader module.
# (Assume the organized module is saved in a file called "choquistic.py".)
from paper_improve import ChoquisticRegression, ChoquetTransformer  # our module from the previous answer
import mod_GenFuzzyRegression as modGF

# ----------------------------
# 1. Load and normalize the data
# ----------------------------
data_imp =  'dados_covid_sbpo_atual'
X, y = modGF.func_read_data(data_imp)

# Normalize to [0,1]
if isinstance(X, pd.DataFrame):
    X = (X - X.min()) / (X.max() - X.min())
else:
    X = (X - X.min()) / (X.max() - X.min())

# ----------------------------
# 2. Split the data into train/test sets
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# ----------------------------
# 3. Baseline Logistic Regression (on original features)
# ----------------------------
lr_baseline = LogisticRegression(random_state=0, penalty=None, solver='newton-cg', max_iter=10000)
lr_baseline.fit(X_train, y_train)
baseline_train_acc = lr_baseline.score(X_train, y_train)
baseline_test_acc  = lr_baseline.score(X_test, y_test)
print("Baseline LR Train Acc: {:.2%}, Test Acc: {:.2%}".format(baseline_train_acc, baseline_test_acc))

results = {'LR': {'train_acc': baseline_train_acc,
                  'test_acc': baseline_test_acc,
                  'coef': lr_baseline.coef_}}

# ----------------------------
# 4. Choquistic Models using various transformations
# ----------------------------
methods = ["choquet_2add", "choquet", "mlm", "mlm_2add"]
for method in methods:
    model = ChoquisticRegression(method=method,
                                 logistic_params={'penalty': None},
                                 scale_data=True,
                                 random_state=0)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc  = model.score(X_test, y_test)
    results[method] = {'train_acc': train_acc,
                       'test_acc': test_acc,
                       'coef': model.classifier_.coef_}  # underlying logistic regression coefficients
    print("Method: {:12s} | Train Acc: {:.2%} | Test Acc: {:.2%}".format(method, train_acc, test_acc))

# ----------------------------
# 5. Plotting log-odds: Marginal contributions and interaction effects (using the choquet_2add model)
# ----------------------------
# We use the choquet_2add model for interpretation.
model_ch2add = ChoquisticRegression(method="choquet_2add", logistic_params={'penalty': None}, random_state=0)
model_ch2add.fit(X_train, y_train)
coef = model_ch2add.classifier_.coef_[0]  # shape: (n_transformed_features,)

# In our 2-additive transformation, the first nAttr coefficients correspond to the singleton (marginal) contributions.
if isinstance(X, pd.DataFrame):
    feature_names = X.columns.tolist()
else:
    nAttr = X.shape[1]
    feature_names = [f"Feature {i}" for i in range(nAttr)]
nAttr = len(feature_names)
marginal_coef = coef[:nAttr]
interaction_coef = coef[nAttr:]  # remaining coefficients for pairwise interactions

# Plot marginal contributions (interpreted as log-odds changes).
ordered_indices = np.argsort(marginal_coef)[::-1]
ordered_names = np.array(feature_names)[ordered_indices]
ordered_values = marginal_coef[ordered_indices]
plt.figure(figsize=(10, 8))
plt.barh(ordered_names, ordered_values, color='blue', edgecolor='black')
plt.xlabel("Marginal Contribution (log-odds)", fontsize=16)
plt.title("Marginal Contributions from Choquet 2-additive Model", fontsize=18)
plt.gca().invert_yaxis()  # highest values at the top
plt.tight_layout()
marginal_plot_path = os.path.join("plots", "marginal_contribution.png")
plt.savefig(marginal_plot_path)
plt.close()
print("Saved marginal contribution plot to:", marginal_plot_path)

# For interaction effects, we need to map the remaining coefficients to the corresponding feature pairs.
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
# 6. Compute log-odds for a sample input
# ----------------------------
# Take the first sample from the test set.
if isinstance(X_test, pd.DataFrame):
    X_sample = X_test.iloc[0:1]
else:
    X_sample = X_test[0:1]
# Apply the same scaling and transformation as in our model.
if model_ch2add.scale_data:
    X_sample_scaled = model_ch2add.scaler_.transform(X_sample)
else:
    X_sample_scaled = X_sample
X_sample_transformed = model_ch2add.transformer_.transform(X_sample_scaled)
# Compute decision function (log-odds)
log_odds = model_ch2add.classifier_.decision_function(X_sample_transformed)
prob_sample = model_ch2add.classifier_.predict_proba(X_sample_transformed)
print("Log-odds for a sample input:", log_odds)
print("Predicted probability for a sample input:", prob_sample)

# ----------------------------
# 7. (Optional) Plot decision boundaries for 2D data
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
# 8. Save overall results for later inspection.
# ----------------------------
with open("results.pkl", "wb") as f:
    pickle.dump(results, f)
print("Saved overall results to results.pkl")
