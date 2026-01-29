from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
print(housing.DESCR) 
X = housing.frame.drop(columns=["MedHouseVal"])
y = housing.frame["MedHouseVal"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(random_state=42,
                   hidden_layer_sizes=(10,5),
                   alpha=1e-3, 
                   max_iter=200,
                   batch_size=1000,
                   activation="relu",
                   validation_fraction=0.2,
                   early_stopping=True) # important!
mlp.fit(X_train_scaled, y_train)
# 5) Predict on all splits
y_pred_train = mlp.predict(X_train_scaled)
y_pred_test  = mlp.predict(X_test_scaled)

# 7) Scatterplots: predicted vs actual (one figure per split; y=x reference line)
import numpy as np
import matplotlib.pyplot as plt

def scatter_with_reference(y_true, y_pred, title, filename):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    plt.plot([lo, hi], [lo, hi], linewidth=1, color='red')  # reference line
    plt.xlabel("Actual MedHouseVal")
    plt.ylabel("Predicted MedHouseVal")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()
    plt.close()

scatter_with_reference(y_train, y_pred_train, "Predicted vs Actual — Train", "figs/pred_vs_actual_train.png")
scatter_with_reference(y_test,  y_pred_test,  "Predicted vs Actual — Test", "figs/pred_vs_actual_test.png")