"""
Train and evaluate QSAR models for aqueous solubility prediction.

Trains two models on RDKit descriptors from data/esol_features.csv:
  - Linear Regression (interpretable baseline)
  - Random Forest Regressor (nonlinear model)

Evaluates both with an 80/20 train/test split and 5-fold cross-validation
(R^2 and RMSE), saves the better model to models/solubility_model.joblib,
and writes a predicted-vs-actual scatter plot to results/.
"""

import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

FEATURES_CSV = os.path.join("data", "esol_features.csv")
MODEL_PATH = os.path.join("models", "solubility_model.joblib")
METRICS_CSV = os.path.join("results", "model_performance.csv")
PLOT_PATH = os.path.join("results", "predicted_vs_actual.png")

FEATURE_COLS = [
    "mol_weight", "logp", "tpsa", "num_h_donors", "num_h_acceptors",
    "num_rotatable_bonds", "num_aromatic_rings", "num_rings",
    "num_heavy_atoms", "fraction_csp3", "molar_refractivity",
]
TARGET_COL = "log_solubility"


def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    print(f"  {name}: test RMSE={rmse:.3f}, test R^2={r2:.3f}, "
          f"5-fold CV R^2={cv_r2.mean():.3f} (+/- {cv_r2.std():.3f})")
    return {
        "model": name,
        "test_rmse": round(rmse, 3),
        "test_r2": round(r2, 3),
        "cv_r2_mean": round(cv_r2.mean(), 3),
        "cv_r2_std": round(cv_r2.std(), 3),
    }, preds


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(FEATURES_CSV)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training on {len(X_train)} compounds, testing on {len(X_test)}")

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    }

    results = []
    predictions = {}
    for name, model in models.items():
        metrics, preds = evaluate(name, model, X_train, X_test, y_train, y_test)
        results.append(metrics)
        predictions[name] = preds

    results_df = pd.DataFrame(results)
    results_df.to_csv(METRICS_CSV, index=False)

    best_name = results_df.sort_values("test_r2", ascending=False).iloc[0]["model"]
    best_model = models[best_name]
    joblib.dump({"model": best_model, "feature_cols": FEATURE_COLS}, MODEL_PATH)
    print(f"\nBest model: {best_name} -> saved to {MODEL_PATH}")

    # Predicted vs actual plot for the best model
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, predictions[best_name], alpha=0.6, edgecolor="k", linewidth=0.3)
    lims = [min(y_test.min(), predictions[best_name].min()),
            max(y_test.max(), predictions[best_name].max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Measured log solubility (mol/L)")
    ax.set_ylabel("Predicted log solubility (mol/L)")
    ax.set_title(f"{best_name}: Predicted vs. Measured Solubility")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"Wrote plot -> {PLOT_PATH}")
    print(f"Wrote metrics -> {METRICS_CSV}")


if __name__ == "__main__":
    main()
