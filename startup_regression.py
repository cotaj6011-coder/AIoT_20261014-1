"""50 Startups multiple linear regression using the CRISP-DM framework."""

from itertools import combinations
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import jarque_bera


DATA_PATH = Path("50_Startups.csv")
COMPARISON_PATH = Path("model_comparison.csv")
CHART_PATH = Path("model_comparison.png")
IMPORTANCE_PATH = Path("feature_importance.csv")
WEB_DATA_PATH = Path("dic-regression-dashboard/dist/data")
MODEL_DETAILS_PATH = WEB_DATA_PATH / "model_details.json"
CURVE_PATH = Path("feature_selection_curves.csv")
TARGET = "Profit"
FEATURE_GROUPS = ["R&D Spend", "Administration", "Marketing Spend", "State"]
RANDOM_STATE = 42
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def business_understanding() -> None:
    """Define the business objective and the modeling goal."""
    print("Business objective: predict startup profit from company spending and state.")


def data_understanding(path: Path) -> pd.DataFrame:
    """Load and summarize the dataset."""
    data = pd.read_csv(path)
    print("\nDataset shape:", data.shape)
    print("\nFirst five rows:\n", data.head())
    print("\nMissing values:\n", data.isna().sum())
    print("\nNumeric summary:\n", data.describe())
    return data


def data_preparation(data: pd.DataFrame):
    """Split the test set before any feature selection or model tuning."""
    X = data.drop(columns=TARGET)
    y = data[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def build_preprocessor(features: list[str]) -> ColumnTransformer:
    """Build preprocessing for a raw feature-group subset."""
    numeric_features = [feature for feature in features if feature != "State"]
    transformers = []
    if numeric_features:
        transformers.append(("numeric", "passthrough", numeric_features))
    if "State" in features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                ["State"],
            )
        )
    return ColumnTransformer(transformers=transformers)


def modeling(features: list[str] | ColumnTransformer) -> Pipeline:
    """Create the preprocessing and baseline Linear Regression pipeline."""
    preprocessor = (
        features if isinstance(features, ColumnTransformer) else build_preprocessor(features)
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def cv_rmse(
    features: list[str], X_train: pd.DataFrame, y_train: pd.Series
) -> float:
    """Return 5-fold CV RMSE using training data only."""
    scores = cross_val_score(
        modeling(features),
        X_train[features],
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=CV,
    )
    return float(-scores.mean())


def cv_metrics(
    features: list[str], X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[float, float]:
    """Return training-only 5-fold CV RMSE and R2 for one feature subset."""
    model = modeling(features)
    rmse = -cross_val_score(
        model, X_train[features], y_train,
        scoring="neg_root_mean_squared_error", cv=CV,
    ).mean()
    r2 = cross_val_score(
        model, X_train[features], y_train, scoring="r2", cv=CV
    ).mean()
    return float(rmse), float(r2)


def feature_selection_curves(
    X_train: pd.DataFrame, y_train: pd.Series
) -> pd.DataFrame:
    """Build real CV curves without using or estimating any test-set values."""
    paths: dict[str, list[list[str]]] = {"Baseline": [FEATURE_GROUPS.copy()]}

    best_subset_path = []
    for size in range(1, len(FEATURE_GROUPS) + 1):
        choices = [
            (list(subset), cv_rmse(list(subset), X_train, y_train))
            for subset in combinations(FEATURE_GROUPS, size)
        ]
        best_subset_path.append(min(choices, key=lambda item: item[1])[0])
    paths["Best Subset"] = best_subset_path

    forward_path, selected = [], []
    while len(selected) < len(FEATURE_GROUPS):
        remaining = [feature for feature in FEATURE_GROUPS if feature not in selected]
        selected = min(
            (
                (selected + [feature], cv_rmse(selected + [feature], X_train, y_train))
                for feature in remaining
            ),
            key=lambda item: item[1],
        )[0]
        forward_path.append(selected.copy())
    paths["Forward SFS"] = forward_path

    backward_path, selected = [], FEATURE_GROUPS.copy()
    while selected:
        backward_path.append(selected.copy())
        if len(selected) == 1:
            break
        selected = min(
            (
                (
                    [feature for feature in selected if feature != removed],
                    cv_rmse(
                        [feature for feature in selected if feature != removed],
                        X_train,
                        y_train,
                    ),
                )
                for removed in selected
            ),
            key=lambda item: item[1],
        )[0]
    paths["Backward SBS"] = backward_path

    rfe_path, selected = [], FEATURE_GROUPS.copy()
    while selected:
        rfe_path.append(selected.copy())
        if len(selected) == 1:
            break
        ranking_model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(selected)),
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )
        ranking_model.fit(X_train[selected], y_train)
        names = ranking_model.named_steps["preprocessor"].get_feature_names_out()
        coefficients = ranking_model.named_steps["regressor"].coef_
        importance = {}
        for feature in selected:
            prefix = "categorical__State_" if feature == "State" else f"numeric__{feature}"
            importance[feature] = float(
                np.linalg.norm(
                    [
                        coefficient
                        for name, coefficient in zip(names, coefficients)
                        if name == prefix or name.startswith(prefix)
                    ]
                )
            )
        selected.remove(min(importance, key=importance.get))
    paths["RFECV"] = rfe_path

    rows = []
    for method, feature_sets in paths.items():
        for features in feature_sets:
            rmse, r2 = cv_metrics(features, X_train, y_train)
            rows.append(
                {
                    "Method": method,
                    "Feature Count": len(features),
                    "Selected Features": ", ".join(features),
                    "CV RMSE": rmse,
                    "CV R2": r2,
                }
            )

    lasso_features, lasso_rmse, lasso_model = lasso_selection(X_train, y_train)
    lasso_r2 = cross_val_score(
        lasso_model, X_train[FEATURE_GROUPS], y_train, scoring="r2", cv=CV
    ).mean()
    rows.append(
        {
            "Method": "LassoCV",
            "Feature Count": len(lasso_features),
            "Selected Features": ", ".join(lasso_features),
            "CV RMSE": lasso_rmse,
            "CV R2": float(lasso_r2),
        }
    )

    curves = pd.DataFrame(rows).sort_values(["Method", "Feature Count"])
    curves["Is Best"] = curves.groupby("Method")["CV RMSE"].transform("min").eq(curves["CV RMSE"])
    curves.to_csv(CURVE_PATH, index=False)
    WEB_DATA_PATH.mkdir(parents=True, exist_ok=True)
    curves.to_csv(WEB_DATA_PATH / CURVE_PATH.name, index=False)
    return curves


def best_subset_selection(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[list[str], float]:
    """Evaluate every non-empty raw feature-group subset with 5-fold CV."""
    candidates = []
    for size in range(1, len(FEATURE_GROUPS) + 1):
        for subset in combinations(FEATURE_GROUPS, size):
            features = list(subset)
            candidates.append((features, cv_rmse(features, X_train, y_train)))
    return min(candidates, key=lambda item: item[1])


def sequential_selection(
    X_train: pd.DataFrame, y_train: pd.Series, direction: str
) -> tuple[list[str], float]:
    """Run grouped forward or backward sequential selection with 5-fold CV."""
    selected = [] if direction == "forward" else FEATURE_GROUPS.copy()
    candidates = []

    while True:
        if selected:
            candidates.append((selected.copy(), cv_rmse(selected, X_train, y_train)))

        if direction == "forward":
            remaining = [feature for feature in FEATURE_GROUPS if feature not in selected]
            if not remaining:
                break
            choices = [
                (selected + [feature], cv_rmse(selected + [feature], X_train, y_train))
                for feature in remaining
            ]
            selected = min(choices, key=lambda item: item[1])[0]
        else:
            if len(selected) == 1:
                break
            choices = [
                (
                    [feature for feature in selected if feature != removed],
                    cv_rmse(
                        [feature for feature in selected if feature != removed],
                        X_train,
                        y_train,
                    ),
                )
                for removed in selected
            ]
            selected = min(choices, key=lambda item: item[1])[0]

    return min(candidates, key=lambda item: item[1])


def grouped_rfecv(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[list[str], float]:
    """Recursively eliminate whole feature groups and choose size by 5-fold CV."""
    selected = FEATURE_GROUPS.copy()
    candidates = []

    while selected:
        candidates.append((selected.copy(), cv_rmse(selected, X_train, y_train)))
        if len(selected) == 1:
            break

        ranking_model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(selected)),
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )
        ranking_model.fit(X_train[selected], y_train)
        names = ranking_model.named_steps["preprocessor"].get_feature_names_out()
        coefficients = ranking_model.named_steps["regressor"].coef_
        importance = {}
        for feature in selected:
            prefix = (
                "categorical__State_" if feature == "State" else f"numeric__{feature}"
            )
            values = [
                coefficient
                for name, coefficient in zip(names, coefficients)
                if name == prefix or name.startswith(prefix)
            ]
            importance[feature] = float(np.linalg.norm(values))
        selected.remove(min(importance, key=importance.get))

    return min(candidates, key=lambda item: item[1])


def lasso_selection(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[list[str], float, Pipeline]:
    """Tune Lasso with internal 5-fold CV after one-hot encoding and scaling."""
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(FEATURE_GROUPS)),
            ("scaler", StandardScaler()),
            (
                "regressor",
                LassoCV(cv=CV, random_state=RANDOM_STATE, max_iter=100_000),
            ),
        ]
    )
    nested_scores = cross_val_score(
        model,
        X_train[FEATURE_GROUPS],
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=CV,
    )
    model.fit(X_train[FEATURE_GROUPS], y_train)

    names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["regressor"].coef_
    selected = []
    for feature in FEATURE_GROUPS:
        prefix = "categorical__State_" if feature == "State" else f"numeric__{feature}"
        group_coefficients = [
            coefficient
            for name, coefficient in zip(names, coefficients)
            if name == prefix or name.startswith(prefix)
        ]
        if any(not np.isclose(coefficient, 0.0) for coefficient in group_coefficients):
            selected.append(feature)

    return selected, float(-nested_scores.mean()), model


def evaluate_method(
    method: str,
    features: list[str],
    cv_score: float,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model: Pipeline | None = None,
) -> dict:
    """Fit on the training set and evaluate once on the untouched test set."""
    fitted_model = model or modeling(features)
    if model is None:
        fitted_model.fit(X_train[features], y_train)
        predictions = fitted_model.predict(X_test[features])
    else:
        predictions = fitted_model.predict(X_test[FEATURE_GROUPS])

    return {
        "Method": method,
        "Selected Features": ", ".join(features),
        "Feature Count": len(features),
        "CV RMSE": cv_score,
        "Test RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "Test R2": r2_score(y_test, predictions),
    }


def calculate_group_importance(
    method: str,
    features: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: Pipeline | None = None,
) -> list[dict]:
    """Calculate normalized absolute coefficient importance by feature group."""
    importance_model = model
    if importance_model is None:
        importance_model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(features)),
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )
        importance_model.fit(X_train[features], y_train)

    names = importance_model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = importance_model.named_steps["regressor"].coef_
    raw_importance = {}
    for feature in features:
        prefix = "categorical__State_" if feature == "State" else f"numeric__{feature}"
        values = [
            coefficient
            for name, coefficient in zip(names, coefficients)
            if name == prefix or name.startswith(prefix)
        ]
        raw_importance[feature] = float(np.linalg.norm(values))

    total = sum(raw_importance.values()) or 1.0
    return [
        {
            "Method": method,
            "Feature": feature,
            "Importance": importance / total,
        }
        for feature, importance in raw_importance.items()
    ]


def clean_feature_name(name: str) -> str:
    """Convert transformed feature names to dashboard-friendly labels."""
    return name.replace("numeric__", "").replace("categorical__", "")


def fitted_model_details(
    method: str,
    features: list[str],
    cv_score: float,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model: Pipeline | None,
) -> dict:
    """Export real fitted coefficients, predictions, residuals and diagnostics."""
    fitted = model or modeling(features)
    model_features = FEATURE_GROUPS if model is not None else features
    if model is None:
        fitted.fit(X_train[features], y_train)

    predictions = fitted.predict(X_test[model_features])
    residuals = y_test.to_numpy() - predictions
    preprocessor = fitted.named_steps["preprocessor"]
    transformed_names = [
        clean_feature_name(name) for name in preprocessor.get_feature_names_out()
    ]
    transformed_train = preprocessor.transform(X_train[model_features])
    regressor = fitted.named_steps["regressor"]
    coefficients = np.asarray(regressor.coef_, dtype=float)
    intercept = float(regressor.intercept_)

    if "scaler" in fitted.named_steps:
        scaler = fitted.named_steps["scaler"]
        coefficients = coefficients / scaler.scale_
        intercept -= float(np.dot(regressor.coef_, scaler.mean_ / scaler.scale_))

    coefficient_rows = [
        {"feature": name, "coefficient": float(coefficient)}
        for name, coefficient in zip(transformed_names, coefficients)
        if not np.isclose(coefficient, 0.0)
    ]

    prediction_residual_correlation = float(
        np.corrcoef(predictions, residuals)[0, 1]
    )
    scale_correlation = float(
        np.corrcoef(predictions, np.abs(residuals))[0, 1]
    )
    durbin_watson = float(
        np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2)
    )
    jb_result = jarque_bera(residuals)

    vif_values = []
    if transformed_train.shape[1] == 1:
        vif_values = [1.0]
    else:
        for column in range(transformed_train.shape[1]):
            other_columns = np.delete(transformed_train, column, axis=1)
            vif_model = LinearRegression().fit(
                other_columns, transformed_train[:, column]
            )
            vif_values.append(1.0 / max(1e-12, 1.0 - vif_model.score(
                other_columns, transformed_train[:, column]
            )))
    max_vif = float(max(vif_values))

    assumptions = [
        {
            "name": "線性關係",
            "value": abs(prediction_residual_correlation),
            "display": f"|corr(ŷ, residual)| = {abs(prediction_residual_correlation):.3f}",
            "passed": bool(abs(prediction_residual_correlation) < 0.2),
        },
        {
            "name": "誤差獨立性",
            "value": durbin_watson,
            "display": f"Durbin-Watson = {durbin_watson:.3f}",
            "passed": bool(1.5 <= durbin_watson <= 2.5),
        },
        {
            "name": "同質變異性",
            "value": abs(scale_correlation),
            "display": f"|corr(ŷ, |residual|)| = {abs(scale_correlation):.3f}",
            "passed": bool(abs(scale_correlation) < 0.3),
        },
        {
            "name": "殘差常態性",
            "value": float(jb_result.pvalue),
            "display": f"Jarque-Bera p = {jb_result.pvalue:.3f}",
            "passed": bool(jb_result.pvalue > 0.05),
        },
        {
            "name": "無嚴重共線性",
            "value": max_vif,
            "display": f"Maximum VIF = {max_vif:.3f}",
            "passed": bool(max_vif < 5),
        },
    ]

    return {
        "method": method,
        "selectedFeatures": features,
        "cvRmse": cv_score,
        "testRmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "testR2": float(r2_score(y_test, predictions)),
        "intercept": intercept,
        "coefficients": coefficient_rows,
        "testRows": [
            {
                "index": int(index),
                "actual": float(actual),
                "predicted": float(predicted),
                "residual": float(residual),
            }
            for index, actual, predicted, residual in zip(
                y_test.index, y_test, predictions, residuals
            )
        ],
        "assumptions": assumptions,
    }


def feature_selection_experiments(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Run and compare the baseline and five requested selection methods."""
    experiments = []

    baseline_cv = cv_rmse(FEATURE_GROUPS, X_train, y_train)
    experiments.append(("Baseline", FEATURE_GROUPS, baseline_cv, None))

    best_features, best_cv = best_subset_selection(X_train, y_train)
    experiments.append(("Best Subset", best_features, best_cv, None))

    rfe_features, rfe_cv = grouped_rfecv(X_train, y_train)
    experiments.append(("RFECV", rfe_features, rfe_cv, None))

    lasso_features, lasso_cv, lasso_model = lasso_selection(X_train, y_train)
    experiments.append(("LassoCV", lasso_features, lasso_cv, lasso_model))

    sfs_features, sfs_cv = sequential_selection(X_train, y_train, "forward")
    experiments.append(("Forward SFS", sfs_features, sfs_cv, None))

    sbs_features, sbs_cv = sequential_selection(X_train, y_train, "backward")
    experiments.append(("Backward SBS", sbs_features, sbs_cv, None))

    rows = [
        evaluate_method(
            method,
            features,
            cv_score,
            X_train,
            X_test,
            y_train,
            y_test,
            model,
        )
        for method, features, cv_score, model in experiments
    ]
    comparison = pd.DataFrame(rows).sort_values("CV RMSE").reset_index(drop=True)
    importance = pd.DataFrame(
        row
        for method, features, _, model in experiments
        for row in calculate_group_importance(
            method, features, X_train, y_train, model
        )
    )

    comparison.to_csv(COMPARISON_PATH, index=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)
    WEB_DATA_PATH.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(WEB_DATA_PATH / COMPARISON_PATH.name, index=False)
    importance.to_csv(WEB_DATA_PATH / IMPORTANCE_PATH.name, index=False)
    dashboard_details = {
        "metadata": {
            "dataset": "50_Startups.csv",
            "randomState": RANDOM_STATE,
            "testSize": 0.2,
            "cvFolds": 5,
            "trainIndices": [int(index) for index in X_train.index],
            "testIndices": [int(index) for index in X_test.index],
        },
        "models": [
            fitted_model_details(
                method,
                features,
                cv_score,
                X_train,
                X_test,
                y_train,
                y_test,
                model,
            )
            for method, features, cv_score, model in experiments
        ],
    }
    MODEL_DETAILS_PATH.write_text(
        json.dumps(dashboard_details, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    feature_selection_curves(X_train, y_train)
    create_comparison_chart(comparison)
    return comparison


def create_comparison_chart(comparison: pd.DataFrame) -> None:
    """Create comparison charts and highlight the best CV RMSE method."""
    best_method = comparison.loc[comparison["CV RMSE"].idxmin(), "Method"]
    colors = ["#1f6f54" if method == best_method else "#9fb8ad" for method in comparison["Method"]]
    positions = np.arange(len(comparison))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].bar(positions - width / 2, comparison["CV RMSE"], width, label="CV RMSE", color=colors)
    axes[0].bar(positions + width / 2, comparison["Test RMSE"], width, label="Test RMSE", color="#d79a3b")
    axes[0].set_title("RMSE Comparison")
    axes[0].set_ylabel("RMSE")
    axes[0].set_xticks(positions, comparison["Method"], rotation=25, ha="right")
    axes[0].legend()

    axes[1].bar(comparison["Method"], comparison["Test R2"], color=colors)
    axes[1].set_title("Test R-squared")
    axes[1].set_ylabel("R-squared")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylim(min(0, comparison["Test R2"].min() - 0.05), 1)
    axes[1].text(0.02, 0.96, f"Best CV RMSE: {best_method}", transform=axes[1].transAxes, va="top", fontweight="bold", color="#1f6f54")

    fig.suptitle("50 Startups Feature Selection Comparison", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)


def evaluation(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Evaluate the original baseline and display its coefficients."""
    predictions = model.predict(X_test)
    results = pd.DataFrame({"Actual Profit": y_test, "Predicted Profit": predictions})
    print("\nBaseline prediction results:\n", results.round(2))
    print(f"\nBaseline MAE:  {mean_absolute_error(y_test, predictions):,.2f}")
    print(f"Baseline RMSE: {mean_squared_error(y_test, predictions) ** 0.5:,.2f}")
    print(f"Baseline R2:   {r2_score(y_test, predictions):.4f}")


def deployment(model: Pipeline) -> None:
    """Demonstrate how the trained baseline predicts new startup data."""
    example = pd.DataFrame(
        {
            "R&D Spend": [120000],
            "Administration": [110000],
            "Marketing Spend": [250000],
            "State": ["California"],
        }
    )
    predicted_profit = model.predict(example)[0]
    print(f"\nExample predicted profit: {predicted_profit:,.2f}")


def main() -> None:
    business_understanding()
    data = data_understanding(DATA_PATH)
    X_train, X_test, y_train, y_test = data_preparation(data)

    baseline = modeling(FEATURE_GROUPS)
    baseline.fit(X_train, y_train)
    evaluation(baseline, X_test, y_test)

    comparison = feature_selection_experiments(
        X_train, X_test, y_train, y_test
    )
    print("\nModel comparison (sorted by CV RMSE):\n")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print(f"\nBest method: {comparison.iloc[0]['Method']}")
    print(f"Best feature group: {comparison.iloc[0]['Selected Features']}")
    print(
        f"\nSaved: {COMPARISON_PATH}, {IMPORTANCE_PATH}, and {CHART_PATH}"
    )

    deployment(baseline)


if __name__ == "__main__":
    main()
