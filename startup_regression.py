"""50 Startups multiple linear regression using the CRISP-DM framework."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("50_Startups.csv")
TARGET = "Profit"
RANDOM_STATE = 42


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
    """Split predictors/target and prepare numeric and categorical features."""
    X = data.drop(columns=TARGET)
    y = data[TARGET]

    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_features),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    return train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    ), preprocessor


def modeling(preprocessor: ColumnTransformer) -> Pipeline:
    """Create the preprocessing and multiple linear regression pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def evaluation(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Evaluate predictions and display the fitted regression coefficients."""
    predictions = model.predict(X_test)
    results = pd.DataFrame({"Actual Profit": y_test, "Predicted Profit": predictions})

    print("\nPrediction results:\n", results.round(2))
    print(f"\nMAE:  {mean_absolute_error(y_test, predictions):,.2f}")
    print(f"RMSE: {mean_squared_error(y_test, predictions) ** 0.5:,.2f}")
    print(f"R2:   {r2_score(y_test, predictions):.4f}")

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["regressor"].coef_
    coefficient_table = pd.DataFrame(
        {"Feature": feature_names, "Coefficient": coefficients}
    ).sort_values("Coefficient", key=abs, ascending=False)
    print("\nModel coefficients:\n", coefficient_table.to_string(index=False))
    print(f"Intercept: {model.named_steps['regressor'].intercept_:,.2f}")


def deployment(model: Pipeline) -> None:
    """Demonstrate how the trained pipeline can predict new startup data."""
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
    (X_train, X_test, y_train, y_test), preprocessor = data_preparation(data)

    model = modeling(preprocessor)
    model.fit(X_train, y_train)

    evaluation(model, X_test, y_test)
    deployment(model)


if __name__ == "__main__":
    main()
