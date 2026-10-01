from collections import Counter
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor


def process_weather(df):
    weather = df["Weather_Condition"].fillna("Unknown").str.lower()

    df["weather_rain"] = weather.str.contains(
        "rain|drizzle|shower", regex=True
    ).astype(int)
    df["weather_snow"] = weather.str.contains("snow", regex=True).astype(int)
    df["weather_fog"] = weather.str.contains("fog|mist", regex=True).astype(int)
    df["weather_thunderstorm"] = weather.str.contains(
        "thunder|t-storm", regex=True
    ).astype(int)
    df["weather_wind"] = weather.str.contains("windy", regex=True).astype(int)
    df["weather_hail"] = weather.str.contains("hail", regex=True).astype(int)
    df["weather_freezing"] = weather.str.contains(
        "freezing|ice pellets|sleet|wintry mix", regex=True
    ).astype(int)
    df["weather_haze"] = weather.str.contains(
        "haze|smoke|dust", regex=True
    ).astype(int)
    df["weather_clear"] = weather.str.contains(
        "fair|clear", regex=True
    ).astype(int)

    return df


def process_time(df):
    start_dt = pd.to_datetime(df["Start_Time"], format="mixed")

    df["start_hour"] = start_dt.dt.hour
    df["start_dayofweek"] = start_dt.dt.dayofweek
    df["start_month"] = start_dt.dt.month
    df["start_year"] = start_dt.dt.year

    # REMOVED: duration_minutes (End_Time - Start_Time) causes target leakage
    return df


def prepare_dataset(filepath):
    df = pd.read_csv(filepath)

    df["Severity"] = df["Severity"] * 10

    df = process_weather(df)
    df = process_time(df)

    # Drop non-predictive, high-cardinality, or leaky columns
    drop_cols = [
        "Start_Time",
        "End_Time",  # Leaky feature source
        "Weather_Condition",
        "Street",
        "City",
        "County",
        "State",
        "Timezone",
        "Wind_Direction",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Convert twilight boolean columns cleanly
    twilight_cols = [
        "Sunrise_Sunset",
        "Civil_Twilight",
        "Nautical_Twilight",
        "Astronomical_Twilight",
    ]
    for col in twilight_cols:
        if col in df.columns:
            df[col] = (df[col] == "Day").astype(int)

    return df


def get_models():
    """Returns a dict of model identifiers to estimator instances."""
    return {
        "Average Regressor": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Regression Tree, max_depth=5": DecisionTreeRegressor(
            max_depth=5, random_state=42
        ),
        "Random Forest, max_depth=5": RandomForestRegressor(
            max_depth=5, n_jobs=-1, random_state=42
        ),
    }


def cross_validate_regressors(X, y):
    """Perform 10-fold cross-validation cleanly without feature leakage across folds."""
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    models = get_models()
    results = {}

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()

    # Define per-fold imputer pipeline to prevent global mean leakage
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="mean"), numeric_features)
        ],
        remainder="drop",
    )

    for model_name, base_estimator in models.items():
        print(f"\nRunning {model_name}...")

        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("regressor", base_estimator)]
        )

        mse_scores, rmse_scores, mae_scores, r2_scores = [], [], [], []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Fit preprocessor + model ONLY on training fold
            pipeline.fit(X_train, y_train)

            # Predict on unseen test fold
            y_pred = pipeline.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mse_scores.append(mse)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)

            print(
                f"  Fold {fold}: MSE={mse:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}"
            )

        results[model_name] = {
            "MSE": mse_scores,
            "RMSE": rmse_scores,
            "MAE": mae_scores,
            "R2": r2_scores,
            "MSE_mean": np.mean(mse_scores),
            "MSE_std": np.std(mse_scores),
            "RMSE_mean": np.mean(rmse_scores),
            "RMSE_std": np.std(rmse_scores),
            "MAE_mean": np.mean(mae_scores),
            "MAE_std": np.std(mae_scores),
            "R2_mean": np.mean(r2_scores),
            "R2_std": np.std(r2_scores),
        }

    return results


if __name__ == "__main__":
    df = prepare_dataset("US_Accidents_filtered.csv")

    y = df["Severity"]
    X = df.drop(columns=["Severity"])

    print("\nFeatures used:")
    print(X.columns.tolist())

    results = cross_validate_regressors(X, y)

    for model_name, metrics in results.items():
        print("\n" + "=" * 60)
        print(model_name)
        print("=" * 60)

        for metric in ["MSE", "RMSE", "MAE", "R2"]:
            print(f"\n{metric}:")
            print([round(x, 3) for x in metrics[metric]])
            print(f"Mean: {metrics[f'{metric}_mean']:.3f}")
            print(f"Std:  {metrics[f'{metric}_std']:.3f}")