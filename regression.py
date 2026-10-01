import pandas as pd
import matplotlib
from collections import Counter
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import numpy as np
from preprocess import process_time, process_weather

def create_models():
    """
    Create all required regression models.
    A fresh set of models is created for each fold.
    """

    models = {
        "Average Regressor": DummyRegressor(strategy="mean"),

        "Linear Regression": LinearRegression(),

        # "Regression Tree, max_depth=None": DecisionTreeRegressor(),

        "Regression Tree, max_depth=5": DecisionTreeRegressor(
            max_depth=5
        ),

        # "Random Forest, max_depth=None": RandomForestRegressor(
        #     n_jobs=-1
        # ),

        "Random Forest, max_depth=5": RandomForestRegressor(
            max_depth=5,
            n_jobs=-1
        )
    }

    return models


def cross_validate_regressors(X, y):
    """
    Perform 10-fold cross-validation for all regression models.

    Returns
    -------
    results : dict
        Contains the 10 scores, mean, and standard deviation
        for MSE, RMSE, MAE, and R2.
    """

    kf = KFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    results = {}

    for model_name in create_models().keys():

        print(f"\nRunning {model_name}...")

        mse_scores = []
        rmse_scores = []
        mae_scores = []
        r2_scores = []

        for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):

            # Split data for this fold
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]

            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]

            # Create a fresh model for this fold
            model = create_models()[model_name]

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mse_scores.append(mse)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)

            print(
                f"  Fold {fold}: "
                f"MSE={mse:.3f}, "
                f"RMSE={rmse:.3f}, "
                f"MAE={mae:.3f}, "
                f"R2={r2:.3f}"
            )

        # Store all 10 folds + mean + std
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
            "R2_std": np.std(r2_scores)
        }

    return results


df = pd.read_csv("./US_Accidents_filtered.csv")
df["Severity"] *= 10

process_weather(df=df)
df = df.drop(columns=["Weather_Condition"])

process_time(df=df)

df = df.drop(columns=[
    "Start_Time",
    "End_Time",
    "Street",
    "City",
    "County",
    "State",
    "Timezone",
    "Wind_Direction"
])


for day in [
    "Sunrise_Sunset",
    "Civil_Twilight",
    "Nautical_Twilight",
    "Astronomical_Twilight"
]:
    df[day] = (df[day] == "Day").astype(int)


numeric_columns = df.select_dtypes(include=np.number).columns

df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].mean()
)


categorical_columns = df.select_dtypes(exclude=np.number).columns

for column in categorical_columns:
    df[column] = df[column].fillna(
        df[column].mode()[0]
    )


df = df.select_dtypes(include=np.number)


print("\nColumns being used:")
print(df.columns.tolist())

print("\nTotal missing values:")
print(df.isna().sum().sum())


y = df["Severity"]
X = df.drop(columns=["Severity"])



results = cross_validate_regressors(X, y)


for model_name, metrics in results.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print("\nMSE:")
    print(
        [round(x, 3) for x in metrics["MSE"]]
    )
    print(f"Mean: {metrics['MSE_mean']:.3f}")
    print(f"Std:  {metrics['MSE_std']:.3f}")

    print("\nRMSE:")
    print(
        [round(x, 3) for x in metrics["RMSE"]]
    )
    print(f"Mean: {metrics['RMSE_mean']:.3f}")
    print(f"Std:  {metrics['RMSE_std']:.3f}")

    print("\nMAE:")
    print(
        [round(x, 3) for x in metrics["MAE"]]
    )
    print(f"Mean: {metrics['MAE_mean']:.3f}")
    print(f"Std:  {metrics['MAE_std']:.3f}")

    print("\nR2:")
    print(
        [round(x, 3) for x in metrics["R2"]]
    )
    print(f"Mean: {metrics['R2_mean']:.3f}")
    print(f"Std:  {metrics['R2_std']:.3f}")


