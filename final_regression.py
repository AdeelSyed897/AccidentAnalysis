import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from preprocess import process_weather, process_time
import matplotlib.pyplot as plt



print("Loading dataset...")

df = pd.read_csv("./US_Accidents_filtered.csv")

df["Severity"] *= 10

print(f"Number of rows: {len(df):,}")


process_weather(df)

df = df.drop(
    columns=["Weather_Condition"]
)
process_time(df)

df = df.drop(
    columns=[
        "Start_Time",
        "End_Time",
        "Street",
        "City",
        "County",
        "State",
        "Timezone",
        "Wind_Direction"
    ],
    errors="ignore"
)

for column in [
    "Sunrise_Sunset",
    "Civil_Twilight",
    "Nautical_Twilight",
    "Astronomical_Twilight"
]:

    if column in df.columns:
        df[column] = (
            df[column] == "Day"
        ).astype(int)

# Numeric columns -> mean
numeric_columns = df.select_dtypes(
    include=np.number
).columns

df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].mean()
)


# Categorical columns -> most common value
categorical_columns = df.select_dtypes(
    exclude=np.number
).columns

for column in categorical_columns:

    if not df[column].mode().empty:

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )

df = df.select_dtypes(
    include=np.number
)



X = df.drop(
    columns=["Severity"]
)

y = df["Severity"]


print("\nNumber of features:", X.shape[1])



print("\nTraining final Decision Tree...")
print("max_depth = 5")

tree = DecisionTreeRegressor(
    max_depth=5
)

tree.fit(X, y)

y_pred = tree.predict(X)


mse = mean_squared_error(
    y,
    y_pred
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    y,
    y_pred
)

r2 = r2_score(
    y,
    y_pred
)


print("\n")
print("=" * 60)
print("FINAL REGRESSION TREE RESULTS")
print("=" * 60)

print(f"MSE:  {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE:  {mae:.3f}")
print(f"R²:   {r2:.3f}")


tree_depth = tree.get_depth()
node_count = tree.tree_.node_count
leaf_count = tree.get_n_leaves()


print("\n")
print("=" * 60)
print("TREE SIZE")
print("=" * 60)

print(f"Maximum depth requested: 5")
print(f"Actual tree depth:       {tree_depth}")
print(f"Total nodes:             {node_count}")
print(f"Leaf nodes:              {leaf_count}")



importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": tree.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance.to_string(
        index=False
    )
)


print("\n")
print("=" * 60)
print("TOP 3 MOST IMPORTANT FEATURES")
print("=" * 60)

top_3 = importance.head(3)

for i, row in enumerate(
    top_3.itertuples(index=False),
    start=1
):

    print(
        f"{i}. {row.Feature}: "
        f"{row.Importance:.4f}"
    )



print("\nCreating tree visualization...")


plt.figure(
    figsize=(24, 14)
)

plot_tree(
    tree,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title(
    "Decision Tree Regressor (max_depth=5)"
)

plt.tight_layout()

plt.savefig(
    "decision_tree_max_depth_5.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Tree visualization saved to:")
print("decision_tree_max_depth_5.png")



print("\n")
print("=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"""
Model:
DecisionTreeRegressor(max_depth=5)

Dataset size:
{len(df):,} observations

Number of features:
{X.shape[1]}

MSE:
{mse:.3f}

RMSE:
{rmse:.3f}

MAE:
{mae:.3f}

R²:
{r2:.3f}

Tree depth:
{tree_depth}

Total nodes:
{node_count}

Leaf nodes:
{leaf_count}

Top 3 features:
1. {importance.iloc[0]["Feature"]}
2. {importance.iloc[1]["Feature"]}
3. {importance.iloc[2]["Feature"]}
""")

print("=" * 60)
print("DONE")
print("=" * 60)