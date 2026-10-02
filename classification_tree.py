"""
PART II.1 - Final Classification Model and Analysis

Requirements:
- Decision Tree with max_depth=5
- 10-fold Stratified Cross-Validation
- Estimated classification accuracy
- Decision tree depiction
- Tree size and readability
- Three salient accident patterns

Input:
    US_Accidents_filtered.csv

Output:
    decision_tree.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_FILE = "US_Accidents_filtered.csv"

df = pd.read_csv(DATA_FILE)
df = df.sample(10000)

# Target
y = df["Severity"]

# Features
X = df.drop(columns=["Severity"])


# ============================================================
# 2. SIMPLE PREPROCESSING
# ============================================================

from preprocess import process_time, process_weather

# Separate numerical and categorical columns
numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns
process_weather(df=df)
process_time(df=df)
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


# Numerical columns:
# replace missing values with median
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# Categorical columns:
# replace missing values with most common value
# then convert categories to numbers
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])


preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])


# ============================================================
# 3. DECISION TREE
# ============================================================

tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)


model = Pipeline([
    ("preprocessing", preprocessor),
    ("tree", tree)
])


# ============================================================
# 4. 10-FOLD CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)


print("=" * 60)
print("10-FOLD CROSS-VALIDATION")
print("=" * 60)

print("Fold accuracies:")

for i, score in enumerate(scores, 1):
    print(
        f"Fold {i}: {score:.4f}"
    )

mean_accuracy = scores.mean()
std_accuracy = scores.std()

print(
    f"\nEstimated accuracy: "
    f"{mean_accuracy:.4f}"
)

print(
    f"Estimated accuracy (%): "
    f"{mean_accuracy * 100:.2f}%"
)

print(
    f"Standard deviation: "
    f"{std_accuracy:.4f}"
)


# ============================================================
# 5. TRAIN FINAL TREE ON ENTIRE DATASET
# ============================================================

model.fit(X, y)


# Get the actual fitted decision tree
fitted_tree = model.named_steps["tree"]

# Get feature names after one-hot encoding
fitted_preprocessor = (
    model.named_steps["preprocessing"]
)

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)


# ============================================================
# 6. TREE SIZE
# ============================================================

print("\n" + "=" * 60)
print("TREE SIZE")
print("=" * 60)

print(
    "Maximum depth:",
    fitted_tree.tree_.max_depth
)

print(
    "Number of nodes:",
    fitted_tree.tree_.node_count
)

print(
    "Number of leaves:",
    fitted_tree.tree_.n_leaves
)


# ============================================================
# 7. DEPICT THE TREE
# ============================================================

plt.figure(figsize=(25, 15))

plot_tree(
    fitted_tree,
    feature_names=feature_names,
    class_names=[
        str(x)
        for x in fitted_tree.classes_
    ],
    filled=True,
    rounded=True,
    fontsize=7
)

plt.title(
    "Decision Tree for Accident Severity (max_depth=5)"
)

plt.tight_layout()

plt.savefig(
    "decision_tree.png",
    dpi=200
)

plt.show()
plt.savefig("classification_tree")


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": fitted_tree.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("MOST IMPORTANT FEATURES")
print("=" * 60)

print(
    importance.head(10).to_string(
        index=False
    )
)


# ============================================================
# 9. THREE MOST SALIENT PATTERNS
# ============================================================

print("\n" + "=" * 60)
print("THREE MOST SALIENT TREE PATTERNS")
print("=" * 60)

# Find the features used by the tree
used_features = []

for feature_index in fitted_tree.tree_.feature:

    if feature_index >= 0:

        used_features.append(
            feature_names[feature_index]
        )

# Count how often each feature is used
feature_counts = pd.Series(
    used_features
).value_counts()

print(
    "\nThe three features used most often "
    "in the tree are:\n"
)

for feature, count in feature_counts.head(3).items():

    print(
        f"- {feature} "
        f"(used {count} times)"
    )


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(
    f"Estimated classification accuracy: "
    f"{mean_accuracy * 100:.2f}%"
)

print(
    f"Accuracy standard deviation: "
    f"{std_accuracy * 100:.2f}%"
)

print(
    f"Tree depth: "
    f"{fitted_tree.tree_.max_depth}"
)

print(
    f"Tree nodes: "
    f"{fitted_tree.tree_.node_count}"
)

print(
    f"Tree leaves: "
    f"{fitted_tree.tree_.n_leaves}"
)

print(
    "\nDecision tree saved as: decision_tree.png"
)
