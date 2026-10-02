"""US Accidents analysis: preprocessing, regression, classification, clustering and anomaly detection.

Usage: python final_code.py [path/to/US_Accidents_March23.csv]
The raw CSV is only needed if US_Accidents_filtered.csv does not exist yet.
"""
import os
import sys
import time
import warnings
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             homogeneity_completeness_v_measure, mean_absolute_error,
                             mean_squared_error, normalized_mutual_info_score, r2_score,
                             silhouette_score)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor, plot_tree)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220, "display.max_columns", 60)

RAW_CSV = "US_Accidents_March23.csv"
FILTERED_CSV = "US_Accidents_filtered.csv"
FIG_DIR = "figures"
SEED = 0
T0 = time.time()


# ============================================================
# PREPROCESSING
# ============================================================
CHUNK_ROWS = 500_000
MAX_MISSING = 1
DROP_COLS = [
    "ID", "Source", "Description", "Country", "Airport_Code", "Weather_Timestamp",
    "Zipcode", "End_Lat", "End_Lng", "Turning_Loop",
]
NON_FEATURE_COLS = [
    "Start_Time", "End_Time", "Weather_Condition", "Street", "City", "County",
    "State", "Timezone", "Wind_Direction",
]
TWILIGHT_COLS = ["Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight"]
WEATHER_PATTERNS = {
    "weather_rain": "rain|drizzle|shower",
    "weather_snow": "snow",
    "weather_fog": "fog|mist",
    "weather_thunderstorm": "thunder|t-storm",
    "weather_wind": "windy",
    "weather_hail": "hail",
    "weather_freezing": "freezing|ice pellets|sleet|wintry mix",
    "weather_haze": "haze|smoke|dust",
    "weather_clear": "fair|clear",
}


def process_weather(df):
    weather = df["Weather_Condition"].fillna("Unknown").str.lower()
    for column, pattern in WEATHER_PATTERNS.items():
        df[column] = weather.str.contains(pattern, regex=True).astype(int)
    return df


def process_time(df):
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], format="mixed")
    df["End_Time"] = pd.to_datetime(df["End_Time"], format="mixed")
    df["start_hour"] = df["Start_Time"].dt.hour
    df["start_dayofweek"] = df["Start_Time"].dt.dayofweek
    df["start_month"] = df["Start_Time"].dt.month
    df["start_year"] = df["Start_Time"].dt.year
    df["duration_minutes"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60

    for col in TWILIGHT_COLS:
        if col in df.columns:
            df[col] = (df[col] == "Day").astype(int)
    return df


def engineer_features(df):
    df = process_weather(df)
    df = process_time(df)
    return df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])


def filter_raw_data(raw_csv):
    total_in = 0
    total_out = 0
    missing_hist = {}
    missing_kept = None
    country_values = set()
    first = True

    for chunk in pd.read_csv(raw_csv, chunksize=CHUNK_ROWS):
        n_missing = chunk.isna().sum(axis=1)
        for k, v in n_missing.value_counts().items():
            missing_hist[k] = missing_hist.get(k, 0) + v
        country_values.update(chunk["Country"].dropna().unique())

        kept = chunk[n_missing <= MAX_MISSING].drop(columns=DROP_COLS)
        col_missing = kept.isna().sum()
        missing_kept = col_missing if missing_kept is None else missing_kept + col_missing

        kept.to_csv(FILTERED_CSV, mode="w" if first else "a", header=first, index=False)
        first = False

        total_in += len(chunk)
        total_out += len(kept)
        print(f"read {total_in:>9,} rows, kept {total_out:>9,}", flush=True)

    print("\nRows per number of missing values (raw data, first 3):")
    for k in sorted(missing_hist)[:3]:
        print(f"  {k} missing: {missing_hist[k]:,}")
    print(f"\nCountry values in raw data: {country_values}")
    print(f"Rows kept: {total_out:,} of {total_in:,}")
    print("\nMissing values per remaining column (kept rows):")
    print(missing_kept[missing_kept > 0].to_string())


# ============================================================
# REGRESSION
# ============================================================
def load_regression_data(path):
    print("Loading dataset...")
    df = pd.read_csv(path)
    df["Severity"] *= 10
    print(f"Number of rows: {len(df):,}")

    df = engineer_features(df)
    y = df["Severity"]
    X = df.drop(columns=["Severity"]).select_dtypes(include=np.number)
    print("\nFeatures used:")
    print(X.columns.tolist())
    return X, y


def get_regressors():
    return {
        "Average Regressor": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Regression Tree, max_depth=5": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest, max_depth=5": RandomForestRegressor(max_depth=5, n_jobs=-1, random_state=42),
    }


def cross_validate_regressors(X, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    results = {}

    for name, estimator in get_regressors().items():
        print(f"\nRunning {name}...")
        pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("regressor", estimator)])
        scores = {"MSE": [], "RMSE": [], "MAE": [], "R2": []}

        for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred = pipeline.predict(X.iloc[test_idx])
            y_test = y.iloc[test_idx]

            mse = mean_squared_error(y_test, y_pred)
            fold_scores = {
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "MAE": mean_absolute_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred),
            }
            for metric, value in fold_scores.items():
                scores[metric].append(value)
            print(f"  Fold {fold}: " + ", ".join(f"{m}={v:.3f}" for m, v in fold_scores.items()))

        results[name] = scores
    return results


def print_cv_results(results):
    for name, scores in results.items():
        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)
        for metric, values in scores.items():
            print(f"\n{metric}:")
            print([round(v, 3) for v in values])
            print(f"Mean: {np.mean(values):.3f}")
            print(f"Std:  {np.std(values):.3f}")


def run_final_regression(X, y):
    X = X.fillna(X.mean())

    print("\nTraining final Decision Tree (max_depth = 5)...")
    tree = DecisionTreeRegressor(max_depth=5, random_state=42)
    tree.fit(X, y)
    y_pred = tree.predict(X)

    mse = mean_squared_error(y, y_pred)
    print("\n" + "=" * 60)
    print("FINAL REGRESSION TREE RESULTS")
    print("=" * 60)
    print(f"MSE:  {mse:.3f}")
    print(f"RMSE: {np.sqrt(mse):.3f}")
    print(f"MAE:  {mean_absolute_error(y, y_pred):.3f}")
    print(f"R²:   {r2_score(y, y_pred):.3f}")

    print("\n" + "=" * 60)
    print("TREE SIZE")
    print("=" * 60)
    print("Maximum depth requested: 5")
    print(f"Actual tree depth:       {tree.get_depth()}")
    print(f"Total nodes:             {tree.tree_.node_count}")
    print(f"Leaf nodes:              {tree.get_n_leaves()}")

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": tree.feature_importances_,
    }).sort_values(by="Importance", ascending=False)

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)
    print(importance.to_string(index=False))

    print("\n" + "=" * 60)
    print("TOP 3 MOST IMPORTANT FEATURES")
    print("=" * 60)
    for i, row in enumerate(importance.head(3).itertuples(index=False), start=1):
        print(f"{i}. {row.Feature}: {row.Importance:.4f}")

    plt.figure(figsize=(24, 14))
    plot_tree(tree, feature_names=X.columns, filled=True, rounded=True, fontsize=8)
    plt.title("Decision Tree Regressor (max_depth=5)")
    plt.tight_layout()
    plt.savefig("decision_tree_max_depth_5.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("\nTree visualization saved to: decision_tree_max_depth_5.png")


# ============================================================
# CLASSIFICATION
# ============================================================
def run_classification(path, sample_size=10_000):
    df = pd.read_csv(path).sample(sample_size, random_state=42)
    df = engineer_features(df)

    y = df["Severity"]
    X = df.drop(columns=["Severity"])
    X = X.astype({c: int for c in X.select_dtypes(include=bool).columns})

    numeric_columns = X.select_dtypes(include=np.number).columns
    categorical_columns = X.select_dtypes(exclude=np.number).columns

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("tree", DecisionTreeClassifier(max_depth=5, random_state=42)),
    ])

    # 10-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    mean_accuracy, std_accuracy = scores.mean(), scores.std()

    print("=" * 60)
    print("10-FOLD CROSS-VALIDATION")
    print("=" * 60)
    print("Fold accuracies:")
    for i, score in enumerate(scores, 1):
        print(f"Fold {i}: {score:.4f}")
    print(f"\nEstimated accuracy: {mean_accuracy:.4f}")
    print(f"Estimated accuracy (%): {mean_accuracy * 100:.2f}%")
    print(f"Standard deviation: {std_accuracy:.4f}")

    # Final tree on the whole sample
    model.fit(X, y)
    fitted_tree = model.named_steps["tree"]
    feature_names = model.named_steps["preprocessing"].get_feature_names_out()

    print("\n" + "=" * 60)
    print("TREE SIZE")
    print("=" * 60)
    print("Maximum depth:", fitted_tree.tree_.max_depth)
    print("Number of nodes:", fitted_tree.tree_.node_count)
    print("Number of leaves:", fitted_tree.tree_.n_leaves)

    plt.figure(figsize=(25, 15))
    plot_tree(fitted_tree, feature_names=feature_names,
              class_names=[str(c) for c in fitted_tree.classes_],
              filled=True, rounded=True, fontsize=7)
    plt.title("Decision Tree for Accident Severity (max_depth=5)")
    plt.tight_layout()
    plt.savefig("decision_tree.png", dpi=200)
    plt.close()

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": fitted_tree.feature_importances_,
    }).sort_values("Importance", ascending=False)

    print("\n" + "=" * 60)
    print("MOST IMPORTANT FEATURES")
    print("=" * 60)
    print(importance.head(10).to_string(index=False))

    used_features = [feature_names[i] for i in fitted_tree.tree_.feature if i >= 0]
    feature_counts = pd.Series(used_features).value_counts()

    print("\n" + "=" * 60)
    print("THREE MOST SALIENT TREE PATTERNS")
    print("=" * 60)
    print("\nThe three features used most often in the tree are:\n")
    for feature, count in feature_counts.head(3).items():
        print(f"- {feature} (used {count} times)")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Estimated classification accuracy: {mean_accuracy * 100:.2f}%")
    print(f"Accuracy standard deviation: {std_accuracy * 100:.2f}%")
    print(f"Tree depth: {fitted_tree.tree_.max_depth}")
    print(f"Tree nodes: {fitted_tree.tree_.node_count}")
    print(f"Tree leaves: {fitted_tree.tree_.n_leaves}")
    print("\nDecision tree saved as: decision_tree.png")


# ============================================================
# CLUSTERING AND ANOMALY DETECTION
# ============================================================
N_SAMPLE = 20_000
K_RANGE = range(1, 11)
HIER_SETS = ["all", "no_geo"]
LINKAGES = ["ward", "complete", "average", "single"]
DB_MIN_SAMPLES = [5, 10, 20, 50]
DB_EPS_QUANTILES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DB_MAX_NOISE = 0.35
DB_SETS = ["all", "no_geo", "conditions"]
N_DB_REPORT = 5
TSNE_N = 5_000
HEAT_N = 2_000
OUTLIER_Q = 0.01
HIER_MIN_SIZE = 200
SIL_N = 10_000
STRING_COLS = ["Street", "City", "County", "State", "Timezone", "Wind_Direction", "Weather_Condition"]
CLIP_COLS = ["Distance(mi)", "Temperature(F)", "Wind_Chill(F)", "Pressure(in)", "Visibility(mi)",
             "Wind_Speed(mph)", "Precipitation(in)", "duration_minutes"]
RAW_COLS = ["Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)",
            "Precipitation(in)", "Distance(mi)", "duration_minutes", "start_hour", "start_month"]


# --- output helpers
def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else (f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3f}")
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return str(v)


def md(df, index=False):
    if index:
        df = df.reset_index()
    df = df.astype(object)
    cols = list(df.columns)
    print("| " + " | ".join(map(str, cols)) + " |")
    print("|" + "---|" * len(cols))
    for _, r in df.iterrows():
        print("| " + " | ".join(fmt(v) for v in r) + " |")
    print(flush=True)


def severity_pct(values):
    return pd.Series(values).value_counts(normalize=True).reindex([1, 2, 3, 4]).fillna(0) * 100


# --- data loading
def load_random_rows(path, n, seed):
    r = np.random.default_rng(seed)
    frac = min(1.0, n / 3_799_251 * 1.05)
    parts = [c.sample(frac=frac, random_state=int(r.integers(1 << 31)))
             for c in pd.read_csv(path, chunksize=500_000)]
    df = pd.concat(parts)
    return df.sample(n=min(n, len(df)), random_state=seed)


def build_cluster_data(path):
    df = load_random_rows(path, N_SAMPLE, SEED)
    y = df["Severity"].to_numpy()

    df = process_time(df)
    df = process_weather(df)
    F = df.drop(columns=["Severity", "Start_Time", "End_Time"] + STRING_COLS).astype(float)
    log(f"sample {F.shape[0]:,} rows x {F.shape[1]} features; severity counts "
        f"{ {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))} }")
    print("missing values before imputation:", F.isna().sum()[F.isna().sum() > 0].to_dict())

    # Median imputation, clip extreme values to p1-p99, then MinMax scaling
    G = F.fillna(F.median())
    lo, hi = G[CLIP_COLS].quantile(0.01), G[CLIP_COLS].quantile(0.99)
    G[CLIP_COLS] = G[CLIP_COLS].clip(lo, hi, axis=1)
    E = pd.DataFrame(MinMaxScaler().fit_transform(G), columns=G.columns, index=G.index)

    geo = ["Start_Lat", "Start_Lng"]
    year_dur = ["start_year", "duration_minutes"]
    conditions = ["Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)",
                  "Wind_Speed(mph)", "Precipitation(in)", "Sunrise_Sunset", "Civil_Twilight",
                  "Nautical_Twilight", "Astronomical_Twilight", "start_hour", "start_dayofweek",
                  "start_month"] + [c for c in E.columns if c.startswith("weather_")]
    sets = {
        "all": list(E.columns),
        "no_year_dur": [c for c in E.columns if c not in year_dur],
        "no_geo": [c for c in E.columns if c not in geo],
        "conditions": conditions,
    }
    XS = {k: E[v].to_numpy() for k, v in sets.items()}
    print({k: len(v) for k, v in sets.items()}, "features per set")
    return SimpleNamespace(df=df, F=F, E=E, y=y, sets=sets, XS=XS, n=len(E))


# --- evaluation helpers
def sse(X, labels):
    total = 0.0
    for c in np.unique(labels[labels >= 0]):
        pts = X[labels == c]
        total += ((pts - pts.mean(axis=0)) ** 2).sum()
    return total


def silhouette(X, labels):
    m = labels >= 0
    if len(np.unique(labels[m])) < 2:
        return np.nan
    try:
        return silhouette_score(X[m], labels[m], sample_size=min(SIL_N, m.sum()), random_state=SEED)
    except ValueError:
        return np.nan


def compare(a, b):
    return dict(ARI=adjusted_rand_score(a, b), NMI=normalized_mutual_info_score(a, b),
                AMI=adjusted_mutual_info_score(a, b))


def sizes_str(labels, maxn=6):
    vals = np.sort(np.bincount(labels[labels >= 0]))[::-1] / len(labels) * 100
    s = " / ".join(f"{v:.1f}" for v in vals[:maxn]) + (f" / …(+{len(vals) - maxn})" if len(vals) > maxn else "")
    return s + (f" (noise {np.mean(labels < 0) * 100:.1f})" if (labels < 0).any() else "")


def dist_incidence_corr(X, labels):
    pool = np.where(labels >= 0)[0]
    idx = np.random.default_rng(SEED).choice(pool, min(HEAT_N, len(pool)), replace=False)
    lab = labels[idx]
    if len(np.unique(lab)) < 2:
        return np.nan
    i, j = np.triu_indices(len(idx), 1)
    return np.corrcoef(pdist(X[idx]), (lab[i] == lab[j]).astype(float))[0, 1]


def evaluate(rec, d):
    X, lab = d.XS[rec["set"]], rec["labels"]
    h, c, v = homogeneity_completeness_v_measure(d.y, lab)
    rec.update(clusters=len(np.unique(lab[lab >= 0])), sizes=sizes_str(lab), SSE=sse(X, lab),
               silhouette=silhouette(X, lab), corr=dist_incidence_corr(X, lab),
               homog=h, compl=c, vmeas=v)
    return rec


def top_features(d, set_name, mask, k=5):
    Es = d.E[d.sets[set_name]]
    diff = Es[mask].mean() - Es[~mask].mean()
    diff = diff.reindex(diff.abs().sort_values(ascending=False).index)[:k]
    return ", ".join(f"{a} {b:+.2f}" for a, b in diff.items())


def raw_profile(d, name, labels):
    cols = ["start_hour", "start_month", "start_year", "Start_Lat", "Start_Lng", "Temperature(F)",
            "Visibility(mi)", "Distance(mi)", "duration_minutes", "Sunrise_Sunset", "weather_clear",
            "weather_rain", "weather_snow", "weather_fog", "Traffic_Signal", "Junction", "Crossing"]
    shares = ["Sunrise_Sunset", "weather_clear", "weather_rain", "weather_snow", "weather_fog",
              "Traffic_Signal", "Junction", "Crossing"]
    rows = []
    for c in np.unique(labels):
        m = labels == c
        members = d.F[m]
        sev = severity_pct(d.y[m])
        r = {"cluster": "noise" if c == -1 else int(c), "n": int(m.sum()), "%": m.mean() * 100}
        r.update(members[cols].median().to_dict() | {k: members[k].mean() for k in shares})
        r["sev3+4 %"] = sev[3] + sev[4]
        rows.append(r)
    print(f"\n**Cluster members in original units: {name}** (medians; flags/day = share)")
    md(pd.DataFrame(rows))


# --- K-means
def run_kmeans(d):
    log("K-means sweeps")
    sweep, km_fit = {}, {}
    for name, X in d.XS.items():
        rows = []
        for k in K_RANGE:
            t = time.perf_counter()
            km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
            km_fit[(name, k)] = dict(model=km, time=time.perf_counter() - t)
            rows.append(dict(k=k, SSE=km.inertia_,
                             silhouette=silhouette(X, km.labels_) if k > 1 else np.nan,
                             iterations=km.n_iter_))
        sweep[name] = pd.DataFrame(rows)
        print(f"\nK-means sweep, feature set '{name}' ({X.shape[1]} features):")
        md(sweep[name])
    best_k = {s: int(t.loc[t.silhouette.idxmax(), "k"]) for s, t in sweep.items()}
    print("best-silhouette k per feature set:", best_k)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, t in sweep.items():
        ax[0].plot(t.k, t.SSE / t.SSE.iloc[0], "o-", label=name)
        ax[1].plot(t.k[1:], t.silhouette[1:], "o-", label=name)
    ax[0].set(xlabel="k", ylabel="SSE / SSE(k=1)", title="K-means SSE (elbow)")
    ax[1].set(xlabel="k", ylabel="silhouette", title="K-means silhouette")
    ax[0].legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cluster_kmeans_sse.png", dpi=130)
    plt.close()

    best_k3 = {s: int(t[t.k >= 3].loc[t[t.k >= 3].silhouette.idxmax(), "k"]) for s, t in sweep.items()}
    print("best-silhouette k >= 3 per feature set:", best_k3)
    specs = [("all", k) for k in (2, 3, 4, 5, 7)] + [(s, best_k3[s]) for s in d.sets if s != "all"]
    km_exps = []
    for s, k in specs:
        f = km_fit[(s, k)]
        km_exps.append(evaluate(dict(id=f"KM-{s}-k{k}", method="K-means", set=s, k=k,
                                     labels=f["model"].labels_, time=f["time"],
                                     iters=f["model"].n_iter_, model=f["model"]), d))
    return best_k, km_exps


# --- hierarchical clustering
def linkage_from_model(m):
    counts = np.zeros(m.children_.shape[0])
    nl = len(m.labels_)
    for i, (a, b) in enumerate(m.children_):
        counts[i] = sum(1 if c < nl else counts[c - nl] for c in (a, b))
    return np.column_stack([m.children_, m.distances_, counts]).astype(float)


def run_hierarchical(d, best_k):
    log("hierarchical clustering")
    Z, hier_time = {}, {}
    for s in HIER_SETS:
        for link in LINKAGES:
            t = time.perf_counter()
            m = AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage=link).fit(d.XS[s])
            hier_time[(s, link)] = time.perf_counter() - t
            Z[(s, link)] = linkage_from_model(m)
            log(f"  {link:>8} / {s}: {hier_time[(s, link)]:.0f}s")

    hier_k = {s: max(best_k[s], 3) for s in HIER_SETS}
    hier_exps = []
    for s in HIER_SETS:
        for link in LINKAGES:
            lab = fcluster(Z[(s, link)], hier_k[s], criterion="maxclust") - 1
            hier_exps.append(evaluate(dict(id=f"H-{link}-{s}", method="Hierarchical", set=s, link=link,
                                           k=hier_k[s], labels=lab, time=hier_time[(s, link)]), d))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, link in zip(axes.ravel(), LINKAGES):
        dendrogram(Z[("all", link)], truncate_mode="lastp", p=30, no_labels=True, ax=ax,
                   color_threshold=0, above_threshold_color="#4C72B0")
        ax.set_title(f"{link} linkage, feature set 'all' (truncated to 30 leaves)")
        ax.set_ylabel("merge distance")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cluster_dendrograms.png", dpi=130)
    plt.close()
    return Z, hier_exps


# --- DBSCAN
def eligible(r):
    return r["clusters"] >= 2 and np.mean(r["labels"] < 0) <= DB_MAX_NOISE


def best_db(db_all, s, k=1):
    c = [r for r in db_all if r["set"] == s and eligible(r)] or [r for r in db_all if r["set"] == s]
    c = sorted(c, key=lambda r: -np.nan_to_num(r["silhouette"], nan=-1))
    return c[:k]


def run_dbscan(d):
    log("DBSCAN")
    db_all = []
    fig, axes = plt.subplots(1, len(DB_SETS), figsize=(5 * len(DB_SETS), 3.8))
    for ax, s in zip(axes, DB_SETS):
        X = d.XS[s]
        kd = NearestNeighbors(n_neighbors=max(DB_MIN_SAMPLES), n_jobs=-1).fit(X).kneighbors(X)[0]
        for ms in DB_MIN_SAMPLES:
            ax.plot(np.sort(kd[:, ms - 1]), label=f"min_samples={ms}")
        ax.set(title=f"k-distance, set '{s}'", xlabel="points sorted",
               ylabel="distance to (min_samples-1)-th neighbour")
        ax.legend(fontsize=7)

        grid = [(ms, eps) for ms in DB_MIN_SAMPLES
                for eps in sorted(set(np.round(np.quantile(kd[:, ms - 1], DB_EPS_QUANTILES), 4)))]
        t = time.perf_counter()
        graph = NearestNeighbors(radius=max(e for _, e in grid), n_jobs=-1).fit(X) \
            .radius_neighbors_graph(X, mode="distance")
        log(f"  set {s}: neighbour graph {time.perf_counter() - t:.0f}s, {graph.nnz / d.n:.0f} neighbours/point")
        for ms, eps in grid:
            t = time.perf_counter()
            m = DBSCAN(eps=eps, min_samples=ms, metric="precomputed").fit(graph)
            dt = time.perf_counter() - t
            core = np.zeros(d.n, bool)
            core[m.core_sample_indices_] = True
            db_all.append(evaluate(dict(id=f"DB-{s}-ms{ms}-e{eps:.3f}", method="DBSCAN", set=s, eps=eps,
                                        ms=ms, labels=m.labels_, time=dt, core=core), d))
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cluster_dbscan_kdist.png", dpi=130)
    plt.close()

    grid_df = pd.DataFrame([{k: r[k] for k in ["set", "ms", "eps", "clusters", "sizes", "silhouette", "vmeas", "time"]}
                            for r in db_all])
    grid_df["noise %"] = [np.mean(r["labels"] < 0) * 100 for r in db_all]
    print("\nFull DBSCAN grid (silhouette excludes noise):")
    md(grid_df)

    db_exps = best_db(db_all, "all", N_DB_REPORT) + [r for s in DB_SETS[1:] for r in best_db(db_all, s)]
    for r in db_exps:
        lab, core = r["labels"], r["core"]
        r["core_border_noise"] = f"{core.sum()} / {((lab >= 0) & ~core).sum()} / {(lab < 0).sum()}"
    return db_all, db_exps


# --- experiment summaries and comparisons
def print_experiment_tables(d, km_exps, hier_exps, db_exps):
    print("\n===== K-means experiments (Euclidean) =====")
    md(pd.DataFrame([dict(exp=r["id"], features=len(d.sets[r["set"]]), k=r["k"], iterations=r["iters"],
                          time_s=r["time"], sizes_pct=r["sizes"], SSE=r["SSE"], silhouette=r["silhouette"],
                          corr=r["corr"], vmeasure=r["vmeas"]) for r in km_exps]))
    print("===== Hierarchical experiments (Euclidean) =====")
    md(pd.DataFrame([dict(exp=r["id"], link=r["link"], set=r["set"], k=r["k"], time_s=r["time"],
                          sizes_pct=r["sizes"], SSE=r["SSE"], silhouette=r["silhouette"], corr=r["corr"],
                          vmeasure=r["vmeas"]) for r in hier_exps]))
    print("===== DBSCAN experiments (Euclidean) =====")
    md(pd.DataFrame([dict(exp=r["id"], eps=r["eps"], minPts=r["ms"], clusters=r["clusters"], sizes_pct=r["sizes"],
                          core_border_noise=r["core_border_noise"], time_s=r["time"], SSE=r["SSE"],
                          silhouette=r["silhouette"], corr=r["corr"], vmeasure=r["vmeas"]) for r in db_exps]))


def compare_clusterings(d, km_exps, hier_exps, db_all):
    km_all = [r for r in km_exps if r["set"] == "all"]
    P_KM = max(km_all, key=lambda r: np.nan_to_num(r["silhouette"], nan=-1))
    all_hier = [r for r in hier_exps if r["set"] == "all"]
    ok_h = [r for r in all_hier if np.bincount(r["labels"]).max() / d.n <= 0.9]
    P_H = max(ok_h or all_hier, key=lambda r: np.nan_to_num(r["silhouette"], nan=-1))
    primary = {"K-means": P_KM, "Hierarchical": P_H, "DBSCAN": best_db(db_all, "all")[0]}
    print("primary clusterings:", {k: v["id"] for k, v in primary.items()})

    print("\n===== Relative indices: K-means vs the hierarchical k-cuts (same feature set, same k) =====")
    rows = []
    for h in hier_exps:
        km = next((r for r in km_exps if r["set"] == h["set"] and r["k"] == h["k"]), None)
        km_labels = km["labels"] if km else \
            KMeans(n_clusters=h["k"], n_init=10, random_state=SEED).fit(d.XS[h["set"]]).labels_
        rows.append(dict(exp=h["id"], k=h["k"], KM_SSE=sse(d.XS[h["set"]], km_labels), H_SSE=h["SSE"],
                         **compare(km_labels, h["labels"])))
    md(pd.DataFrame(rows))

    print("===== Relative indices between the three primary clusterings (noise = own label) =====")
    names = list(primary)
    md(pd.DataFrame([dict(a=primary[names[i]]["id"], b=primary[names[j]]["id"],
                          **compare(primary[names[i]]["labels"], primary[names[j]]["labels"]))
                     for i in range(3) for j in range(i + 1, 3)]))

    print("===== Relative indices: agreement between K-means experiments (ARI) =====")
    ids = [r["id"] for r in km_exps]
    md(pd.DataFrame([[adjusted_rand_score(a["labels"], b["labels"]) for b in km_exps] for a in km_exps],
                    index=ids, columns=ids).round(2), index=True)

    print("===== External indices & contingency matrices (primary clusterings) =====")
    md(pd.DataFrame([dict(exp=r["id"], homogeneity=r["homog"], completeness=r["compl"], v_measure=r["vmeas"])
                     for r in primary.values()]))
    for r in primary.values():
        cm = contingency_matrix(d.y, r["labels"])
        print(f"\nContingency matrix {r['id']} (rows Severity 1-4, columns cluster; -1 = noise)")
        md(pd.DataFrame(cm, index=[f"sev{s}" for s in np.unique(d.y)],
                        columns=[str(c) for c in np.unique(r["labels"])]), index=True)
    return primary


# --- t-SNE and heatmaps
def plot_embeddings(d, primary):
    log("t-SNE")
    rng = np.random.default_rng(SEED)
    ti = rng.choice(d.n, TSNE_N, replace=False)
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(d.XS["all"][ti])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [(r["id"], r["labels"]) for r in primary.values()] + [("Severity (truth)", d.y)]
    for ax, (nm, lab) in zip(axes.ravel(), panels):
        lt = lab[ti]
        noise = lt < 0
        ax.scatter(emb[noise, 0], emb[noise, 1], c="#bbbbbb", s=3)
        sc = ax.scatter(emb[~noise, 0], emb[~noise, 1], c=lt[~noise] % 20, s=3,
                        cmap="tab20" if nm != "Severity (truth)" else "tab10")
        ax.set_title(nm + (" (grey = noise)" if noise.any() else ""), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if nm == "Severity (truth)" or lt.max() < 10:
            ax.legend(*sc.legend_elements(), fontsize=6, loc="best", markerscale=0.6)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cluster_tsne.png", dpi=130)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, r in zip(axes, primary.values()):
        lab = r["labels"]
        pool = np.where(lab >= 0)[0]
        sample = np.random.default_rng(SEED).choice(pool, min(HEAT_N, len(pool)), replace=False)
        order = sample[np.argsort(lab[sample], kind="stable")]
        im = ax.imshow(squareform(pdist(d.XS["all"][order])), cmap="viridis_r")
        ax.set_title(f"{r['id']}\ncorr(distance, incidence)={r['corr']:.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cluster_heatmaps.png", dpi=130)
    plt.close()

    for r in primary.values():
        raw_profile(d, r["id"], r["labels"])
    return emb, ti


# --- anomaly detection
def km_score(r, d):
    cent = r["model"].cluster_centers_
    dist = np.linalg.norm(d.XS[r["set"]] - cent[r["labels"]], axis=1)
    return dist / pd.Series(dist).groupby(r["labels"]).transform("mean").to_numpy()


def hier_score(Zh, n):
    size = np.concatenate([np.ones(n), Zh[:, 3]])
    height = np.concatenate([np.zeros(n), Zh[:, 2]])
    parent = np.full(2 * n - 1, -1)
    for i, (a, b) in enumerate(Zh[:, :2].astype(int)):
        parent[a] = parent[b] = n + i
    anc = np.arange(2 * n - 1)
    for v in range(2 * n - 2, -1, -1):
        if size[v] < HIER_MIN_SIZE:
            anc[v] = anc[parent[v]]
    return height[anc[:n]]


def db_score(r, d):
    X = d.XS[r["set"]]
    return NearestNeighbors(n_neighbors=1).fit(X[r["core"]]).kneighbors(X)[0].ravel()


def plot_anomaly_grids(d, meth, group, emb, ti):
    cols = 4
    nrows = int(np.ceil(len(group) / cols))

    fig, axes = plt.subplots(nrows, cols, figsize=(4.2 * cols, 3.6 * nrows))
    for ax, r in zip(axes.ravel(), group):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=r["score"][ti], s=3, cmap="viridis")
        m = r["flag"][ti]
        ax.scatter(emb[m, 0], emb[m, 1], s=16, facecolors="none", edgecolors="red", linewidths=0.6)
        plt.colorbar(sc, ax=ax, fraction=0.046)
        ax.set_title(f"{r['short']} = {r['id']}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(group):]:
        ax.axis("off")
    plt.suptitle(f"{meth}: anomaly score f(x) on the t-SNE map (red circles = top 1 %)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cluster_fx_map_{meth.lower().replace('-', '')}.png", dpi=110)
    plt.close()

    fig, axes = plt.subplots(nrows, cols, figsize=(4 * cols, 2.8 * nrows))
    for ax, r in zip(axes.ravel(), group):
        ax.hist(r["score"], bins=60, color="#4C72B0")
        ax.set_yscale("log")
        ax.axvline(np.quantile(r["score"], 1 - OUTLIER_Q), color="r", ls="--")
        ax.set_title(f"{r['short']} = {r['id']}", fontsize=8)
    for ax in axes.ravel()[len(group):]:
        ax.axis("off")
    plt.suptitle(f"{meth}: distribution of f(x) (red = top 1% threshold)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cluster_fx_hist_{meth.lower().replace('-', '')}.png", dpi=120)
    plt.close()


def run_anomaly_detection(d, km_exps, hier_exps, db_exps, Z, primary, emb, ti):
    for r in km_exps:
        r["score"] = km_score(r, d)
    for r in hier_exps:
        r["score"] = hier_score(Z[(r["set"], r["link"])], d.n)
    for r in db_exps:
        r["score"] = db_score(r, d)

    for i, r in enumerate(km_exps, 1):
        r["short"] = f"KM{i}"
    ordered_hier = sorted(hier_exps, key=lambda r: (LINKAGES.index(r["link"]), HIER_SETS.index(r["set"])))
    for i, r in enumerate(ordered_hier, 1):
        r["short"] = f"H{i}"
    for i, r in enumerate(db_exps, 1):
        r["short"] = f"DB{i}"

    print("\n===== Anomaly detection: score statistics and top-1% outliers per experiment =====")
    rows = []
    for r in km_exps + hier_exps + db_exps:
        s = r["score"]
        r["flag"] = s >= np.quantile(s, 1 - OUTLIER_Q)
        q = np.quantile(s, [0.5, 0.99, 1.0])
        sev = severity_pct(d.y[r["flag"]])
        rows.append(dict(exp=r["id"], median=q[0], p99=q[1], max=q[2], flagged=int(r["flag"].sum()),
                         sev3_4_pct=sev[3] + sev[4], top_features=top_features(d, r["set"], r["flag"])))
    md(pd.DataFrame(rows))
    for r in db_exps:
        print(f"{r['id']}: {np.sum(r['labels'] < 0)} noise points; "
              f"{np.sum(r['flag'] & (r['labels'] < 0))}/{r['flag'].sum()} top-1% scores are noise")

    for meth, group in [("K-means", km_exps), ("Hierarchical", hier_exps), ("DBSCAN", db_exps)]:
        group = sorted(group, key=lambda r: int(r["short"][1:] if r["short"][0] == "H" else r["short"][2:]))
        plot_anomaly_grids(d, meth, group, emb, ti)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (meth, r) in zip(axes, primary.items()):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=r["score"][ti], s=4, cmap="viridis")
        plt.colorbar(sc, ax=ax, fraction=0.046)
        ax.set_title(f"{meth} f(x) on t-SNE ({r['id']})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cluster_anomaly_tsne.png", dpi=130)
    plt.close()

    print("\n===== Overlap of top-1% outliers between the primary clusterings =====")
    nm = list(primary)
    rows = []
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = primary[nm[i]]["flag"], primary[nm[j]]["flag"]
            rows.append(dict(a=nm[i], b=nm[j], shared=int((a & b).sum()), jaccard=(a & b).sum() / (a | b).sum(),
                             spearman=spearmanr(primary[nm[i]]["score"], primary[nm[j]]["score"])[0]))
    md(pd.DataFrame(rows))
    cnt = sum(primary[k]["flag"].astype(int) for k in nm)
    print(f"flagged by all three: {(cnt == 3).sum()}; by at least two: {(cnt >= 2).sum()}")

    print("\n===== All experiments: do outliers recur? Jaccard of top-1% sets across K-means experiments =====")
    ids = [r["id"] for r in km_exps]
    md(pd.DataFrame([[(a["flag"] & b["flag"]).sum() / (a["flag"] | b["flag"]).sum() for b in km_exps]
                     for a in km_exps], index=ids, columns=ids).round(2), index=True)

    base = severity_pct(d.y)
    flag_cols = [c for c in d.F.columns if c.startswith("weather_")] + \
                ["Sunrise_Sunset", "Traffic_Signal", "Junction", "Crossing"]
    for meth, r in primary.items():
        m = r["flag"]
        sev = severity_pct(d.y[m])
        print(f"\n===== {meth} ({r['id']}): {m.sum()} flagged outliers =====")
        print(f"severity % outliers {sev.round(1).tolist()} vs all {base.round(1).tolist()}")
        print("most different features:", top_features(d, r["set"], m, 10))
        md(pd.DataFrame({"outliers (median)": d.F[m][RAW_COLS].median(),
                         "rest (median)": d.F[~m][RAW_COLS].median()}).T)
        md(pd.DataFrame({"outliers": d.F[m][flag_cols].mean() * 100,
                         "rest": d.F[~m][flag_cols].mean() * 100}).T)
        print("states: outliers", d.df["State"][m].value_counts(normalize=True).head(4).round(2).to_dict(),
              "| rest", d.df["State"][~m].value_counts(normalize=True).head(4).round(2).to_dict())


def run_clustering(path):
    os.makedirs(FIG_DIR, exist_ok=True)
    d = build_cluster_data(path)
    best_k, km_exps = run_kmeans(d)
    Z, hier_exps = run_hierarchical(d, best_k)
    db_all, db_exps = run_dbscan(d)
    print_experiment_tables(d, km_exps, hier_exps, db_exps)
    primary = compare_clusterings(d, km_exps, hier_exps, db_all)
    emb, ti = plot_embeddings(d, primary)
    run_anomaly_detection(d, km_exps, hier_exps, db_exps, Z, primary, emb, ti)
    log("done")


# ============================================================
# MAIN
# ============================================================
def main():
    raw_csv = sys.argv[1] if len(sys.argv) > 1 else RAW_CSV
    if not os.path.exists(FILTERED_CSV):
        filter_raw_data(raw_csv)

    X, y = load_regression_data(FILTERED_CSV)
    print_cv_results(cross_validate_regressors(X, y))
    run_final_regression(X, y)
    run_classification(FILTERED_CSV)
    run_clustering(FILTERED_CSV)


if __name__ == "__main__":
    main()
