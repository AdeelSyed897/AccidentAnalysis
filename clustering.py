"""Parts IV (Clustering) and V (Cluster-based Anomaly Detection).

Run:  python clustering.py > clustering_output.txt     (plots go to figures/cluster_*.png)

Preprocessing (identical in spirit to the rest of the project, see report.md):
  * process_time / process_weather from preprocess.py (hour, day-of-week, month, year,
    duration, twilight 0/1 flags, 9 weather flags)
  * the remaining string columns (Street, City, County, State, Timezone, Wind_Direction,
    Weather_Condition) are dropped
  * the only two extra techniques, both needed for clustering:
      1. median imputation of the remaining NaNs (sklearn cannot cluster NaN)
      2. clipping of the columns with extreme values to their 1st-99th percentile
  * MinMaxScaler on every attribute (required by the assignment)
`Severity` is never a feature; it is only used for external evaluation.

All experiments run on a random sample of N_SAMPLE accidents (hierarchical clustering, DBSCAN,
silhouette and the distance/incidence matrix need O(n^2) memory).
Sections are separated with `# %%` so the file also runs cell-by-cell in VS Code / Jupyter.
"""
# %% imports and settings
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             homogeneity_completeness_v_measure, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from preprocess import process_time, process_weather

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220, "display.max_columns", 60)

CSV = "US_Accidents_filtered.csv"
FIG = "figures/cluster_"
SEED = 0
N_SAMPLE = 20_000
K_RANGE = range(1, 11)
HIER_SETS = ["all", "no_geo"]
LINKAGES = ["ward", "complete", "average", "single"]
DB_MIN_SAMPLES = [5, 10, 20, 50]
DB_EPS_QUANTILES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DB_MAX_NOISE = 0.35
DB_SETS = ["all", "no_geo", "conditions"]
N_DB_REPORT = 5          # best DBSCAN configurations reported on the "all" feature set
TSNE_N = 5_000
HEAT_N = 2_000
OUTLIER_Q = 0.01
HIER_MIN_SIZE = 200
SIL_N = 10_000
T0 = time.time()
rng = np.random.default_rng(SEED)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else (f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3f}")
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return str(v)


def md(df, index=False):
    """Print a DataFrame as a markdown table (copied straight into report.md)."""
    if index:
        df = df.reset_index()
    df = df.astype(object)          # keeps integer columns as integers
    cols = list(df.columns)
    print("| " + " | ".join(map(str, cols)) + " |")
    print("|" + "---|" * len(cols))
    for _, r in df.iterrows():
        print("| " + " | ".join(fmt(v) for v in r) + " |")
    print(flush=True)


# %% load a random sample and preprocess (process_time + process_weather + 2 extras)
def load_random_rows(path, n, seed):
    r = np.random.default_rng(seed)
    frac = min(1.0, n / 3_799_251 * 1.05)
    parts = [c.sample(frac=frac, random_state=int(r.integers(1 << 31)))
             for c in pd.read_csv(path, chunksize=500_000)]
    df = pd.concat(parts)
    return df.sample(n=min(n, len(df)), random_state=seed)


df = load_random_rows(CSV, N_SAMPLE, SEED)
y = df["Severity"].to_numpy()
STRING_COLS = ["Street", "City", "County", "State", "Timezone", "Wind_Direction", "Weather_Condition"]

out = process_time(df)                       # adds start_hour/dayofweek/month/year, duration_minutes, twilight flags
df = df if out is None else out
df = process_weather(df)
F = df.drop(columns=["Severity", "Start_Time", "End_Time"] + STRING_COLS)
F = F.astype(float)                          # POI booleans -> 0/1
log(f"sample {F.shape[0]:,} rows x {F.shape[1]} features; severity counts "
    f"{ {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))} }")
print("missing values before imputation:", F.isna().sum()[F.isna().sum() > 0].to_dict())

# extra technique 1: median imputation; extra technique 2: clip extreme values to p1-p99
CLIP_COLS = ["Distance(mi)", "Temperature(F)", "Wind_Chill(F)", "Pressure(in)", "Visibility(mi)",
             "Wind_Speed(mph)", "Precipitation(in)", "duration_minutes"]
G = F.fillna(F.median())
lo, hi = G[CLIP_COLS].quantile(0.01), G[CLIP_COLS].quantile(0.99)
G[CLIP_COLS] = G[CLIP_COLS].clip(lo, hi, axis=1)
E = pd.DataFrame(MinMaxScaler().fit_transform(G), columns=G.columns, index=G.index)

GEO = ["Start_Lat", "Start_Lng"]
YEAR_DUR = ["start_year", "duration_minutes"]
CONDITIONS = ["Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)",
              "Wind_Speed(mph)", "Precipitation(in)", "Sunrise_Sunset", "Civil_Twilight",
              "Nautical_Twilight", "Astronomical_Twilight", "start_hour", "start_dayofweek",
              "start_month"] + [c for c in E.columns if c.startswith("weather_")]
FEATURE_SETS = {
    "all": list(E.columns),
    "no_year_dur": [c for c in E.columns if c not in YEAR_DUR],
    "no_geo": [c for c in E.columns if c not in GEO],
    "conditions": CONDITIONS,
}
XS = {k: E[v].to_numpy() for k, v in FEATURE_SETS.items()}
print({k: len(v) for k, v in FEATURE_SETS.items()}, "features per set")
n = len(E)


# %% shared helpers
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


def evaluate(rec):
    X, lab = XS[rec["set"]], rec["labels"]
    h, c, v = homogeneity_completeness_v_measure(y, lab)
    rec.update(clusters=len(np.unique(lab[lab >= 0])), sizes=sizes_str(lab), SSE=sse(X, lab),
               silhouette=silhouette(X, lab), corr=dist_incidence_corr(X, lab),
               homog=h, compl=c, vmeas=v)
    return rec


def raw_profile(name, labels):
    """Cluster members described in ORIGINAL units (means; flags are shares of members)."""
    cols = ["start_hour", "start_month", "start_year", "Start_Lat", "Start_Lng", "Temperature(F)",
            "Visibility(mi)", "Distance(mi)", "duration_minutes", "Sunrise_Sunset", "weather_clear",
            "weather_rain", "weather_snow", "weather_fog", "Traffic_Signal", "Junction", "Crossing"]
    rows = []
    for c in np.unique(labels):
        m = labels == c
        sev = pd.Series(y[m]).value_counts(normalize=True).reindex([1, 2, 3, 4]).fillna(0) * 100
        r = {"cluster": "noise" if c == -1 else int(c), "n": int(m.sum()), "%": m.mean() * 100}
        r.update(F.loc[F.index[m], cols].median().to_dict() | {k: F.loc[F.index[m], k].mean() for k in
                  ["Sunrise_Sunset", "weather_clear", "weather_rain", "weather_snow", "weather_fog",
                   "Traffic_Signal", "Junction", "Crossing"]})
        r["sev3+4 %"] = sev[3] + sev[4]
        rows.append(r)
    print(f"\n**Cluster members in original units: {name}** (medians; flags/day = share)")
    md(pd.DataFrame(rows))


def top_features(cols_set, mask, k=5):
    Es = E[FEATURE_SETS[cols_set]]
    d = Es[mask].mean() - Es[~mask].mean()
    d = d.reindex(d.abs().sort_values(ascending=False).index)[:k]
    return ", ".join(f"{a} {b:+.2f}" for a, b in d.items())


# %% K-means: sweep k = 1..10 on each feature set (SSE elbow + silhouette)
log("K-means sweeps")
sweep, km_fit = {}, {}
for name, X in XS.items():
    rows = []
    for k in K_RANGE:
        t = time.perf_counter()
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        km_fit[(name, k)] = dict(model=km, time=time.perf_counter() - t)
        rows.append(dict(k=k, SSE=km.inertia_, silhouette=silhouette(X, km.labels_) if k > 1 else np.nan,
                         iterations=km.n_iter_))
    sweep[name] = pd.DataFrame(rows)
    print(f"\nK-means sweep, feature set '{name}' ({X.shape[1]} features):")
    md(sweep[name])
BEST_K = {s: int(d.loc[d.silhouette.idxmax(), "k"]) for s, d in sweep.items()}
print("best-silhouette k per feature set:", BEST_K)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for name, d in sweep.items():
    ax[0].plot(d.k, d.SSE / d.SSE.iloc[0], "o-", label=name)
    ax[1].plot(d.k[1:], d.silhouette[1:], "o-", label=name)
ax[0].set(xlabel="k", ylabel="SSE / SSE(k=1)", title="K-means SSE (elbow)")
ax[1].set(xlabel="k", ylabel="silhouette", title="K-means silhouette"); ax[0].legend()
plt.tight_layout(); plt.savefig(FIG + "kmeans_sse.png", dpi=130); plt.close()

# K-means experiments. Different feature sets give (nearly) identical clusterings for the same k
# (checked with ARI), so the experiments vary k on the full feature set ("all": k = 2 is the global
# silhouette optimum, 3-5 and 7 are the k values around the SSE bend / local silhouette maxima) plus
# the best k >= 3 of every other feature set.
BEST_K3 = {s: int(d[d.k >= 3].loc[d[d.k >= 3].silhouette.idxmax(), "k"]) for s, d in sweep.items()}
print("best-silhouette k >= 3 per feature set:", BEST_K3)
KM_SPECS = [("all", k) for k in (2, 3, 4, 5, 7)] + [(s, BEST_K3[s]) for s in FEATURE_SETS if s != "all"]
km_exps = []
for s, k in KM_SPECS:
    if True:
        f = km_fit[(s, k)]
        km_exps.append(evaluate(dict(id=f"KM-{s}-k{k}", method="K-means", set=s, k=k, labels=f["model"].labels_,
                                     time=f["time"], iters=f["model"].n_iter_, model=f["model"])))


# %% Hierarchical: full trees for all four linkages
log("hierarchical clustering")


def linkage_from_model(m):
    counts = np.zeros(m.children_.shape[0])
    nl = len(m.labels_)
    for i, (a, b) in enumerate(m.children_):
        counts[i] = sum(1 if c < nl else counts[c - nl] for c in (a, b))
    return np.column_stack([m.children_, m.distances_, counts]).astype(float)


Z, hier_time = {}, {}
for s in HIER_SETS:
    for link in LINKAGES:
        t = time.perf_counter()
        m = AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage=link).fit(XS[s])
        hier_time[(s, link)] = time.perf_counter() - t
        Z[(s, link)] = linkage_from_model(m)
        log(f"  {link:>8} / {s}: {hier_time[(s, link)]:.0f}s")

HIER_K = {s: max(BEST_K[s], 3) for s in HIER_SETS}      # cut at the K-means k (at least 3) of that feature set
hier_exps = []
for s in HIER_SETS:
    for link in LINKAGES:
        lab = fcluster(Z[(s, link)], HIER_K[s], criterion="maxclust") - 1
        hier_exps.append(evaluate(dict(id=f"H-{link}-{s}", method="Hierarchical", set=s, link=link, k=HIER_K[s],
                                       labels=lab, time=hier_time[(s, link)])))

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, link in zip(axes.ravel(), LINKAGES):
    dendrogram(Z[("all", link)], truncate_mode="lastp", p=30, no_labels=True, ax=ax,
               color_threshold=0, above_threshold_color="#4C72B0")
    ax.set_title(f"{link} linkage, feature set 'all' (truncated to 30 leaves)"); ax.set_ylabel("merge distance")
plt.tight_layout(); plt.savefig(FIG + "dendrograms.png", dpi=130); plt.close()


# %% DBSCAN: k-distance plots + (eps, min_samples) grid on three feature sets
log("DBSCAN")
db_all, db_graph = [], {}
fig, axes = plt.subplots(1, len(DB_SETS), figsize=(5 * len(DB_SETS), 3.8))
for ax, s in zip(axes, DB_SETS):
    X = XS[s]
    kd = NearestNeighbors(n_neighbors=max(DB_MIN_SAMPLES), n_jobs=-1).fit(X).kneighbors(X)[0]
    for ms in DB_MIN_SAMPLES:
        ax.plot(np.sort(kd[:, ms - 1]), label=f"min_samples={ms}")
    ax.set(title=f"k-distance, set '{s}'", xlabel="points sorted", ylabel="distance to (min_samples-1)-th neighbour")
    ax.legend(fontsize=7)
    grid = [(ms, eps) for ms in DB_MIN_SAMPLES
            for eps in sorted(set(np.round(np.quantile(kd[:, ms - 1], DB_EPS_QUANTILES), 4)))]
    t = time.perf_counter()
    graph = NearestNeighbors(radius=max(e for _, e in grid), n_jobs=-1).fit(X).radius_neighbors_graph(X, mode="distance")
    db_graph[s] = graph
    log(f"  set {s}: neighbour graph {time.perf_counter() - t:.0f}s, {graph.nnz / n:.0f} neighbours/point")
    for ms, eps in grid:
        t = time.perf_counter()
        m = DBSCAN(eps=eps, min_samples=ms, metric="precomputed").fit(graph)
        dt = time.perf_counter() - t
        core = np.zeros(n, bool); core[m.core_sample_indices_] = True
        lab = m.labels_
        db_all.append(evaluate(dict(id=f"DB-{s}-ms{ms}-e{eps:.3f}", method="DBSCAN", set=s, eps=eps, ms=ms,
                                    labels=lab, time=dt, core=core)))
plt.tight_layout(); plt.savefig(FIG + "dbscan_kdist.png", dpi=130); plt.close()

dbg = pd.DataFrame([{k: r[k] for k in ["set", "ms", "eps", "clusters", "sizes", "silhouette", "vmeas", "time"]}
                    for r in db_all])
dbg["noise %"] = [np.mean(r["labels"] < 0) * 100 for r in db_all]
print("\nFull DBSCAN grid (silhouette excludes noise):")
md(dbg)


def eligible(r):
    return r["clusters"] >= 2 and np.mean(r["labels"] < 0) <= DB_MAX_NOISE


def best_db(s, k=1):
    c = [r for r in db_all if r["set"] == s and eligible(r)] or [r for r in db_all if r["set"] == s]
    c = sorted(c, key=lambda r: -np.nan_to_num(r["silhouette"], nan=-1))
    return c[:k]


db_exps = best_db("all", N_DB_REPORT) + [r for s in DB_SETS[1:] for r in best_db(s)]


# %% experiment summary tables (Part IV questions 1-3)
for r in db_exps:
    lab = r["labels"]
    core = r["core"]
    r["core_border_noise"] = f"{core.sum()} / {((lab >= 0) & ~core).sum()} / {(lab < 0).sum()}"

print("\n===== K-means experiments (Euclidean) =====")
md(pd.DataFrame([dict(exp=r["id"], features=len(FEATURE_SETS[r["set"]]), k=r["k"], iterations=r["iters"],
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

# %% primary clusterings (one per method), used for the visual and anomaly analysis
P_KM = max([r for r in km_exps if r["set"] == "all"], key=lambda r: np.nan_to_num(r["silhouette"], nan=-1))
ok_h = [r for r in hier_exps if r["set"] == "all" and
        np.bincount(r["labels"]).max() / n <= 0.9]
P_H = max(ok_h or [r for r in hier_exps if r["set"] == "all"], key=lambda r: np.nan_to_num(r["silhouette"], nan=-1))
P_DB = best_db("all")[0]
primary = {"K-means": P_KM, "Hierarchical": P_H, "DBSCAN": P_DB}
print("primary clusterings:", {k: v["id"] for k, v in primary.items()})

# %% Part IV Q4: relative / external indices
print("\n===== Relative indices: K-means vs the hierarchical k-cuts (same feature set, same k) =====")
rows = []
for h in hier_exps:
    km = next(r for r in km_exps if r["set"] == h["set"] and r["k"] == h["k"]) if any(
        r["set"] == h["set"] and r["k"] == h["k"] for r in km_exps) else None
    mdl = km["labels"] if km else KMeans(n_clusters=h["k"], n_init=10, random_state=SEED).fit(XS[h["set"]]).labels_
    rows.append(dict(exp=h["id"], k=h["k"], KM_SSE=sse(XS[h["set"]], mdl), H_SSE=h["SSE"], **compare(mdl, h["labels"])))
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
    cm = contingency_matrix(y, r["labels"])
    print(f"\nContingency matrix {r['id']} (rows Severity 1-4, columns cluster; -1 = noise)")
    md(pd.DataFrame(cm, index=[f"sev{s}" for s in np.unique(y)], columns=[str(c) for c in np.unique(r["labels"])]),
       index=True)

# %% visualisation (t-SNE) and heatmaps
log("t-SNE")
ti = rng.choice(n, TSNE_N, replace=False)
emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(XS["all"][ti])
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (nm, lab) in zip(axes.ravel(), [(r["id"], r["labels"]) for r in primary.values()] + [("Severity (truth)", y)]):
    lt = lab[ti]
    noise = lt < 0                                    # DBSCAN noise is drawn in grey, clusters cycle through tab20
    ax.scatter(emb[noise, 0], emb[noise, 1], c="#bbbbbb", s=3)
    sc = ax.scatter(emb[~noise, 0], emb[~noise, 1], c=lt[~noise] % 20, s=3, cmap="tab20" if nm != "Severity (truth)" else "tab10")
    ax.set_title(nm + (" (grey = noise)" if noise.any() else ""), fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    if nm in ("Severity (truth)",) or lt.max() < 10:
        ax.legend(*sc.legend_elements(), fontsize=6, loc="best", markerscale=0.6)
plt.tight_layout(); plt.savefig(FIG + "tsne.png", dpi=130); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, r in zip(axes, primary.values()):
    lab = r["labels"]; pool = np.where(lab >= 0)[0]
    hi = np.random.default_rng(SEED).choice(pool, min(HEAT_N, len(pool)), replace=False)
    order = hi[np.argsort(lab[hi], kind="stable")]
    im = ax.imshow(squareform(pdist(XS["all"][order])), cmap="viridis_r")
    ax.set_title(f"{r['id']}\ncorr(distance, incidence)={r['corr']:.3f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout(); plt.savefig(FIG + "heatmaps.png", dpi=130); plt.close()

for r in primary.values():
    raw_profile(r["id"], r["labels"])


# %% PART V: anomaly scores for every reported experiment
def km_score(r):
    cent = r["model"].cluster_centers_
    d = np.linalg.norm(XS[r["set"]] - cent[r["labels"]], axis=1)
    return d / pd.Series(d).groupby(r["labels"]).transform("mean").to_numpy()


def hier_score(Zh):
    size = np.concatenate([np.ones(n), Zh[:, 3]]); height = np.concatenate([np.zeros(n), Zh[:, 2]])
    parent = np.full(2 * n - 1, -1)
    for i, (a, b) in enumerate(Zh[:, :2].astype(int)):
        parent[a] = parent[b] = n + i
    anc = np.arange(2 * n - 1)
    for v in range(2 * n - 2, -1, -1):
        if size[v] < HIER_MIN_SIZE:
            anc[v] = anc[parent[v]]
    return height[anc[:n]]


def db_score(r):
    return NearestNeighbors(n_neighbors=1).fit(XS[r["set"]][r["core"]]).kneighbors(XS[r["set"]])[0].ravel()


for r in km_exps:
    r["score"] = km_score(r)
for r in hier_exps:
    r["score"] = hier_score(Z[(r["set"], r["link"])])
for r in db_exps:
    r["score"] = db_score(r)

# short labels used in the report: KM1.., H1.. (ordered by linkage, then feature set), DB1..
for i, r in enumerate(km_exps, 1):
    r["short"] = f"KM{i}"
for i, r in enumerate(sorted(hier_exps, key=lambda r: (LINKAGES.index(r["link"]), HIER_SETS.index(r["set"]))), 1):
    r["short"] = f"H{i}"
for i, r in enumerate(db_exps, 1):
    r["short"] = f"DB{i}"

print("\n===== Anomaly detection: score statistics and top-1% outliers per experiment =====")
rows = []
for r in km_exps + hier_exps + db_exps:
    s = r["score"]; r["flag"] = s >= np.quantile(s, 1 - OUTLIER_Q)
    q = np.quantile(s, [0.5, 0.99, 1.0])
    sev = pd.Series(y[r["flag"]]).value_counts(normalize=True).reindex([1, 2, 3, 4]).fillna(0) * 100
    rows.append(dict(exp=r["id"], median=q[0], p99=q[1], max=q[2], flagged=int(r["flag"].sum()),
                     sev3_4_pct=sev[3] + sev[4], top_features=top_features(r["set"], r["flag"])))
md(pd.DataFrame(rows))
for r in db_exps:
    print(f"{r['id']}: {np.sum(r['labels'] < 0)} noise points; "
          f"{np.sum(r['flag'] & (r['labels'] < 0))}/{r['flag'].sum()} top-1% scores are noise")

for meth, group in [("K-means", km_exps), ("Hierarchical", hier_exps), ("DBSCAN", db_exps)]:
    group = sorted(group, key=lambda r: int(r["short"][1:] if r["short"][0] == "H" else r["short"][2:]))
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(len(group) / cols)), cols, figsize=(4.2 * cols, 3.6 * np.ceil(len(group) / cols)))
    for ax, r in zip(axes.ravel(), group):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=r["score"][ti], s=3, cmap="viridis")
        m = r["flag"][ti]
        ax.scatter(emb[m, 0], emb[m, 1], s=16, facecolors="none", edgecolors="red", linewidths=0.6)
        plt.colorbar(sc, ax=ax, fraction=0.046); ax.set_title(f"{r['short']} = {r['id']}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(group):]:
        ax.axis("off")
    plt.suptitle(f"{meth}: anomaly score f(x) on the t-SNE map (red circles = top 1 %)"); plt.tight_layout()
    plt.savefig(FIG + f"fx_map_{meth.lower().replace('-', '')}.png", dpi=110); plt.close()

    cols = 4
    fig, axes = plt.subplots(int(np.ceil(len(group) / cols)), cols, figsize=(4 * cols, 2.8 * np.ceil(len(group) / cols)))
    for ax, r in zip(axes.ravel(), group):
        ax.hist(r["score"], bins=60, color="#4C72B0"); ax.set_yscale("log")
        ax.axvline(np.quantile(r["score"], 1 - OUTLIER_Q), color="r", ls="--"); ax.set_title(f"{r['short']} = {r['id']}", fontsize=8)
    for ax in axes.ravel()[len(group):]:
        ax.axis("off")
    plt.suptitle(f"{meth}: distribution of f(x) (red = top 1% threshold)"); plt.tight_layout()
    plt.savefig(FIG + f"fx_hist_{meth.lower().replace('-', '')}.png", dpi=120); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, (meth, r) in zip(axes, primary.items()):
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=r["score"][ti], s=4, cmap="viridis")
    plt.colorbar(sc, ax=ax, fraction=0.046); ax.set_title(f"{meth} f(x) on t-SNE ({r['id']})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(FIG + "anomaly_tsne.png", dpi=130); plt.close()

# %% overlap and thorough analysis of the primary outliers
print("\n===== Overlap of top-1% outliers between the primary clusterings =====")
nm = list(primary); rows = []
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
md(pd.DataFrame([[(a["flag"] & b["flag"]).sum() / (a["flag"] | b["flag"]).sum() for b in km_exps] for a in km_exps],
                index=ids, columns=ids).round(2), index=True)

RAW_COLS = ["Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)",
            "Precipitation(in)", "Distance(mi)", "duration_minutes", "start_hour", "start_month"]
for meth, r in primary.items():
    m = r["flag"]
    sev = pd.Series(y[m]).value_counts(normalize=True).reindex([1, 2, 3, 4]).fillna(0) * 100
    base = pd.Series(y).value_counts(normalize=True).reindex([1, 2, 3, 4]).fillna(0) * 100
    print(f"\n===== {meth} ({r['id']}): {m.sum()} flagged outliers =====")
    print(f"severity % outliers {sev.round(1).tolist()} vs all {base.round(1).tolist()}")
    print("most different features:", top_features(r["set"], m, 10))
    md(pd.DataFrame({"outliers (median)": F.loc[F.index[m], RAW_COLS].median(),
                     "rest (median)": F.loc[F.index[~m], RAW_COLS].median()}).T)
    flags = [c for c in F.columns if c.startswith("weather_")] + ["Sunrise_Sunset", "Traffic_Signal", "Junction", "Crossing"]
    md(pd.DataFrame({"outliers": F.loc[F.index[m], flags].mean() * 100,
                     "rest": F.loc[F.index[~m], flags].mean() * 100}).T)
    print("states: outliers", df.loc[df.index[m], "State"].value_counts(normalize=True).head(4).round(2).to_dict(),
          "| rest", df.loc[df.index[~m], "State"].value_counts(normalize=True).head(4).round(2).to_dict())
log("done")
