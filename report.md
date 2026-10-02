# Knowledge Discovery from US Accidents Data — Project Report

## Part I: Data Exploration and Preprocessing

Code for basic preprocessing: [`preprocess.py`](preprocess.py). All numbers below were computed from the raw file (row filter, missing-value counts) or from the filtered file (`US_Accidents_filtered.csv`, everything else).

### I.1 Data Description and Exploration

**Domain and size.** The US Accidents dataset is a countrywide collection of car accidents (traffic incidents) in the contiguous US, February 2016 – March 2023, gathered from traffic-incident feeds. The raw file has **7,728,394 records and 46 attributes** (3.06 GB). Each record describes one accident: severity, start/end time, location (coordinates, street, city, county, state, zip, timezone), weather at the nearest station, nearby points of interest (POI flags such as `Crossing`, `Junction`, `Traffic_Signal`), and day/night indicators.

**Attribute types (raw):**

| Kind | Attributes |
|---|---|
| Identifier / provenance | `ID`, `Source` |
| Target | `Severity` (1–4; ordinal, describes impact on traffic, 1 = short delay, 4 = long delay) |
| Time | `Start_Time`, `End_Time`, `Weather_Timestamp` |
| Location | `Start_Lat/Lng`, `End_Lat/Lng`, `Street`, `City`, `County`, `State`, `Zipcode`, `Country`, `Timezone`, `Airport_Code` |
| Numeric | `Distance(mi)`, `Temperature(F)`, `Wind_Chill(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`, `Wind_Speed(mph)`, `Precipitation(in)` |
| Categorical | `Wind_Direction`, `Weather_Condition`, `Sunrise_Sunset`, `Civil/Nautical/Astronomical_Twilight` |
| Boolean POI flags (13) | `Amenity`, `Bump`, `Crossing`, `Give_Way`, `Junction`, `No_Exit`, `Railway`, `Roundabout`, `Station`, `Stop`, `Traffic_Calming`, `Traffic_Signal`, `Turning_Loop` |
| Free text | `Description` |

**Missing values (raw).** Number of missing values per record: 0 → 3,554,549; 1 → 244,702; 2 → 1,943,062; 3 → 511,801; 4 → 1,011,403; up to 18 for a few records. The main culprit is `End_Lat`/`End_Lng` (3,402,762 missing each, always together, so every record lacking them already has ≥ 2 missing values). Other heavy hitters: `Precipitation(in)` 2.20 M, `Wind_Chill(F)` 2.00 M, `Wind_Speed(mph)` 571 k, and 120–177 k each for the other weather fields. As required, the project dataset keeps only records with **at most one** missing value: 3,554,549 + 244,702 = **3,799,251 records**, exactly the count given in the assignment.

**Missing values (kept data).** Because each kept record has at most one missing value, the missing values are spread over disjoint records (their sum is exactly 244,702):

| Attribute | Missing | % of kept |
|---|---|---|
| `Precipitation(in)` | 177,065 | 4.66 |
| `Wind_Chill(F)` | 34,816 | 0.92 |
| `Visibility(mi)` | 9,846 | 0.26 |
| `Street` | 8,328 | 0.22 |
| `Weather_Condition` | 6,926 | 0.18 |
| `Humidity(%)` | 5,461 | 0.14 |
| `Pressure(in)` | 2,245 | 0.06 |
| `Wind_Direction` | 15 | ~0 |

**Target distribution.** `Severity` is extremely imbalanced, and the filter made it more so:

| Severity | Raw data | Kept data (3.80 M) |
|---|---|---|
| 1 | 67,366 (0.87 %) | 27,671 (0.73 %) |
| 2 | 6,156,981 (79.67 %) | 3,545,535 (93.32 %) |
| 3 | 1,299,337 (16.81 %) | 92,025 (2.42 %) |
| 4 | 204,710 (2.65 %) | 134,020 (3.53 %) |

![Severity distribution](figures/severity.png)

A majority-class classifier will therefore already reach ~93 % accuracy; accuracy alone will be a poor measure and per-class precision/recall/F1 will matter (Part II).

#### Salient observations

* **The row filter is not random.** Severity 3 drops from 16.8 % to 2.4 % of the data while severity 2 rises from 79.7 % to 93.3 %. Records that survive are essentially those that have end coordinates and full weather; whatever mechanism produced the missing `End_Lat/Lng` is strongly associated with severity. Conclusions drawn from this subset describe *this* subset, not all US accidents (relevant for the "estimated accuracy over the entire domain" questions in Parts II.2 and III.2).
* **Coverage grows over time.** Accidents per year in the kept data: 2016: 23 k, 2017: 42 k, 2018: 49 k, 2019: 231 k, 2020: 680 k, 2021: 1.07 M, 2022: 1.46 M, 2023 (Jan–Mar): 237 k. The growth reflects expanded data collection, not more accidents, so `year` is a proxy for data coverage and should not be read as a trend.
* **Geography is skewed.** California (25.3 %) and Florida (14.3 %) account for ~40 % of records, followed by TX, VA, NY, PA, SC, NC. Locations follow population centres and the interstate network:

  ![Accident locations](figures/map.png)

* **Strong time-of-day pattern.** Two commute peaks (7–8 AM and 3–5 PM), with the afternoon peak highest (~300 k accidents in the 4 PM hour vs ~60 k at 3 AM). 64.7 % of accidents occur in daylight.

  ![Accidents by hour](figures/hour.png)

* **Weather.** Most accidents occur in benign conditions: `Fair` 46.6 %, `Cloudy` 14.2 %, `Mostly Cloudy` 12.7 %, `Partly Cloudy` 8.7 %. Rain/snow/fog are individually small (`Light Rain` 4.8 %, `Light Snow` 2.1 %, `Fog` 1.5 %). Raw counts say nothing about *risk* (no exposure data), only about where accidents are recorded.
* **POI flags are sparse.** `Crossing` 9.3 %, `Traffic_Signal` 9.1 %, `Junction` 8.1 % are the only common flags; `Roundabout` occurs in just 118 records and `Turning_Loop` is never true.
* **Severity vs. features (kept data).**
  * Severity 1 accidents are unusually often at a `Traffic_Signal` (47.7 % vs 8.6 % for severity 2) and a `Crossing` (35.9 % vs 9.2 %). Severity 3 is over-represented at `Junction` (18.2 % vs 7.8 %). These are the most visible categorical patterns.
  * Severity 4 accidents are the longest on average (`Distance` 1.18 mi vs 0.20 mi for severity 1) and occur in cooler weather (mean 55 °F vs 71 °F for severity 1).
  * Median duration (End − Start) by severity is 35, 102, 30 and 122 min for severities 1–4: not monotonic, so severity is not simply "how long the incident lasted".
* **Correlation.** Linear correlation between `Severity` and every numeric attribute is weak (|r| ≤ 0.09; strongest are latitude/longitude 0.09 and temperature −0.07). `Temperature` and `Wind_Chill` are almost perfectly correlated (r = 0.99), and `Visibility` is moderately negatively correlated with `Humidity` (−0.40). Expect non-linear/interaction patterns (trees, clusters) to matter more than linear ones.

  ![Correlation matrix](figures/corr.png)

#### Data quality issues

* **Impossible values / outliers**: `Temperature(F)` from −89 to **207**; `Wind_Speed(mph)` up to **1,087**; `Pressure(in)` from 0 to **58.6** (real values are ~25–32); `Visibility(mi)` up to 140; `Precipitation(in)` up to 24; `Distance(mi)` up to 155 with 9.3 % exactly 0; accident duration from 2 minutes to **2.2 million minutes** (~4 years; mean 760 min vs median 100 min).
* **Redundant attributes**: `Temperature` ≈ `Wind_Chill`; the four twilight attributes agree with `Sunrise_Sunset` in 96 %, 91 % and 87 % of records.
* **High-cardinality categoricals**: `Street` 209,050 distinct values, `City` 10,792, `County` 1,725, `Weather_Condition` 129, `Wind_Direction` 23 (with inconsistent labels: `VAR`/`Variable`, `W`/`West`, `N`/`North`, `S`/`South`, `E`/`East`).
* **Severity imbalance** (above) and the non-random filter bias.
* **Leakage risk**: `End_Time`, `Distance(mi)` and duration are only known *after* an accident happens and are consequences of it, so a model using them to predict severity would not be usable for prevention.

### I.2 Basic Data Preprocessing

The raw CSV (3 GB) does not fit comfortably in memory (an initial full load exhausted memory/disk), so [`preprocess.py`](preprocess.py) streams it in 500,000-row chunks and writes `US_Accidents_filtered.csv` (3,799,251 rows × 36 columns, ~1 GB). Only two operations are applied, since the assignment asks for a bare minimum:

**1. Row filter — keep records with at most 1 missing value**, with missing values counted over all 46 original columns *before* any column is dropped (so the result matches the 3,799,251 required). No imputation is done here.

**2. Drop 10 attributes that cannot contribute to any later part of the project** (46 → 36 attributes):

| Dropped | Rationale |
|---|---|
| `ID` | Row identifier; unique per row, no pattern to mine. |
| `Source` | Which data provider reported the accident; a property of data collection, not of the accident. |
| `Description` | Free text, unique per row (5 missing in raw). Its contents restate attributes already present (street, location, "expect delays") and cannot be used by the trees, regression or distance-based clustering without a separate NLP pipeline. |
| `Country` | Constant (`US` in every record). |
| `Turning_Loop` | Constant (`False` in every kept record). |
| `Airport_Code` | Id of the weather station that supplied the weather; only a lookup key. The weather values themselves are kept. |
| `Weather_Timestamp` | Time of the weather observation, not of the accident; the accident time is already in `Start_Time`. |
| `Zipcode` | Very high cardinality (values mix 5-digit and `ZIP+4` formats), redundant with `State`/`County`/`City` and `Start_Lat/Lng`. |
| `End_Lat`, `End_Lng` | Redundant with `Start_Lat/Lng` + `Distance(mi)` (accidents are short: median 0.27 mi). Being missing in 44 % of the raw data is what drives the row filter, but after the filter they are complete, and add nothing beyond start point + distance. |

Everything else is kept, even attributes that are redundant or high-cardinality but that a domain expert could use in Part I.3: `End_Time` (needed for duration), `Street`/`City`/`County` (road type, urban density), `Timezone` (local time), `Wind_Chill`, the twilight attributes, and the rare POI flags (`Roundabout` has 118 positives but is not constant).

**Deliberately *not* done in the basic step** (each depends on the model and must be fitted on training folds only, to avoid leakage): imputing the remaining missing values, encoding categoricals, scaling, outlier treatment, discretisation, and feature construction. These are described next.

### I.3 Advanced Data Preprocessing

Preprocessing is kept deliberately light and **identical across all parts of the project** (it lives in `process_time` and `process_weather` in [`preprocess.py`](preprocess.py)). The ideas:

**Light score (considered, not implemented).** There are four twilight definitions (`Sunrise_Sunset`, `Civil_`, `Nautical_`, `Astronomical_Twilight`) that could be collapsed into one numerical value in [0–4] showing how dark it is. We opted to leave them as separate 0/1 attributes (1 = day), since any non-linear model can learn the relationship and the transformation did not seem very profitable.

**Group `Weather_Condition` into boolean weather attributes.** The weather text is very messy: many sparsely populated values, strings that are almost unique, and no relationship preserved between similar values (e.g., *heavy rain* and *rain*). We instead derive one boolean attribute per weather type (rain/drizzle/shower, snow, fog/mist, thunderstorm, windy, hail, freezing/sleet, haze/smoke/dust, fair/clear), set when the string contains the corresponding word. Every weather string becomes useful and related values share attributes.

**Process time.** `Start_Time` is split into hour, day of week, month and year (retaining all the information, but exposing relationships between rows more easily), and the accident duration in minutes (`End_Time − Start_Time`) is added.

**Additional techniques used only for the clustering / anomaly-detection experiments (Parts IV–V; the allowed maximum of two):**
1. *Median imputation* of the remaining missing values (needed: scikit-learn's clustering algorithms cannot handle NaN). In the 20,000-row sample used for clustering this affects `Precipitation` (872 values), `Wind_Chill` (193), `Visibility` (46), `Humidity` (25) and `Pressure` (15).
2. *Clipping to the 1st–99th percentile* of the columns with extreme values (`Distance`, `Temperature`, `Wind_Chill`, `Pressure`, `Visibility`, `Wind_Speed`, `Precipitation`, `duration_minutes`; e.g., durations up to 2.2 M minutes, wind speeds of 1,087 mph). Without it, MinMax scaling would squash all normal values into a thin band near 0.

In addition, the assignment-mandated `MinMaxScaler` (all attributes to [0, 1]) is applied for clustering, and the leftover string columns (`Street`, `City`, `County`, `State`, `Timezone`, `Wind_Direction`, raw `Weather_Condition`) plus `Severity`, `Start_Time` and `End_Time` are not used as features. This leaves **40 numeric features**: 2 coordinates, distance, 7 weather measurements, 12 POI flags, 4 twilight flags, 9 weather flags, and the 5 time attributes (hour, day of week, month, year, duration).

---

## Parts IV and V: Clustering and Anomaly Detection

Code: [`clustering.py`](clustering.py) · full run log with every table: [`clustering_output.txt`](clustering_output.txt) · figures: `figures/cluster_*.png`.

**Setup common to all experiments.**
* **Data:** uniform random sample of **20,000 accidents** (hierarchical clustering, DBSCAN, silhouette and distance matrices need O(n²) memory); severity 1/2/3/4 = 148 / 18,668 / 474 / 710. `Severity` is never a feature, only used for external evaluation.
* **Preprocessing "P"** (Part I.3): `process_time` + `process_weather` → median imputation → clipping to the 1st–99th percentile → `MinMaxScaler` (all attributes in [0, 1]). Feature sets: **all** (40 features), **no_year_dur** (38, without year and duration), **no_geo** (38, without latitude/longitude), **conditions** (23: weather, twilight, hour/day/month only).
* **Distance:** Euclidean in every experiment.
* **Anomaly detection:** each section defines an anomaly score f(x) (higher = more anomalous). The **top 1 % (200 accidents)** are flagged as outliers (our choice; no score has a natural gap). Outliers are described by the difference between the (scaled) attribute means of flagged vs. other accidents; "overall" = the 20,000 sample (severity 3+4 = 5.9 %).
* Times were measured on this machine and fluctuate 2–3× between runs.

---

## 1. Summary of Experiments with K-means

(1) Preprocessing = P with the feature set shown; (2) distance = Euclidean; `KMeans(n_init=10, random_state=0)`.

| Exp | (1) Preprocessing | (3) Iterations | (4) # clusters | (5) % of instances per cluster | (6) SSE | Silhouette |
|---|---|---|---|---|---|---|
| KM1 | P, all (40) | 5 | 2 | 68.4 / 31.6 | 33,967 | **0.307** |
| KM2 | P, all | 10 | 3 | 36.1 / 32.4 / 31.6 | 30,139 | 0.191 |
| KM3 | P, all | 9 | 4 | 36.1 / 32.3 / 16.1 / 15.5 | 28,378 | 0.168 |
| KM4 | P, all | 10 | 5 | 33.8 / 30.4 / 13.7 / 13.3 / 8.8 | 27,093 | 0.171 |
| KM5 | P, all | 8 | 7 | 27.7 / 23.5 / 13.7 / 13.3 / 8.0 / 8.0 / 5.9 | 25,056 | 0.166 |
| KM6 | P, no_year_dur (38) | 12 | 3 | 36.1 / 32.4 / 31.6 | 29,098 | 0.197 |
| KM7 | P, no_geo (38) | 10 | 3 | 36.1 / 32.4 / 31.6 | 27,029 | 0.211 |
| KM8 | P, conditions (23) | 7 | 3 | 36.1 / 32.4 / 31.6 | 19,027 | 0.276 |

SSE is on [0, 1]-scaled data, so it is comparable only for the same feature set.

**(7) Observations** (centroids in original units; t-SNE in Section 5):

| Exp | Observation |
|---|---|
| KM1 | **Day vs. night.** Cluster 1 (31.6 %) is entirely night (median hour 18, 52 °F); cluster 0 is daytime (hour 14, 68 °F). Night has more severe accidents (severity 3+4: 7.1 % vs 5.4 %). |
| KM2 | Night / daytime clear (100 % `weather_clear`) / daytime not clear (0 % clear; rain 13.6 %, snow 4.4 %). |
| KM3, KM4 | Night splits by weather: *night clear* (13.3 %) and *night bad weather* (13.7 %: rain 14.6 %, snow 7.8 %, fog 5.5 %; **severity 3+4 = 8.3 %, the highest cluster**); KM4 adds a small *dusk* cluster (8.8 %). |
| KM5 | New specific groups: *signalised intersections* (5.9 %: 100 % at a traffic signal, 51 % at a crossing, median 0.04 mi vs 0.3 mi) and *daytime bad weather* (8.0 %: rain 57 %, snow 19 %, fog 10 %, visibility 3 mi). |
| KM6–KM8 | The k=3 clustering is **identical** to KM2 (ARI 1.000) after removing year/duration, geography, or everything but weather/light/time: the partition is determined by the twilight and weather flags. The four near-duplicate twilight flags count "day/night" four times in the distance, so it is the first split at every k. |

**(8) Anomaly detection.** f(x) = distance of x to its cluster centroid ÷ mean such distance in its cluster. Plots of the data points by f(x): [`cluster_fx_map_kmeans.png`](figures/cluster_fx_map_kmeans.png) (t-SNE map coloured by f(x), outliers circled) and [`cluster_fx_hist_kmeans.png`](figures/cluster_fx_hist_kmeans.png) (histograms).

| Exp | f(x) median / 99 % / max | Outliers identified? | Analysis of the 200 flagged accidents |
|---|---|---|---|
| KM1 | 0.95 / 1.71 / 2.24 | Weakly | **Adverse weather:** snow 31 %, rain 26 %, fog 14 %, windy 30 % (vs 3 %, 7 %, 2 %, 2 % in the rest), visibility 2 mi (vs 10), humidity 88 %; also crossing 40 % and signal 37 % (vs 9 %); severity 3+4 8.5 % |
| KM2–KM4, KM6, KM7 | 0.94–0.96 / 1.79–1.90 / 2.40–2.55 | Weakly | **Busy road features:** `Crossing` (+0.60 to +0.66), `Traffic_Signal` (+0.42 to +0.47), `Station` (+0.30 to +0.37); severity 3+4 4.5–7.0 % |
| KM5 | 0.94 / 1.80 / 2.51 | Weakly | Crossing (+0.52), Stop (+0.32), Station (+0.32), windy (+0.21); severity 3+4 4 % |
| KM8 | 0.91 / 2.02 / 2.93 | Weakly | Windy (+0.46), low visibility (−0.33), high wind speed (+0.31); severity 3+4 4.5 % |

*Outliers identified?* Only weakly: the scores have a thin, continuous tail (99th percentile = 1.8× the median) with no gap and no tiny cluster, so flagged accidents are the extreme end of a continuum. They are interpretable, though: rare adverse-weather combinations (2 clusters) or rare combinations of road features (3–7 clusters) that no centroid represents. The flagged sets are stable across KM2–KM4/KM6/KM7 (Jaccard 0.77–0.88) but not for the weather-only KM8 (≈ 0.13).

---

## 2. Summary of Experiments with Hierarchical Clustering

(1) Preprocessing = P with the feature set shown; (2) distance = Euclidean. The full tree is built with `AgglomerativeClustering(n_clusters=None, distance_threshold=0)` and cut at k = 3 (the K-means k of the comparison). Two experiments (`all`, `no_geo`) per link type.

| Exp | (1) Preprocessing | (3) Link | (4) # clusters | (5) % of instances per cluster | (6) Time (s) | SSE | Silhouette* |
|---|---|---|---|---|---|---|---|
| H1 | P, all | ward | 3 | 35.2 / 32.7 / 32.1 | 11.9 | 32,304 | 0.148 |
| H2 | P, no_geo | ward | 3 | 35.0 / 32.5 / 32.4 | 9.9 | 29,177 | 0.165 |
| H3 | P, all | complete | 3 | 43.9 / 31.0 / 25.1 | 11.0 | 33,472 | 0.133 |
| H4 | P, no_geo | complete | 3 | 73.2 / 21.9 / 4.9 | 10.4 | 30,577 | 0.284 |
| H5 | P, all | average | 3 | 99.9 / 0.1 / 0.0 | 9.9 | 47,810 | 0.291 |
| H6 | P, no_geo | average | 3 | 100.0 / 0.0 / 0.0 | 9.1 | 44,686 | 0.265 |
| H7 | P, all | single | 3 | 100.0 / 0.0 / 0.0 | 3.4 | 47,864 | 0.303 |
| H8 | P, no_geo | single | 3 | 100.0 / 0.0 / 0.0 | 3.5 | 44,699 | 0.318 |

\* Not meaningful for H5–H8 (one cluster holds essentially all points). K-means k=3 for comparison: SSE 30,139 (all) / 27,029 (no_geo).

**(7) Observations** (dendrograms: [`cluster_dendrograms.png`](figures/cluster_dendrograms.png)):

| Exp | Observation |
|---|---|
| H1, H2 (ward) | Balanced, and close to K-means (ARI 0.69; unchanged without geography: 0.688). Members: cluster 0 (35.2 %) **night** (median hour 17); cluster 1 (32.1 %) **daytime, not clear** (clear 8 %, rain 14 %, snow 4.5 %, junction 15 %); cluster 2 (32.7 %) **daytime, clear** (85 % clear, traffic signal 16 %, crossing 17.5 %). *Nested structure:* one dominant top merge (height ≈ 160 vs ≈ 70 below), a 3–4-group level at heights 45–70, then many merges at ≈ 20. |
| H3, H4 (complete) | Flat dendrogram top (heights 3.5–4.3), no obvious level to cut. With all features the clusters are 44/31/25 % (ARI 0.49 with K-means); without geography 73/22/5 % (ARI 0.35). |
| H5–H8 (average, single) | **Chaining:** the dendrograms are a "staircase" of merges at almost the same height (average ≈ 2.3–3.0, single 1.5–1.9); at k = 3 and larger ≥ 99.9 % of the data is one cluster, the rest are isolated accidents. |

**(8) Anomaly detection.** f(x) = height of the dendrogram merge at which x first belongs to a cluster of **≥ 200 members** (1 % of the data); accidents that join the bulk only at a large distance are isolated from every dense group. Ties in merge heights make the flagged set slightly larger than 200. Plots: [`cluster_fx_map_hierarchical.png`](figures/cluster_fx_map_hierarchical.png), [`cluster_fx_hist_hierarchical.png`](figures/cluster_fx_hist_hierarchical.png).

| Exp | f(x) median / 99 % / max | Flagged | Outliers identified? | Analysis of the flagged accidents |
|---|---|---|---|---|
| H1, H2 | 7.1 / 19.3 / 20.4 and 6.3 / 19.9 / 20.4 | 297, 306 | **Yes** (a group with merge heights 15–20 vs median 7) | **Fog and haze/smoke:** 55.6 % haze, 43.8 % fog (vs 0.3 % and 1.1 % in the rest), 0 % clear, visibility 3 mi, 57 % in California (25 % overall); *less* severe (3+4 = 2.7 % vs 5.9 %) |
| H3, H4 | 2.0 / 3.2–3.4 / 3.7 | 313, 406 | Weakly | Night (twilight flags −0.4 to −0.65) with crossings (+0.77) (H3); night and snow (+0.54), severity 3+4 8.4 % (H4) |
| H5, H6 | 1.0–1.1 / 2.3 / 3.1–3.2 | 248, 202 | Weakly | Crossing (+0.44 to +0.59), rain (+0.39) / night (−0.47), traffic signal (+0.37); severity 3+4 8–10 % |
| H7, H8 | 0.4–0.5 / 1.2 / 1.9 | 200 | Weakly | Crossing (+0.38), night/not clear (−0.30); severity 3+4 8.5–10 % |

---

## 3. Summary of Experiments with DBSCAN

(1) Preprocessing = P with the feature set shown; (2) eps and (3) minPts as listed; distance = Euclidean. A sparse neighbourhood graph (built once, ≈ 1 s) is passed to `DBSCAN(metric="precomputed")`; time = DBSCAN call only. Reported: the five best configurations on `all` (selection: ≥ 2 clusters, ≤ 35 % noise, highest silhouette on non-noise points) plus the best on `no_geo` and `conditions`. Percentages: the six largest clusters; noise in parentheses.

| Exp | (1) Preprocessing | (2) eps | (3) minPts | (4) # clusters | (5) % of instances per cluster (noise) | Core / border / noise | (6) Time (s) | Silhouette |
|---|---|---|---|---|---|---|---|---|
| DB1 | P, all | 0.763 | 50 | 19 | 22.3 / 17.3 / 7.8 / 5.3 / 2.3 / 1.8 … (32.5) | 10,000 / 3,495 / 6,505 | 4.8 | 0.239 |
| DB2 | P, all | 0.981 | 50 | 30 | 22.4 / 17.6 / 8.1 / 5.8 / 3.2 / 2.0 … (19.1) | 14,001 / 2,173 / 3,826 | 6.1 | 0.236 |
| DB3 | P, all | 0.861 | 50 | 25 | 22.4 / 17.5 / 8.0 / 5.7 / 3.1 / 1.9 … (25.0) | 12,000 / 2,997 / 5,003 | 6.0 | 0.232 |
| DB4 | P, all | 0.974 | 20 | 46 | 22.4 / 17.6 / 8.1 / 5.8 / 3.2 / 2.0 … (13.2) | 16,000 / 1,358 / 2,642 | 5.7 | 0.231 |
| DB5 | P, all | 0.814 | 20 | 37 | 22.4 / 17.5 / 8.0 / 5.7 / 3.1 / 1.9 … (19.5) | 13,999 / 2,097 / 3,904 | 5.9 | 0.229 |
| DB6 | P, no_geo | 0.638 | 50 | 18 | 22.2 / 17.1 / 7.7 / 5.2 / 2.4 / 1.8 … (33.0) | 10,000 / 3,407 / 6,593 | 6.0 | 0.295 |
| DB7 | P, conditions | 0.408 | 50 | 7 | 28.5 / 22.2 / 5.2 / 4.2 / 2.9 / 2.6 … (33.8) | 9,999 / 3,236 / 6,765 | 3.8 | 0.346 |

**(7) Observations.**
* **Clusters are groups with the same binary pattern.** DB1: the two largest clusters are **daytime accidents with no road feature** (22.3 % clear, 17.3 % not clear), then the **night** equivalents (7.8 % / 5.3 %). The small clusters combine one road feature with light/weather: junction (1.8 % clear; 1.6 % not clear, severity 3+4 = 11 %), traffic signal (1.2 %; 0.9 % not clear, 9.7 %), crossing (0.8 %), signal + crossing (0.9 %, 0.7 %), rain at 5 mi visibility (2.3 %), snow at ≈ 26 °F (0.5 %). Geography does not appear in the cluster definitions. DB2–DB6 show the same structure; DB7 (weather/light only) gives the fewest clusters and the highest silhouette.
* **Core / border / noise (DB1):** core points (50 %) lie inside the big pattern groups; border points (17 %) are slightly off them; **noise (32.5 %)** are rarer combinations: signal 17 %, crossing 19 %, junction 14 %, rain 15 %, snow 7 %, fog 5 % (vs 9, 9, 8, 7, 2.6, 1.6 % overall), severity 3+4 = 7.1 %.
* **Parameter effect:** larger eps → less noise (32.5 → 19.1 → 13.2 %) and finer clusters; minPts 5 gives > 100 micro-clusters; eps above the 90th percentile of the k-distance merges everything into one cluster (e.g., minPts 20, eps 1.167: 97 % in one cluster).

**(8) Anomaly detection.** f(x) = distance from x to the nearest **core point** (0 for core points, ≤ eps for border points, > eps for noise), the continuous version of DBSCAN's noise definition. Plots: [`cluster_fx_map_dbscan.png`](figures/cluster_fx_map_dbscan.png), [`cluster_fx_hist_dbscan.png`](figures/cluster_fx_hist_dbscan.png).

| Exp | f(x) 99 % / max | Outliers identified? | Analysis of the 200 flagged accidents (all are noise points) |
|---|---|---|---|
| DB1 | 1.84 / 2.46 | **Yes** | **Night** (13.5 % daytime vs 65 %), visibility 4 mi, crossing 47 % (vs 9 %), signal 33 %, junction 20 %, rain 33.5 %; **severity 3+4 = 11 %** (vs 5.9 %) |
| DB2–DB5 | 1.5–1.8 / 2.2–2.5 | Yes | Crossing (+0.45 to +0.65), night (−0.4 to −0.5), signal (+0.31 to +0.48), station (+0.35); severity 3+4 8.5–10 % |
| DB6 | 1.80 / 2.46 | Yes | Night (−0.53), low visibility (−0.45), not clear (−0.40), crossing (+0.38) |
| DB7 | 1.77 / 2.42 | Yes | Low visibility (−0.66), not clear (−0.48), night (−0.48), snow (+0.37) |

*Outliers identified?* Yes: DBSCAN labels 13–34 % of the accidents as noise (depending on parameters), and the most isolated ones are **night-time accidents combining low visibility / adverse weather with a road feature** (crossing, signal, junction): a rare and also more severe combination.

---

## 4. Quantitative Analysis of Results and Discussion

**(1) Initial parameter values.**
* *K-means:* SSE and silhouette for k = 1…10 on each feature set ([`cluster_kmeans_sse.png`](figures/cluster_kmeans_sse.png)). On `all`, SSE drops 47,877 → 33,967 (k=2, −29 %) → 30,139 (−11 %) → 28,378 (−6 %) → 27,093 (−5 %) and then 2–5 % per step: no sharp elbow. The silhouette is maximal at k = 2 (0.307) with local maxima at k = 5 (0.171) and 7 (0.166); best k = 2 on all four feature sets (0.31–0.39). We ran k = 2 (silhouette optimum), 3–5 (SSE bend) and 7 (local maximum).
* *DBSCAN:* the k-distance plots ([`cluster_dbscan_kdist.png`](figures/cluster_dbscan_kdist.png)) have no pronounced knee, so eps was searched from the 30th to the 90th percentile of the k-distance, for minPts ∈ {5, 10, 20, 50} on three feature sets (84 configurations, all in the log). Silhouette rises with minPts (best ≈ 0.25 for 50 vs ≈ 0.20 for 5 on `all`); noise falls with eps (≈ 50–60 % at the 30th percentile, ≈ 2–7 % at the 90th).

**(2) Internal indices** (silhouette on a 10,000-point sample; corr = correlation between the distance and incidence matrices of 2,000 points, ideal → −1; [`cluster_heatmaps.png`](figures/cluster_heatmaps.png)):

| Clustering | SSE | Silhouette | corr(distance, incidence) |
|---|---|---|---|
| K-means k=2 (KM1) | 33,967 | 0.307 | **−0.719** |
| K-means k=3 (KM2) / k=5 (KM4) | 30,139 / 27,093 | 0.191 / 0.171 | −0.556 / −0.578 |
| Ward (H1) | 32,304 | 0.148 | −0.467 |
| Complete (H3 / H4) | 33,472 / 30,577 | 0.133 / 0.284 | −0.487 / −0.709 |
| Average, single (H5, H7) | ≈ 47,800 | not meaningful | −0.05 / n/a |
| DBSCAN DB1 (noise excluded) | 8,857 | 0.239 | −0.617 |

Structure is clearly present (silhouette 0.2–0.4; strongly negative correlations for KM1, DB1 and H4). At k = 3 on `all`, SSE is K-means 30,139 < ward 32,304 < complete 33,472 ≪ average/single ≈ 47,800 (essentially one cluster).

**(3) Relative indices:**

| Comparison | ARI | NMI / AMI |
|---|---|---|
| K-means k=3 vs ward / complete / average / single (`all`) | 0.689 / 0.490 / 0.000 / 0.000 | 0.643 / 0.541 / 0.001 / 0.000 |
| K-means k=3 vs ward / complete (`no_geo`) | 0.688 / 0.346 | 0.641 / 0.495 |
| K-means k = 2↔3, 3↔4, 4↔5, 5↔7 | 0.55, 0.88, 0.89, 0.79 | – |
| K-means k=3: `all` vs no_year_dur / no_geo / conditions | 1.000 / 1.000 / 1.000 | – |
| KM1 vs H1 / KM1 vs DB1 / H1 vs DB1 | 0.456 / 0.118 / 0.361 | 0.568 / 0.302 / 0.508 |

Ward and K-means find essentially the same grouping; K-means clusters are refinements of each other as k grows and do not depend on the feature set; DBSCAN differs (road-feature patterns and noise). SSE versus k (relative SSE) is in the elbow plot above.

**(4) External indices (Severity).** Homogeneity / completeness / V-measure: KM1 0.006 / 0.003 / 0.004; H1 0.007 / 0.002 / 0.003; DB1 0.021 / 0.003 / 0.006; all other experiments V ≤ 0.006. **No clustering corresponds to severity** (consistent with Part I.1). Contingency matrix of KM1 (rows severity 1–4):

| | day (c0) | night (c1) |
|---|---|---|
| sev 1 / sev 2 | 118 / 12,823 | 30 / 5,845 |
| sev 3 / sev 4 | 343 / 389 | 131 / **321** |

Night holds 31.6 % of the accidents but 45 % of the severity-4 ones (ward's night cluster: 48 %).

**(5) Other quantitative results.**
* *Robustness:* removing year/duration, geography, or everything but weather/light/time leaves K-means k=3 unchanged (ARI 1.000); ward's agreement with K-means is the same with and without geography (0.689 vs 0.688).
* *Run time* (20,000 points): K-means 0.06–0.20 s; DBSCAN ≈ 4–6 s (+ ≈ 1 s graph); hierarchical ≈ 3–12 s (single linkage consistently fastest).
* *Outlier agreement* (top 1 %): KM1 vs H1: 2 shared accidents (Jaccard 0.004, Spearman of f(x) 0.66); KM1 vs DB1: 49 (0.14, 0.79); H1 vs DB1: 5 (0.01, 0.69); 1 accident flagged by all three, 54 by at least two.
* *Why the structure is clearer than in an earlier run:* 25 of the 40 features are binary (4 twilight, 9 weather, 12 POI flags) and dominate the distance; an earlier run with a richer one-hot-encoded feature set (121 features) gave silhouettes < 0.14.

---

## 5. Qualitative Analysis of Results and Visualizations

**(1) Visualizations.** t-SNE of 5,000 accidents coloured by KM1, H1, DB1 (noise grey) and by severity; the sorted distance matrices; and the f(x) maps. The t-SNE consists of many small tight "islands", each a group of accidents with the same binary pattern (time of day × weather × road feature). KM1 separates the left group of islands (night); H1 separates night / daytime-not-clear / daytime-clear; DBSCAN assigns the big islands to its largest clusters and leaves many small islands as noise. Severity shows no visible structure. Distance matrices show block structure, strongest for KM1 (−0.72) and DB1 (−0.62), weaker for ward (−0.47).

![t-SNE of the clusterings](figures/cluster_tsne.png)
![Distance matrices sorted by cluster](figures/cluster_heatmaps.png)
![Anomaly score f(x) of K-means experiments](figures/cluster_fx_map_kmeans.png)

**(2) Cluster members** (medians in original units; K-means):

| Cluster | Share | Description and typical members | Severity 3+4 |
|---|---|---|---|
| KM4 c0 | 33.8 % | Day, **not clear**: 13:00, June; rain 13.6 %, snow 4.3 %, 65 °F | 6.0 % |
| KM4 c1 | 30.4 % | Day, **clear**: 14:00, July, 72 °F, southern latitudes | 4.6 % |
| KM4 c2 | 13.3 % | Night, clear: 18:00, September, 54 °F | 6.2 % |
| KM4 c3 | 13.7 % | **Night, bad weather**: rain 14.6 %, snow 7.8 %, fog 5.5 %, 50 °F | **8.3 %** |
| KM4 c4 | 8.8 % | Dusk: 17:00, July | 6.1 % |
| KM5 | 5.9 % | **Signalised intersections**: 100 % signal, 51 % crossing, 0.04 mi, 79 min | 6.4 % |
| KM5 | 8.0 % | **Daytime bad weather**: rain 57 %, snow 19 %, fog 10 %, visibility 3 mi | 7.4 % |

Members of a cluster share light and weather type (and, for larger k, the same road feature); clusters differ mainly in light, precipitation/fog and intersection features. The DBSCAN clusters add geography-free road-feature groups (junction, crossing, signal) and weather groups (rain at reduced visibility, snow at ≈ 26 °F).

**(3) Outlier commonalities.** K-means and DBSCAN (Spearman 0.79; 49 shared of 200) flag accidents combining **night / low visibility / adverse weather with intersection features**: 40–47 % at a crossing (vs 9 %) in KM1 and DB1; the DBSCAN outliers are also more severe (8–11 % vs 5.9 %), the K-means ones are not (4–8.5 %). The ward score isolates a different coherent group, **fog/haze/smoke accidents** (mostly California, lower severity), hardly flagged by the other methods (5 and 2 shared). Complete/average/single linkage again recover the crossing/night family. So one family of outliers is robust across methods (rare road feature × poor light/weather) and one is specific to hierarchical clustering. Within K-means the outliers are stable across k = 3–5 and feature sets (Jaccard 0.6–0.9) but not for weather-only features (0.13).

**(4) Domain interpretation.**
* **Light is the dominant structure.** Night is 32 % of the sample but 45 % of severity-4 accidents (severity 3+4: 7.1 % vs 5.4 % by day); night with rain/snow/fog is the most severe cluster (8.3 %). Possible explanations are reduced visibility and driver impairment, but there are no exposure data (traffic volume), so these are rates *among recorded accidents*, not risk per mile.
* **Bad weather by day** is only slightly more severe than clear days (6.0 % vs 4.6 %), and fog/haze outliers are *less* severe, possibly because drivers slow down (not tested).
* **Intersections are their own accident type**: short (0.04 mi), at signals/crossings, and outliers when combined with night or bad weather: candidates for targeted countermeasures (signal timing, lighting).
* **Geography and year did not matter** for the structure here; results describe accident *circumstances*, not locations.
* **Caveats:** one 20,000-accident sample and seed; the ≤ 1-missing-value filter strongly changes the severity distribution (Part I.1); the 1 % threshold and the DBSCAN grid are our choices; the redundant twilight flags weight "light" four-fold.
