"""Part I.3 - Advanced data preprocessing for the (basic-preprocessed) US Accidents data.

Implements the ideas in report.md as an sklearn transformer, `AdvancedPreprocessor`.
Everything that depends on the data (medians, top cities, city frequencies, kept POI
flags, one-hot vocabularies, MinMax scaling) is learned in `fit` from the TRAINING data
only and re-applied unchanged in `transform`, so it can sit inside a cross-validation
pipeline without leakage (Parts II.1 / III.1). Rule-based steps (weather grouping,
hour extraction, road-type parsing, ...) are stateless.

Input : DataFrame from US_Accidents_filtered.csv (see load_filtered), WITHOUT the target.
Output: numeric DataFrame (float32) ready for sklearn models.

Typical use (classification, inside the CV loop):

    X, y_cls, y_reg = make_targets(load_filtered("US_Accidents_filtered.csv"))
    pipe = build_pipeline(DecisionTreeClassifier(max_depth=5))
    cross_validate(pipe, X, y_cls, cv=StratifiedKFold(10, shuffle=True, random_state=0), ...)

Regression / Linear Regression: pass impute=True (assignment: impute for Linear Regression
only). Clustering: scale=True and encoding="onehot". Trees/forests: scale=False; without
impute=True the numeric weather columns may keep NaN, which sklearn's DecisionTree and
RandomForest (recent versions) accept natively.

Imbalance: this script does not resample; pass class_weight="balanced" to the models if
wanted (fold-safe) and report per-class metrics.
"""
import argparse
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

DEFAULT_CSV = "US_Accidents_filtered.csv"

POI_FLAGS = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
    "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal",
]
INTERSECTION_FLAGS = ["Junction", "Crossing", "Traffic_Signal", "Stop", "Give_Way", "Roundabout"]
ROAD_FEATURE_FLAGS = ["Bump", "Traffic_Calming", "Railway", "No_Exit"]
TWILIGHT_COLS = ["Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight"]

ADVERSE_WEATHER = {"Rain", "Heavy Rain", "Thunderstorm", "Snow/Ice", "Fog/Haze/Smoke", "Wind/Dust"}

# Physically plausible ranges; values outside are treated as sensor errors -> NaN.
VALID_RANGE = {
    "Temperature(F)": (-60, 130),
    "Pressure(in)": (25, 32),
    "Wind_Speed(mph)": (0, 150),
}

SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
          6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"}

CENSUS_REGION = {}
for _region, _states in {
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South": "DE FL GA MD NC SC VA DC WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}.items():
    for _s in _states.split():
        CENSUS_REGION[_s] = _region

# Continuous engineered attributes (the ones MinMax-scaled when scale=True).
CONTINUOUS = [
    "Start_Lat", "Start_Lng", "hour", "day_of_week", "month", "log_distance", "Temperature",
    "Humidity", "Pressure", "Visibility", "Wind_Speed", "Precipitation", "light_level",
    "n_poi", "duration_log", "city_log_count",
]
# Numeric columns imputed with the training median when impute=True.
# (Precipitation is imputed with 0 instead: a missing value most plausibly means none.)
MEDIAN_IMPUTE = ["Temperature", "Humidity", "Pressure", "Visibility", "Wind_Speed",
                 "light_level", "duration_log"]


# --------------------------------------------------------------------------- helpers
def _map_unique(s, func):
    """Apply `func` to each distinct non-null value once (fast for high-cardinality text)."""
    mapping = {u: func(u) for u in s.dropna().unique()}
    return s.map(mapping)


def _as_datetime(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, format="mixed")


def _weather_group(cond):
    c = cond.lower()
    if re.search(r"thunder|storm", c):
        return "Thunderstorm"
    if re.search(r"\b(snow|sleet|ice|wintry|freezing|hail)\b", c):
        return "Snow/Ice"
    if re.search(r"rain|drizzle|shower", c):
        return "Heavy Rain" if "heavy" in c else "Rain"
    if re.search(r"fog|haze|mist|smoke", c):
        return "Fog/Haze/Smoke"
    if re.search(r"wind|dust|sand|squall|tornado|funnel", c):
        return "Wind/Dust"
    if re.search(r"cloud|overcast", c):
        return "Cloudy"
    if re.search(r"fair|clear", c):
        return "Clear"
    return "Other"


_WIND_DIR = {"calm": "CALM", "var": "VAR", "variable": "VAR",
             "north": "N", "south": "S", "east": "E", "west": "W"}


def _wind_dir(v):
    v = str(v).strip().lower()
    return _WIND_DIR.get(v, v.upper())


def _road_type(street):
    s = street.upper()
    if re.search(r"\bI-?\s?\d+", s):
        return "interstate"
    if re.search(r"\bUS[- ]?\d+|\bUS (HWY|HIGHWAY|ROUTE)\b", s):
        return "us_highway"
    if re.search(r"EXPRESSWAY|EXPY|FREEWAY|FWY|TURNPIKE|TPKE|PARKWAY|PKWY|TOLLWAY|BYPASS", s):
        return "expressway"
    if re.search(r"\b(STATE ROUTE|STATE HWY|STATE HIGHWAY|SR|SH|ROUTE|RTE|RT|HWY|HIGHWAY)\b|^[A-Z]{2}-\d+", s):
        return "state_route"
    return "local"


def _street_dir(street):
    m = re.search(r"\b([NSEW])$", street.upper().strip())
    return m.group(1) if m else "none"


def _band(s, edges, labels):
    """Fixed-edge discretisation; NaN -> 'unknown'."""
    return pd.cut(s, edges, labels=labels).astype(object).fillna("unknown")


# ------------------------------------------------------------------ the transformer
class AdvancedPreprocessor(BaseEstimator, TransformerMixin):
    """Feature engineering + cleaning + encoding, fit on training data only.

    Parameters
    ----------
    impute : fill remaining numeric NaN (training medians; Precipitation -> 0).
             Use for Linear Regression / clustering.
    scale : MinMax-scale the continuous attributes to [0, 1] (clustering, Part IV).
    encoding : "onehot" (default) or "ordinal" (integer codes, -1 = unseen) for categoricals.
    include_bands : also emit interpretable discretised versions (temperature band, ...).
    include_duration : add accident duration (End_Time - Start_Time). It is only known
        after the accident and is a consequence of it, so keep it False (default) when
        predicting severity; True is for descriptive use / clustering.
    include_distance : add log(Distance) and a zero-distance flag. Distance is also a
        consequence of the accident (same caveat); set False to exclude it from predictors.
    top_n_cities : number of most frequent training cities kept as their own category
        (all others -> "Other").
    min_flag_support : POI flags rarer than this fraction in the training data are not
        kept as individual attributes (they still count toward the aggregate flags).
    """

    def __init__(self, impute=False, scale=False, encoding="onehot", include_bands=True,
                 include_duration=False, include_distance=True, top_n_cities=20,
                 min_flag_support=0.01):
        self.impute = impute
        self.scale = scale
        self.encoding = encoding
        self.include_bands = include_bands
        self.include_duration = include_duration
        self.include_distance = include_distance
        self.top_n_cities = top_n_cities
        self.min_flag_support = min_flag_support

    # ---------------------------------------------------- stateless, rule-based part
    def _engineer(self, X):
        o = {}

        # Time -> context. `year` is deliberately dropped (reflects data coverage growth).
        start = _as_datetime(X["Start_Time"])
        hour, dow, month = start.dt.hour, start.dt.dayofweek, start.dt.month
        period = pd.cut(hour, [-1, 4, 6, 8, 15, 17, 23],
                        labels=["late_night", "early_morning", "am_rush", "midday",
                                "pm_rush", "evening"]).astype(object)
        o["hour"], o["day_of_week"], o["month"] = hour, dow, month
        o["is_weekend"] = (dow >= 5).astype(np.int8)
        o["is_late_night"] = (hour < 5).astype(np.int8)
        o["is_rush_hour"] = ((dow < 5) & period.isin(["am_rush", "pm_rush"])).astype(np.int8)
        o["day_period"] = period
        o["season"] = month.map(SEASON)

        # Duration (post-hoc; opt-in). Capped at 24 h to tame absurd values (max ~4 years).
        if self.include_duration:
            dur = ((_as_datetime(X["End_Time"]) - start).dt.total_seconds() / 60).clip(0, 1440)
            o["duration_log"] = np.log1p(dur)
            o["duration_band"] = _band(dur, [-1, 30, 120, 1441], ["short", "medium", "long"])

        # Distance: right-skewed with many exact zeros.
        if self.include_distance:
            o["log_distance"] = np.log1p(X["Distance(mi)"])
            o["zero_distance"] = (X["Distance(mi)"] == 0).astype(np.int8)

        # Weather: impossible values -> NaN, Wind_Chill dropped (r = 0.99 with Temperature).
        num = {}
        for col, key in [("Temperature(F)", "Temperature"), ("Humidity(%)", "Humidity"),
                         ("Pressure(in)", "Pressure"), ("Visibility(mi)", "Visibility"),
                         ("Wind_Speed(mph)", "Wind_Speed"), ("Precipitation(in)", "Precipitation")]:
            s = X[col]
            if col in VALID_RANGE:
                lo, hi = VALID_RANGE[col]
                s = s.where(s.between(lo, hi))
            num[key] = s
        o.update(num)
        o["precip_missing"] = num["Precipitation"].isna().astype(np.int8)

        group = _map_unique(X["Weather_Condition"], _weather_group).fillna("Unknown")
        o["weather_group"] = group
        o["wind_dir"] = _map_unique(X["Wind_Direction"], _wind_dir).fillna("Unknown")
        o["adverse_weather"] = (group.isin(ADVERSE_WEATHER) | (num["Visibility"] < 2)
                                | (num["Precipitation"] > 0)).astype(np.int8)
        if self.include_bands:
            o["temp_band"] = _band(num["Temperature"], [-np.inf, 32, 50, 80, np.inf],
                                   ["freezing", "cold", "mild", "hot"])
            o["visibility_band"] = _band(num["Visibility"], [-np.inf, 2, 8, np.inf],
                                         ["poor", "reduced", "good"])
            o["wind_band"] = _band(num["Wind_Speed"], [-np.inf, 8, 20, np.inf],
                                   ["light", "moderate", "strong"])
            o["precip_band"] = _band(num["Precipitation"], [-1, 0, 0.1, np.inf],
                                     ["none", "light", "heavy"])

        # Light: 0 = full daylight ... 4 = night by every definition.
        tw = X[TWILIGHT_COLS]
        light = (tw == "Night").sum(axis=1).astype(float)
        light[tw.isna().any(axis=1)] = np.nan
        o["light_level"] = light

        # Road context.
        o["road_type"] = _map_unique(X["Street"], _road_type).fillna("unknown")
        o["street_dir"] = _map_unique(X["Street"], _street_dir).fillna("none")
        flags = X[POI_FLAGS].astype(np.int8)
        for f in POI_FLAGS:
            o[f] = flags[f]
        o["intersection_related"] = flags[INTERSECTION_FLAGS].any(axis=1).astype(np.int8)
        o["road_feature"] = flags[ROAD_FEATURE_FLAGS].any(axis=1).astype(np.int8)
        o["n_poi"] = flags.sum(axis=1)

        # Geography.
        o["Start_Lat"], o["Start_Lng"] = X["Start_Lat"], X["Start_Lng"]
        o["region"] = X["State"].map(CENSUS_REGION).fillna("Other")
        o["timezone"] = X["Timezone"].fillna("Unknown")
        o["_city"] = X["City"]
        return pd.DataFrame(o, index=X.index)

    # ------------------------------------------------- learned-state application
    def _apply_state(self, F):
        city = F.pop("_city")
        F["city_top"] = city.where(city.isin(self.top_cities_)).fillna("Other")
        F["city_log_count"] = np.log1p(city.map(self.city_counts_).fillna(0))
        F = F.drop(columns=[f for f in POI_FLAGS if f not in self.kept_flags_])
        if self.impute:
            for c, m in self.medians_.items():
                if c in F:
                    F[c] = F[c].fillna(m)
            F["Precipitation"] = F["Precipitation"].fillna(0.0)
        return F

    def _encode(self, G):
        cat_cols = self.cat_cols_
        num = G.drop(columns=cat_cols)
        cats = {c: pd.Categorical(G[c], categories=self.cat_levels_[c]) for c in cat_cols}
        if self.encoding == "onehot":
            dummies = pd.get_dummies(pd.DataFrame(cats, index=G.index), dtype=np.float32)
            E = pd.concat([num, dummies], axis=1)
        elif self.encoding == "ordinal":
            codes = pd.DataFrame({c: v.codes for c, v in cats.items()}, index=G.index)
            E = pd.concat([num, codes], axis=1)
        else:
            raise ValueError("encoding must be 'onehot' or 'ordinal'")
        return E.astype(np.float32)

    # ---------------------------------------------------------------- sklearn API
    def _fit_transform(self, X):
        F = self._engineer(X)

        # ---- learn from the training data only ----
        self.medians_ = {c: F[c].median() for c in MEDIAN_IMPUTE if c in F}
        counts = F["_city"].value_counts()
        self.city_counts_ = counts
        self.top_cities_ = list(counts.index[:self.top_n_cities])
        self.kept_flags_ = [f for f in POI_FLAGS if F[f].mean() >= self.min_flag_support]

        G = self._apply_state(F)
        self.cat_cols_ = [c for c in G.columns if G[c].dtype == object]
        self.cat_levels_ = {c: sorted(G[c].dropna().unique()) for c in self.cat_cols_}

        E = self._encode(G)
        self.columns_ = list(E.columns)
        self.continuous_ = [c for c in CONTINUOUS if c in E.columns]
        if self.scale:
            self.scaler_ = MinMaxScaler(clip=True).fit(E[self.continuous_])
            E[self.continuous_] = self.scaler_.transform(E[self.continuous_])
        return E

    def fit(self, X, y=None):
        self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        return self._fit_transform(X)

    def transform(self, X):
        E = self._encode(self._apply_state(self._engineer(X)))
        E = E.reindex(columns=self.columns_, fill_value=0.0)
        if self.scale:
            E[self.continuous_] = self.scaler_.transform(E[self.continuous_])
        return E

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)


# --------------------------------------------------------------------- utilities
def load_filtered(path=DEFAULT_CSV, nrows=None):
    """Load the basic-preprocessed CSV; timestamps are parsed once here (not per fold)."""
    df = pd.read_csv(path, nrows=nrows)
    for c in ["Start_Time", "End_Time"]:
        df[c] = pd.to_datetime(df[c], format="mixed")
    return df


def make_targets(df):
    """Return X (no target), classification target (Severity), regression target (Severity*10, float)."""
    y_cls = df["Severity"]
    y_reg = df["Severity"].astype(float) * 10.0
    return df.drop(columns=["Severity"]), y_cls, y_reg


def build_pipeline(model, **prep_kwargs):
    """Preprocessor + model in one estimator, so cross_validate refits preprocessing per fold."""
    return Pipeline([("prep", AdvancedPreprocessor(**prep_kwargs)), ("model", model)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inspect the advanced-preprocessed feature matrix.")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--rows", type=int, default=200_000, help="rows to load (default 200k)")
    ap.add_argument("--impute", action="store_true")
    ap.add_argument("--scale", action="store_true")
    ap.add_argument("--duration", action="store_true", help="include duration attributes")
    args = ap.parse_args()

    X, _, _ = make_targets(load_filtered(args.csv, nrows=args.rows))
    Xt = AdvancedPreprocessor(impute=args.impute, scale=args.scale,
                              include_duration=args.duration).fit_transform(X)
    print(Xt.shape)
    print(list(Xt.columns))
    print(Xt.isna().sum()[Xt.isna().sum() > 0])
