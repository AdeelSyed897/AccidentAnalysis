"""Part I.2 - Basic data preprocessing for the US Accidents dataset.

Streams the raw 3 GB CSV in chunks (so it never has to fit in RAM) and:
  1. keeps only rows with AT MOST ONE missing value (counted over all 46 original columns),
  2. drops attributes that cannot contribute to any later part of the project.

Output: US_Accidents_filtered.csv (input for Parts II-V).
Usage:  python preprocess.py [path/to/US_Accidents_March23.csv]
"""
import sys

import pandas as pd

RAW_CSV = sys.argv[1] if len(sys.argv) > 1 else "US_Accidents_March23.csv"
OUT_CSV = "US_Accidents_filtered.csv"
CHUNK_ROWS = 500_000
MAX_MISSING = 1

# Attributes removed after the row filter (rationale in report.md).
DROP_COLS = [
    "ID",                 # row identifier
    "Source",             # data-provider tag, not a property of the accident
    "Description",        # free text, unique per row
    "Country",            # constant ("US")
    "Airport_Code",       # id of the nearest weather station
    "Weather_Timestamp",  # time of the weather observation, not of the accident
    "Zipcode",            # very high cardinality; redundant with State/City/County/Lat/Lng
    "End_Lat",            # redundant with Start_Lat/Start_Lng + Distance(mi)
    "End_Lng",
    "Turning_Loop",       # constant (False in every kept row)
]



def process_weather(df):
    weather = df["Weather_Condition"].fillna("Unknown").str.lower()

    df["weather_rain"] = weather.str.contains(
        "rain|drizzle|shower", regex=True
    ).astype(int)

    df["weather_snow"] = weather.str.contains(
        "snow", regex=True
    ).astype(int)

    df["weather_fog"] = weather.str.contains(
        "fog|mist", regex=True
    ).astype(int)

    df["weather_thunderstorm"] = weather.str.contains(
        "thunder|t-storm", regex=True
    ).astype(int)

    df["weather_wind"] = weather.str.contains(
        "windy", regex=True
    ).astype(int)

    df["weather_hail"] = weather.str.contains(
        "hail", regex=True
    ).astype(int)

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
    df["Start_Time"] = pd.to_datetime(
        df["Start_Time"],
        format="mixed"
    )
    df["start_hour"] = df["Start_Time"].dt.hour
    df["start_dayofweek"] = df["Start_Time"].dt.dayofweek
    df["start_month"] = df["Start_Time"].dt.month
    df["start_year"] = df["Start_Time"].dt.year

    df["End_Time"] = pd.to_datetime(
        df["End_Time"],
        format="mixed"
    )

    df["duration_minutes"] = (
        df["End_Time"] - pd.to_datetime(df["Start_Time"])
    ).dt.total_seconds() / 60


def process_weather(df):
    weather = df["Weather_Condition"].fillna("Unknown").str.lower()

    df["weather_rain"] = weather.str.contains(
        "rain|drizzle|shower", regex=True
    ).astype(int)

    df["weather_snow"] = weather.str.contains(
        "snow", regex=True
    ).astype(int)

    df["weather_fog"] = weather.str.contains(
        "fog|mist", regex=True
    ).astype(int)

    df["weather_thunderstorm"] = weather.str.contains(
        "thunder|t-storm", regex=True
    ).astype(int)

    df["weather_wind"] = weather.str.contains(
        "windy", regex=True
    ).astype(int)

    df["weather_hail"] = weather.str.contains(
        "hail", regex=True
    ).astype(int)

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
    df["Start_Time"] = pd.to_datetime(
        df["Start_Time"],
        format="mixed"
    )
    df["start_hour"] = df["Start_Time"].dt.hour
    df["start_dayofweek"] = df["Start_Time"].dt.dayofweek
    df["start_month"] = df["Start_Time"].dt.month
    df["start_year"] = df["Start_Time"].dt.year

    df["End_Time"] = pd.to_datetime(
        df["End_Time"],
        format="mixed"
    )

    df["duration_minutes"] = (
        df["End_Time"] - pd.to_datetime(df["Start_Time"])
    ).dt.total_seconds() / 60


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





def main():
    total_in = 0
    total_out = 0
    missing_hist = {}            # missing-value count -> number of rows (raw data)
    missing_kept = None          # per-column missing counts among kept rows
    country_values = set()
    first = True

    for chunk in pd.read_csv(RAW_CSV, chunksize=CHUNK_ROWS):
        n_missing = chunk.isna().sum(axis=1)
        for k, v in n_missing.value_counts().items():
            missing_hist[k] = missing_hist.get(k, 0) + v
        country_values.update(chunk["Country"].dropna().unique())

        kept = chunk[n_missing <= MAX_MISSING].drop(columns=DROP_COLS)
        col_missing = kept.isna().sum()
        missing_kept = col_missing if missing_kept is None else missing_kept + col_missing

        kept.to_csv(OUT_CSV, mode="w" if first else "a", header=first, index=False)
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




if __name__ == "__main__":
    main()
