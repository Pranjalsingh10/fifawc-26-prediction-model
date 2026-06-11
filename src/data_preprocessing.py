import pandas as pd

print("Loading dataset...")

matches = pd.read_csv(
    "data/raw/WorldCupMatches.csv"
)

matches = matches.dropna(
    subset=[
        "Home Team Name",
        "Away Team Name",
        "Home Team Goals",
        "Away Team Goals"
    ]
)

matches["Home Team Goals"] = pd.to_numeric(
    matches["Home Team Goals"],
    errors="coerce"
)

matches["Away Team Goals"] = pd.to_numeric(
    matches["Away Team Goals"],
    errors="coerce"
)

matches = matches.dropna()

matches.to_csv(
    "data/processed/clean_matches.csv",
    index=False
)

print("Rows:", len(matches))
print("Saved clean_matches.csv")