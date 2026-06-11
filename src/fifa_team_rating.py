import pandas as pd

elo = pd.read_csv(
    "data/processed/elo_ratings.csv"
)

squad = pd.read_csv(
    "data/processed/squad_strength.csv"
)

merged = pd.merge(
    squad,
    elo,
    on="Team",
    how="left"
)

merged["ELO"] = merged["ELO"].fillna(
    1500
)

merged["FinalRating"] = (
    merged["Strength"] * 0.6
    +
    merged["ELO"] * 0.4
)

merged = merged.sort_values(
    "FinalRating",
    ascending=False
)

merged.to_csv(
    "data/processed/fifa_team_ratings.csv",
    index=False
)

print(
    merged[
        ["Team","FinalRating"]
    ].head(20)
)