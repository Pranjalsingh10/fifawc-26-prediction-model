import pandas as pd

matches = pd.read_csv(
    "data/processed/clean_matches.csv"
)

teams = {}

for _, row in matches.iterrows():

    home = row["Home Team Name"]
    away = row["Away Team Name"]

    hg = row["Home Team Goals"]
    ag = row["Away Team Goals"]

    if home not in teams:
        teams[home] = {
            "games": 0,
            "gf": 0,
            "ga": 0,
            "wins": 0
        }

    if away not in teams:
        teams[away] = {
            "games": 0,
            "gf": 0,
            "ga": 0,
            "wins": 0
        }

    teams[home]["games"] += 1
    teams[away]["games"] += 1

    teams[home]["gf"] += hg
    teams[home]["ga"] += ag

    teams[away]["gf"] += ag
    teams[away]["ga"] += hg

    if hg > ag:
        teams[home]["wins"] += 1

    elif ag > hg:
        teams[away]["wins"] += 1

rows = []

for team, stats in teams.items():

    rows.append([
        team,
        stats["games"],
        stats["gf"] / stats["games"],
        stats["ga"] / stats["games"],
        stats["wins"] / stats["games"]
    ])

features = pd.DataFrame(
    rows,
    columns=[
        "team",
        "games",
        "avg_goals_for",
        "avg_goals_against",
        "win_rate"
    ]
)

features.to_csv(
    "data/processed/team_features.csv",
    index=False
)

print(features.head())