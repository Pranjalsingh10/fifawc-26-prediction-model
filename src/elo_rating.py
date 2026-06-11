import pandas as pd

matches = pd.read_csv(
    "data/processed/clean_matches.csv"
)

ratings = {}

K = 20

def expected(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))

for _, row in matches.iterrows():

    home = row["Home Team Name"]
    away = row["Away Team Name"]

    hg = row["Home Team Goals"]
    ag = row["Away Team Goals"]

    ratings.setdefault(home, 1500)
    ratings.setdefault(away, 1500)

    e_home = expected(
        ratings[home],
        ratings[away]
    )

    e_away = expected(
        ratings[away],
        ratings[home]
    )

    if hg > ag:
        s_home = 1
        s_away = 0

    elif hg < ag:
        s_home = 0
        s_away = 1

    else:
        s_home = 0.5
        s_away = 0.5

    ratings[home] += K * (s_home - e_home)
    ratings[away] += K * (s_away - e_away)

elo_df = pd.DataFrame(
    ratings.items(),
    columns=["Team", "ELO"]
)

elo_df = elo_df.sort_values(
    "ELO",
    ascending=False
)

elo_df.to_csv(
    "data/processed/elo_ratings.csv",
    index=False
)

print(elo_df.head(20))