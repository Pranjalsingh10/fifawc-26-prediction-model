import pandas as pd
import numpy as np

elo = pd.read_csv(
    "data/processed/elo_ratings.csv"
)

ratings = dict(
    zip(
        elo["Team"],
        elo["ELO"]
    )
)

team1 = input("Team 1: ")
team2 = input("Team 2: ")

r1 = ratings.get(team1, 1500)
r2 = ratings.get(team2, 1500)

home = 0
away = 0
draw = 0

for _ in range(10000):

    p1 = 1 / (1 + 10 ** ((r2-r1)/400))

    rnd = np.random.random()

    if rnd < p1 - 0.15:
        home += 1

    elif rnd > p1 + 0.15:
        away += 1

    else:
        draw += 1

print()
print(team1, "Win:", round(home/100,2), "%")
print("Draw:", round(draw/100,2), "%")
print(team2, "Win:", round(away/100,2), "%")