import pandas as pd
import csv
import re

players = []

current_team = None

with open(
    "data/raw/SquadLists-English.csv",
    encoding="utf-8",
    errors="ignore"
) as f:

    reader = csv.reader(f)

    for row in reader:

        # TEAM HEADER
        if len(row) > 9:

            team_cell = row[9].strip()

            match = re.match(
                r"(.+)\s+\([A-Z]{3}\)",
                team_cell
            )

            if match:

                current_team = match.group(1).strip()

                continue

        # PLAYER ROW
        if len(row) >= 16 and row[0].isdigit():

            try:

                players.append({

                    "Team": current_team,

                    "Pos": row[1].strip(),

                    "Name": row[2].strip(),

                    "Height": int(row[13]),

                    "Caps": int(row[14]),

                    "Goals": int(row[15])

                })

            except:
                pass

df = pd.DataFrame(players)

df.to_csv(
    "data/processed/clean_squads.csv",
    index=False
)

print("\nPlayers:", len(df))

print("Teams:", df["Team"].nunique())

print("\nFirst rows:\n")

print(df.head())