import pandas as pd

print("Loading squads...")

df = pd.read_csv(
    "data/processed/clean_squads.csv"
)

teams = []

for team in df["Team"].unique():

    squad = df[df["Team"] == team]

    attack = (
        squad["Goals"].mean()
        + len(
            squad[squad["Pos"] == "FW"]
        ) * 2
    )

    defense = (
        len(
            squad[squad["Pos"] == "DF"]
        ) * 5
        +
        len(
            squad[squad["Pos"] == "GK"]
        ) * 8
    )

    experience = squad["Caps"].mean()

    avg_height = squad["Height"].mean()

    strength = (
        attack * 0.4
        + defense * 0.2
        + experience * 0.3
        + avg_height * 0.1
    )

    teams.append({
        "Team": team,
        "Attack": round(attack, 2),
        "Defense": round(defense, 2),
        "Experience": round(experience, 2),
        "AvgHeight": round(avg_height, 2),
        "Strength": round(strength, 2)
    })

strength_df = pd.DataFrame(teams)

strength_df = strength_df.sort_values(
    "Strength",
    ascending=False
)

strength_df.to_csv(
    "data/processed/squad_strength.csv",
    index=False
)

print("\nTop 20 Teams:\n")

print(
    strength_df.head(20)
)