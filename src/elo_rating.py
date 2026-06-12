import pandas as pd
import numpy as np

def calculate_elo():
    df = pd.read_csv("data/processed/clean_matches.csv")
    df = df.sort_values(by=["Year", "Datetime"]).reset_index(drop=True)
    
    # Initialize all teams at a baseline of 1500
    elo_dict = {}
    
    # Track elo history to append as features later
    home_elos = []
    away_elos = []
    
    K = 32 # Maximum points shifted per match
    
    for idx, row in df.iterrows():
        home = row["Home Team Name"]
        away = row["Away Team Name"]
        
        # Assign baseline if team is new to the dataset
        if home not in elo_dict: elo_dict[home] = 1500.0
        if away not in elo_dict: elo_dict[away] = 1500.0
        
        h_elo = elo_dict[home]
        a_elo = elo_dict[away]
        
        home_elos.append(h_elo)
        away_elos.append(a_elo)
        
        # Expected outcomes based on Elo difference formula
        expected_home = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        expected_away = 1.0 - expected_home
        
        # Actual outcomes based on goals scored
        if row["Home Team Goals"] > row["Away Team Goals"]:
            actual_home, actual_away = 1.0, 0.0
        elif row["Home Team Goals"] < row["Away Team Goals"]:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5
            
        # Update ratings in dictionary
        elo_dict[home] += K * (actual_home - expected_home)
        elo_dict[away] += K * (actual_away - expected_away)
        
    df["Home_Elo_Before"] = home_elos
    df["Away_Elo_Before"] = away_elos
    
    df.to_csv("data/processed/elo_ratings.csv", index=False)
    print("✅ Created elo_ratings.csv with chronological strengths!")

if __name__ == "__main__":
    calculate_elo()