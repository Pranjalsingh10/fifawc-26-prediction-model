import pandas as pd
import numpy as np

def build_advanced_matrix():
    # 1. Load historical match Elos
    elo_df = pd.read_csv("data/processed/elo_ratings.csv")
    
    # 2. Load and build modern team metrics from your Squad list
    squad_df = pd.read_csv("data/raw/SquadLists-English.csv", skiprows=4)
    squad_df["CAPS"] = pd.to_numeric(squad_df["CAPS"], errors="coerce").fillna(0)
    squad_df["GOALS"] = pd.to_numeric(squad_df["GOALS"], errors="coerce").fillna(0)
    
    # Aggregate to find squad experience and active goal form
    # Note: Ensure your 'CLUB' column matches country names, or map it accordingly
    team_stats = squad_df.groupby("CLUB").agg(
        Squad_Experience=("CAPS", "mean"),
        Squad_Current_Form=("GOALS", "sum")
    ).reset_index()
    
    # Standardize country names to match your match database mapping
    team_stats = team_stats.rename(columns={"CLUB": "Team_Name"})
    
    # 3. Merge modern features into historical base data
    # Home Team merge
    df_merged = pd.merge(elo_df, team_stats, left_on="Home Team Name", right_on="Team_Name", how="left")
    df_merged = df_merged.rename(columns={"Squad_Experience": "Home_Squad_Exp", "Squad_Current_Form": "Home_Squad_Form"}).drop(columns=["Team_Name"])
    
    # Away Team merge
    df_merged = pd.merge(df_merged, team_stats, left_on="Away Team Name", right_on="Team_Name", how="left")
    df_merged = df_merged.rename(columns={"Squad_Experience": "Away_Squad_Exp", "Squad_Current_Form": "Away_Squad_Form"}).drop(columns=["Team_Name"])
    
    # Fill gaps for old historical teams that don't have 2026 squads with dataset medians
    df_merged["Home_Squad_Exp"] = df_merged["Home_Squad_Exp"].fillna(df_merged["Home_Squad_Exp"].median())
    df_merged["Away_Squad_Exp"] = df_merged["Away_Squad_Exp"].fillna(df_merged["Away_Squad_Exp"].median())
    df_merged["Home_Squad_Form"] = df_merged["Home_Squad_Form"].fillna(df_merged["Home_Squad_Form"].median())
    df_merged["Away_Squad_Form"] = df_merged["Away_Squad_Form"].fillna(df_merged["Away_Squad_Form"].median())
    
    # 4. Define True 2026 Host Status (Neutralizing the historical home bias artifact)
    hosts = ["USA", "Mexico", "Canada", "United States", "Mexico ", "Canada "]
    df_merged["Home_Is_True_Host"] = np.where(df_merged["Home Team Name"].isin(hosts), 1, 0)
    df_merged["Away_Is_True_Host"] = np.where(df_merged["Away Team Name"].isin(hosts), 1, 0)
    
    # Define match labels
    conditions = [
        (df_merged["Home Team Goals"] > df_merged["Away Team Goals"]),
        (df_merged["Home Team Goals"] < df_merged["Away Team Goals"])
    ]
    choices = [1, 0]
    df_merged["Match_Result"] = np.select(conditions, choices, default=2)
    
    # Isolate final multidimensional feature sets
    final_features = df_merged[[
        "Home_Elo_Before", "Away_Elo_Before",
        "Home_Squad_Exp", "Away_Squad_Exp",
        "Home_Squad_Form", "Away_Squad_Form",
        "Home_Is_True_Host", "Away_Is_True_Host",
        "Match_Result"
    ]]
    
    final_features.to_csv("data/processed/team_features.csv", index=False)
    print("🎯 Advanced multidimensional features matrix saved to team_features.csv!")

if __name__ == "__main__":
    build_advanced_matrix()