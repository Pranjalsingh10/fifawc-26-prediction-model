import pandas as pd
import numpy as np
import os

def generate_match_features():
    print("⏳ Loading web profiles, matches, and player matrices with neutral venue indicators...")
    
    profile_path = "data/processed/squad_advanced_profile.csv"
    player_path = "data/processed/web_player_performance.csv"
    
    if not os.path.exists(profile_path) or not os.path.exists(player_path):
        print("❌ Error: Missing master files.")
        return
        
    df_profiles = pd.read_csv(profile_path)
    df_players = pd.read_csv(player_path)
    
    print("🧠 Aggregating player form indexes into team layers...")
    squad_form = df_players.groupby("National_Team")["Recent_Form_Sharpness_Rating"].mean().reset_index()
    squad_form = squad_form.rename(columns={"National_Team": "National_Team", "Recent_Form_Sharpness_Rating": "Squad_Form_Sharpness"})
    
    df_profiles = pd.merge(df_profiles, squad_form, on="National_Team", how="inner")
    
    matches_url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    df_matches = pd.read_csv(matches_url)
    df_matches["date"] = pd.to_datetime(df_matches["date"])
    df_matches = df_matches[df_matches["date"].dt.year >= 2000].copy()
    
    name_cleaner = {"United States": "USA", "South Korea": "South Korea", "Iran": "IR Iran"}
    df_matches["home_team"] = df_matches["home_team"].replace(name_cleaner)
    df_matches["away_team"] = df_matches["away_team"].replace(name_cleaner)
    
    df_matches["Neutral_Venue"] = df_matches["neutral"].astype(int)
    
    # Explicitly mapping column targets to the exact long training strings
    df_features = pd.merge(df_matches, df_profiles, left_on="home_team", right_on="National_Team", how="inner").rename(columns={
        "Latest_Elo_Rating": "Home_Elo", 
        "Win_Rate_Post_2022": "Home_WinRate_Post2022",
        "World_Cup_Titles": "Home_WC_Titles", 
        "Squad_Form_Sharpness": "Home_Squad_Form_Sharpness"
    }).drop(columns=["National_Team"], errors='ignore')
    
    df_features = pd.merge(df_features, df_profiles, left_on="away_team", right_on="National_Team", how="inner").rename(columns={
        "Latest_Elo_Rating": "Away_Elo", 
        "Win_Rate_Post_2022": "Away_WinRate_Post2022",
        "World_Cup_Titles": "Away_WC_Titles", 
        "Squad_Form_Sharpness": "Away_Squad_Form_Sharpness"
    }).drop(columns=["National_Team"], errors='ignore')
    
    df_features["Elo_Difference"] = df_features["Home_Elo"] - df_features["Away_Elo"]
    df_features["WinRate_Difference"] = df_features["Home_WinRate_Post2022"] - df_features["Away_WinRate_Post2022"]
    df_features["Title_Difference"] = df_features["Home_WC_Titles"] - df_features["Away_WC_Titles"]
    df_features["Player_Form_Difference"] = df_features["Home_Squad_Form_Sharpness"] - df_features["Away_Squad_Form_Sharpness"]
    
    conditions = [
        (df_features["home_score"] > df_features["away_score"]),
        (df_features["home_score"] < df_features["away_score"])
    ]
    df_features["Match_Result"] = np.select(conditions, [1, 0], default=2)
    
    ml_feature_columns = [
        "Home_Elo", "Away_Elo", "Elo_Difference",
        "Home_WinRate_Post2022", "Away_WinRate_Post2022", "WinRate_Difference",
        "Home_WC_Titles", "Away_WC_Titles", "Title_Difference",
        "Home_Squad_Form_Sharpness", "Away_Squad_Form_Sharpness", "Player_Form_Difference",
        "Neutral_Venue", "Match_Result"
    ]
    
    df_final = df_features[ml_feature_columns].dropna()
    df_final.to_csv("data/processed/match_training_features.csv", index=False)
    print(f"✅ Success! Feature matrix generated smoothly ({len(df_final)} rows).")

if __name__ == "__main__":
    generate_match_features()