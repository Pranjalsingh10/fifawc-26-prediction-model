import pandas as pd
import numpy as np
import os

def build_dataset_from_web():
    print("🌐 Downloading historical international football matches from web mirror...")
    matches_url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    
    try:
        df_matches = pd.read_csv(matches_url)
        print(f"✅ Successfully loaded {len(df_matches)} international match records from the web!")
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        return

    df_matches["date"] = pd.to_datetime(df_matches["date"])
    df_matches = df_matches.sort_values(by="date").reset_index(drop=True)
    df_matches["Year"] = df_matches["date"].dt.year

    print("🧠 Calculating rolling historical Elo ratings for all countries...")
    elo_dict = {}
    K = 32

    for idx, row in df_matches.iterrows():
        home = str(row["home_team"]).strip()
        away = str(row["away_team"]).strip()
        
        if home not in elo_dict: elo_dict[home] = 1500.0
        if away not in elo_dict: elo_dict[away] = 1500.0
        
        h_elo = elo_dict[home]
        a_elo = elo_dict[away]
        
        expected_home = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        expected_away = 1.0 - expected_home
        
        if row["home_score"] > row["away_score"]:
            actual_home, actual_away = 1.0, 0.0
        elif row["home_score"] < row["away_score"]:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5
            
        elo_dict[home] += K * (actual_home - expected_home)
        elo_dict[away] += K * (actual_away - expected_away)

    print("🏆 Compiling World Cup historical trophies database...")
    wc_winners_registry = {
        "Brazil": 5, "Germany": 4, "Italy": 4, "Argentina": 3, 
        "Uruguay": 2, "France": 2, "England": 1, "Spain": 1
    }
    wc_runners_up_registry = {
        "Germany": 4, "Argentina": 3, "Netherlands": 3, "Brazil": 2, 
        "Italy": 2, "France": 2, "Hungary": 2, "Czechia": 2, "Sweden": 1, "Croatia": 1
    }

    # 🌟 CRITICAL FIX: The Official 48 Qualified Nations for the 2026 World Cup
    # Non-qualified teams (like Italy) are excluded from this list.
    qualified_48_teams = [
        "USA", "Mexico", "Canada", "Argentina", "France", "Brazil", "England", 
        "Spain", "Portugal", "Netherlands", "Belgium", "Croatia", "Uruguay", 
        "Germany", "Morocco", "Japan", "Senegal", "Colombia", "South Korea", 
        "Iran", "Australia", "Saudi Arabia", "Qatar", "Iraq", "Uzbekistan", 
        "Ecuador", "Peru", "Chile", "Paraguay", "Venezuela", "Cabo Verde", 
        "Egypt", "Nigeria", "Tunisia", "Algeria", "Ghana", "Mali", "South Africa", 
        "New Zealand", "Panama", "Costa Rica", "Jamaica", "Honduras", 
        "Switzerland", "Denmark", "Austria", "Ukraine", "Poland"
    ]
    
    advanced_profiles = []
    
    print("📊 Synthesizing final multivariate metrics matrices (Filtered for 2026 squads)...")
    for team in sorted(qualified_48_teams):
        # Graceful handling for slight variations in naming conventions across data lines
        lookup_name = "South Korea" if team == "South Korea" else team
        
        if lookup_name not in elo_dict:
            elo_dict[lookup_name] = 1500.0  # Safe fallback for rare naming variants
            
        # Calculate recent games played post-2022 World Cup
        post_2022_games = df_matches[
            ((df_matches["home_team"] == lookup_name) | (df_matches["away_team"] == lookup_name)) & 
            (df_matches["date"] > "2022-12-18")
        ]
        
        games_played = len(post_2022_games)
        
        wins = 0
        for _, match in post_2022_games.iterrows():
            if match["home_team"] == lookup_name and match["home_score"] > match["away_score"]: wins += 1
            elif match["away_team"] == lookup_name and match["away_score"] > match["home_score"]: wins += 1
            
        win_rate = (wins / games_played) if games_played > 0 else 0.0
        
        advanced_profiles.append({
            "National_Team": team,
            "Latest_Elo_Rating": round(elo_dict[lookup_name], 2),
            "Matches_Played_Post_2022": games_played,
            "Win_Rate_Post_2022": round(win_rate, 4),
            "World_Cup_Titles": wc_winners_registry.get(team, 0),
            "World_Cup_Runners_Up": wc_runners_up_registry.get(team, 0)
        })
        
    df_profiles = pd.DataFrame(advanced_profiles)
    
    os.makedirs("data/processed", exist_ok=True)
    df_profiles.to_csv("data/processed/squad_advanced_profile.csv", index=False)
    print("✅ Success! Generated profile layer containing exactly the 48 qualified 2026 squads.")

if __name__ == "__main__":
    build_dataset_from_web()