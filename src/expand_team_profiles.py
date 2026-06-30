import pandas as pd
import numpy as np
import os

def expand_profiles():
    print("📈 Rebuilding profiles dynamically using 2026 Group Stage performance data...")
    
    match_path = "matches.csv"
    if not os.path.exists(match_path):
        print("❌ Error: matches.csv file not found in root directory!")
        return
        
    df = pd.read_csv(match_path)
    
    # Extract the exact 48 teams active in your group stage dataset
    all_48_teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))
    
    profile_records = []
    
    for team in all_48_teams:
        home_m = df[df['home_team'] == team]
        away_m = df[df['away_team'] == team]
        played = len(home_m) + len(away_m)
        
        wins, draws, losses = 0, 0, 0
        for _, row in home_m.iterrows():
            if row['home_score'] > row['away_score']: wins += 1
            elif row['home_score'] == row['away_score']: draws += 1
            else: losses += 1
        for _, row in away_m.iterrows():
            if row['away_score'] > row['home_score']: wins += 1
            elif row['away_score'] == row['home_score']: draws += 1
            else: losses += 1
            
        pts = wins * 3 + draws * 1
        gf = home_m['home_score'].sum() + away_m['away_score'].sum()
        ga = home_m['away_score'].sum() + away_m['home_score'].sum()
        gd = gf - ga
        
        win_rate = wins / played if played > 0 else 0.4
        
        # Core base international tier mapping
        base_elo = 1750.0
        if team in ['France', 'Argentina', 'Brazil', 'Spain', 'England', 'Portugal', 'Germany', 'Netherlands']:
            base_elo = 2000.0
        elif team in ['Mexico', 'Colombia', 'Morocco', 'Belgium', 'Switzerland', 'United States', 'Uruguay', 'Croatia', 'Japan', 'Korea Republic']:
            base_elo = 1880.0
            
        # Live Tournament Adjustment: Earned points increase Elo (+20 per point), losses decrease it
        latest_elo = base_elo + (pts * 20.0) - (losses * 30.0)
        
        # Historical World Cup Pedigree
        titles = 0
        if team == "Brazil": titles = 5
        if team == "Argentina": titles = 3
        if team in ["France", "Germany", "Uruguay"]: titles = 2
        if team in ["England", "Spain"]: titles = 1

        profile_records.append({
            "National_Team": team,
            "Latest_Elo_Rating": latest_elo,
            "Win_Rate_Post_2022": round(win_rate, 2),
            "World_Cup_Titles": titles
        })
        
    df_expanded = pd.DataFrame(profile_records)
    os.makedirs("data/processed", exist_ok=True)
    df_expanded.to_csv("data/processed/squad_advanced_profile.csv", index=False)
    print(f"✅ Success! Profile registry dynamically compiled for all {len(df_expanded)} teams based on Group Stage form.")

if __name__ == "__main__":
    expand_profiles()