import pickle
import numpy as np
import pandas as pd

def get_user_input_and_predict():
    # 1. Load the serialized model
    try:
        with open("models/match_predictor.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("❌ Error: models/match_predictor.pkl not found. Please run train_match_predictor.py first.")
        return

    print("=== ⚽ FIFA World Cup 2026 Match Simulator ===")
    
    # 2. Get dynamic inputs from the terminal window
    home_team = input("Enter Home Team Name (e.g., Argentina): ").strip()
    away_team = input("Enter Away Team Name (e.g., France): ").strip()
    
    try:
        home_elo = float(input(f"Enter {home_team} Elo Rating (e.g., 1950): "))
        away_elo = float(input(f"Enter {away_team} Elo Rating (e.g., 1920): "))
    except ValueError:
        print("❌ Error: Elo ratings must be numbers.")
        return
        
    # 3. Format features matching the model training structure
    elo_diff = home_elo - away_elo
    input_data = pd.DataFrame([{
        "Home_Elo_Before": home_elo,
        "Away_Elo_Before": away_elo,
        "Elo_Difference": elo_diff
    }])
    
    # 4. Predict probabilities (Classes: [0: Home Win, 1: Away Win, 2: Draw])
    probabilities = model.predict_proba(input_data)[0]
    
    print(f"\n🔮 --- 2026 World Cup Simulation: {home_team} vs {away_team} ---")
    print(f"📊 {home_team} Elo: {home_elo} | {away_team} Elo: {away_elo} (Diff: {elo_diff:+})")
    print("-" * 55)
    print(f"🏠 {home_team} Win Probability: {probabilities[0]:.2%}")
    print(f"🤝 Draw Probability:     {probabilities[2]:.2%}")
    print(f"🚀 {away_team} Win Probability: {probabilities[1]:.2%}")

if __name__ == "__main__":
    get_user_input_and_predict()