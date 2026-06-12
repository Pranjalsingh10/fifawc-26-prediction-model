import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model():
    print("🚀 Starting Neutral-Venue Aware Machine Learning Training...")
    
    feature_path = "data/processed/match_training_features.csv"
    if not os.path.exists(feature_path):
        print(f"❌ Error: Cannot find {feature_path}.")
        return
        
    df = pd.read_csv(feature_path)
    
    X = df[[
        "Home_Elo", "Away_Elo", "Elo_Difference",
        "Home_WinRate_Post2022", "Away_WinRate_Post2022", "WinRate_Difference",
        "Home_WC_Titles", "Away_WC_Titles", "Title_Difference",
        "Home_Squad_Form_Sharpness", "Away_Squad_Form_Sharpness", "Player_Form_Difference",
        "Neutral_Venue"
    ]]
    y = df["Match_Result"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"\n📊 Overall Training Validation Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    
    # Explicitly clear out old saved binaries to override system locks
    os.makedirs("models", exist_ok=True)
    out_path = "models/match_predictor.pkl"
    if os.path.exists(out_path):
        os.remove(out_path)
        
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
        
    print("✅ Fresh match_predictor.pkl successfully written to disk!")

if __name__ == "__main__":
    train_model()