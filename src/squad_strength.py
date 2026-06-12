import pandas as pd

def aggregate_squad_metrics():
    # Load player details, safely skipping any raw decorative header rows
    df = pd.read_csv("data/raw/SquadLists-English.csv", skiprows=4)
    
    # Clean string numeric features
    df["CAPS"] = pd.to_numeric(df["CAPS"], errors="coerce").fillna(0)
    df["GOALS"] = pd.to_numeric(df["GOALS"], errors="coerce").fillna(0)
    df["HEIGHT (CM)"] = pd.to_numeric(df["HEIGHT (CM)"], errors="coerce").fillna(180)
    
    # Map raw, mixed-case country labels to clean team names if required
    # Group by team column to get squad metrics
    squad_strengths = df.groupby("CLUB").agg(
        Avg_Caps=("CAPS", "mean"),
        Total_International_Goals=("GOALS", "sum"),
        Avg_Height=("HEIGHT (CM)", "mean"),
        Squad_Size=("PLAYER NAME", "count")
    ).reset_index()
    
    squad_strengths.to_csv("data/processed/squad_strength.csv", index=False)
    print("✅ Created squad_strength.csv metrics!")

if __name__ == "__main__":
    aggregate_squad_metrics()