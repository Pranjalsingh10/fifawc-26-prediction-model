import pandas as pd

print("Loading dataset...")

# 1. Added encoding to prevent Unicode string crashes
matches = pd.read_csv(
    "data/raw/WorldCupMatches.csv", 
    encoding="latin1"
)

# 2. Drop the thousands of completely blank ghost trailing rows first
matches = matches.dropna(subset=["Year"])

# 3. Clean and isolate target feature columns
target_cols = [
    "Home Team Name",
    "Away Team Name",
    "Home Team Goals",
    "Away Team Goals"
]

matches = matches.dropna(subset=target_cols)

matches["Home Team Goals"] = pd.to_numeric(
    matches["Home Team Goals"],
    errors="coerce"
)

matches["Away Team Goals"] = pd.to_numeric(
    matches["Away Team Goals"],
    errors="coerce"
)

# 4. FIX: Only drop rows where critical predictive numeric elements are missing
matches = matches.dropna(subset=["Home Team Goals", "Away Team Goals"])

# 5. Clean up indices and types for standard data handling
matches["Year"] = matches["Year"].astype(int)
matches["MatchID"] = matches["MatchID"].astype(int)

# Fill missing attendance with the median instead of dropping the match row
matches["Attendance"] = matches["Attendance"].fillna(matches["Attendance"].median())

# Save back out to the target file directory
matches.to_csv(
    "data/processed/clean_matches.csv",
    index=False
)

print("Rows:", len(matches))
print("Saved clean_matches.csv")