import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

matches = pd.read_csv(
    "data/processed/clean_matches.csv"
)

matches["result"] = 0

matches.loc[
    matches["Home Team Goals"] >
    matches["Away Team Goals"],
    "result"
] = 1

matches.loc[
    matches["Home Team Goals"] <
    matches["Away Team Goals"],
    "result"
] = 2

X = matches[
    [
        "Home Team Goals",
        "Away Team Goals"
    ]
]

y = matches["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier()

model.fit(X_train, y_train)

pred = model.predict(X_test)

print(
    "Accuracy:",
    accuracy_score(y_test, pred)
)

joblib.dump(
    model,
    "models/match_predictor.pkl"
)

print("Model saved")