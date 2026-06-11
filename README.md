# 🏆 FIFA World Cup 2026 Prediction Model

An end-to-end Machine Learning project designed to predict match outcomes, tournament trajectories, player performances, and squad strengths for the upcoming **expanded 48-team 2026 FIFA World Cup**. Utilizing historical match datasets, Elo ratings, and custom feature engineering, this repository hosts the data preprocessing pipeline, predictive modeling scripts, and an interactive deployment app.

---

## 📂 Project Structure

The repository is organized cleanly into modular directories following standard data science practices:

```text
fifawc/
├── data/
│   ├── raw/                  # Original datasets
│   │   ├── WorldCups.csv         # Overview data per tournament (1930-2022)
│   │   ├── WorldCupMatches.csv   # Historical match-by-match metrics (1930-2022)
│   │   ├── WorldCupPlayers.csv   # Historic tournament rosters & appearances
│   │   └── SquadLists-English.csv# Modern team/player profile records
│   └── processed/            # Cleaned data engineered for model consumption
│       ├── clean_matches.csv
│       ├── clean_squads.csv
│       ├── elo_ratings.csv
│       └── team_features.csv
├── models/
│   └── match_predictor.pkl   # Serialized, trained ML model ready for inference
├── src/                      # Source codebase
│   ├── app.py                # Core application dashboard (Streamlit / Flask)
│   ├── data_preprocessing.py # Script for cleaning raw CSV inputs
│   ├── feature_engineering.py# Generates custom predictors and rolling statistics
│   ├── elo_rating.py         # Implements dynamic Elo calculation pipelines
│   ├── squad_strength.py     # Quantifies historical and current squad power indexes
│   ├── train_match_predictor.py # Trains and evaluates the ML prediction models
│   ├── predict_match.py      # Independent inference helper for single-match simulation
│   └── penalty_predictor.py  # Special simulator for penalty shootout probabilities
├── requirements.txt          # Third-party dependencies and Python libraries
└── README.md                 # Project documentation
