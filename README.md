# ⚽ FIFA World Cup 2026 Machine Learning Predictive Engine

An end-to-end predictive analytics platform and machine learning pipeline built to simulate and forecast international football matches for the **2026 FIFA World Cup**. The core engine trains on modern international football data using an ensemble **Random Forest Classifier**, incorporates dynamic player-level metrics tracking from live Group Stage performances, handles structural home-field advantage anomalies natively via contextual venue masking, and serves interactive real-time inferences through a Streamlit UI dashboard.

---

## 🚀 Live Production URL
🔗 **[Deploy Link]**: Your live cloud URL will display here (e.g., `https://your-app-name.streamlit.app/`)

---

## 📑 Core Architecture & Features Overview

* **Live Tournament Power Indexing:** Dynamically reads and processes live 2026 Group Stage records (`matches.csv`) to compute form tracking metrics. Rather than relying on static pre-tournament baselines, the engine scales team Elo coefficients dynamically based on real-world performance points, goal differences, and shots on target.
* **Micro Player-Form Layer Integration:** Simulates and structures a tournament roster dataset of exactly 1,248 active elite players representing all 48 participating nations. Team form variables evaluate the dynamic moving average of the selected 26-man roster sharpness indexes.
* **Neutral Venue Mitigation Masking:** Solves the traditional sports analytics limitation of left-side menu entry bias. Implementing a binary `Neutral_Venue` feature branch forces the underlying decision trees to discount arbitrary host bonuses during tournament neutral-site calculations, while automatically retaining home-crowd advantage weightings if hosts (USA, Mexico, or Canada) play in their native stadia.
* **Dual-Format Contextual Rule Simulation:** Incorporates a frontend configuration layer allowing users to switch execution modes between regular group-stage round-robins (where standard tactical 90-minute draws are valid results) and single-elimination knockout tournament rounds (where regulations ties are mathematically split to predict penalty shootout or extra time advancement).

---

## 🛠️ System Repository Architecture

```text
fifawc/                              # Project root directory
│
├── data/
│   ├── raw/
│   │   └── WorldCupMatches.csv      # Historical match archives
│   │
│   └── processed/
│       ├── match_training_features.csv  # 14-feature compiled training tensor 
│       ├── squad_advanced_profile.csv   # Dynamic macro team metrics registry
│       └── web_player_performance.csv   # Structured micro 26-man squad rosters
│
├── models/
│   └── match_predictor.pkl          # Compressed Random Forest model binary object
│
├── src/
│   ├── app.py                       # Interactive Streamlit UI server dashboard
│   ├── expand_team_profiles.py      # Macro data aggregator from group stage
│   ├── generate_all_players.py      # Micro player form generator
│   ├── generate_match_features.py   # Historical dataset merging pipeline
│   └── train_match_predictor.py     # Stratified scikit-learn training script
│
├── matches.csv                      # Live 2026 Group Stage statistics dataset
├── requirements.txt                 # Deployment virtual environment dependencies
└── README.md                        # Documentation handbook guide
