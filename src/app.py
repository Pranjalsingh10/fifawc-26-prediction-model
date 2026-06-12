import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="FIFA 2026 Ultimate Engine", page_icon="⚽", layout="centered")

st.title("🏆 FIFA World Cup 2026 Match Predictor")
st.write("Predict matches using a multi-variable Random Forest model aligned with real-world neutral venue configurations.")
st.markdown("---")

# 1. Exact, rigid feature ordering matching scikit-learn's training shape
FEATURE_ORDER = [
    "Home_Elo", "Away_Elo", "Elo_Difference",
    "Home_WinRate_Post2022", "Away_WinRate_Post2022", "WinRate_Difference",
    "Home_WC_Titles", "Away_WC_Titles", "Title_Difference",
    "Home_Squad_Form_Sharpness", "Away_Squad_Form_Sharpness", "Player_Form_Difference",
    "Neutral_Venue"
]

# 2. Load Assets Safely (Duplicate removed, caching completely disabled for clean disk reads)
def load_all_assets():
    profile_path = "data/processed/squad_advanced_profile.csv"
    player_path = "data/processed/web_player_performance.csv"
    model_path = "models/match_predictor.pkl"
    
    if not os.path.exists(profile_path) or not os.path.exists(player_path) or not os.path.exists(model_path):
        st.error("❌ Required backend assets missing. Run your complete terminal pipeline sequence first!")
        st.stop()
        
    profile_df = pd.read_csv(profile_path)
    player_df = pd.read_csv(player_path)
    
    # Ensure any trailing spaces are cleanly removed
    profile_df["National_Team"] = profile_df["National_Team"].str.strip()
    player_df["National_Team"] = player_df["National_Team"].str.strip()
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    return profile_df, player_df, model

profile_df, player_df, model = load_all_assets()
teams_list = sorted(profile_df["National_Team"].tolist())

# 3. Interactive Selection Menu Columns Configuration
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 Home Team")
    home_team = st.selectbox("Select Team A", teams_list, index=teams_list.index("USA") if "USA" in teams_list else 0)
    h_meta = profile_df[profile_df["National_Team"] == home_team].iloc[0]
    h_sharpness = player_df[player_df["National_Team"] == home_team]["Recent_Form_Sharpness_Rating"].mean()
    
    st.metric("Web Elo Rating", f"{h_meta['Latest_Elo_Rating']:.1f}")
    st.write(f"Avg Squad Form: **{h_sharpness:.2f}/10**")

with col2:
    st.subheader("🚀 Away Team")
    away_team = st.selectbox("Select Team B", teams_list, index=teams_list.index("Portugal") if "Portugal" in teams_list else 1)
    a_meta = profile_df[profile_df["National_Team"] == away_team].iloc[0]
    a_sharpness = player_df[player_df["National_Team"] == away_team]["Recent_Form_Sharpness_Rating"].mean()
    
    st.metric("Web Elo Rating", f"{a_meta['Latest_Elo_Rating']:.1f}")
    st.write(f"Avg Squad Form: **{a_sharpness:.2f}/10**")

st.markdown("---")

# 4. Environment Matchday Toggles
st.subheader("🏟️ Match Environment")
is_neutral = st.checkbox(
    "Is this a Neutral Venue Match? (Uncheck only if a host nation like USA, Mexico, or Canada is playing at home)", 
    value=True
)
neutral_value = 1 if is_neutral else 0

st.markdown("---")

# 5. Process Array Vector Inputs and Run Inference
if st.button("🔮 Run 2026 Match Simulation", use_container_width=True):
    if home_team == away_team:
        st.error("A team cannot play against itself! Select two unique nations.")
    else:
        h_elo, a_elo = h_meta['Latest_Elo_Rating'], a_meta['Latest_Elo_Rating']
        h_wr, a_wr = h_meta['Win_Rate_Post_2022'], a_meta['Win_Rate_Post_2022']
        h_titles, a_titles = h_meta['World_Cup_Titles'], a_meta['World_Cup_Titles']
        
        # Build raw DataFrame matrix
        raw_input = pd.DataFrame([{
            "Home_Elo": h_elo,
            "Away_Elo": a_elo,
            "Elo_Difference": h_elo - a_elo,
            "Home_WinRate_Post2022": h_wr,
            "Away_WinRate_Post2022": a_wr,
            "WinRate_Difference": h_wr - a_wr,
            "Home_WC_Titles": h_titles,
            "Away_WC_Titles": a_titles,
            "Title_Difference": h_titles - a_titles,
            "Home_Squad_Form_Sharpness": h_sharpness,
            "Away_Squad_Form_Sharpness": a_sharpness,
            "Player_Form_Difference": h_sharpness - a_sharpness,
            "Neutral_Venue": neutral_value
        }])
        
        # Force columns to match the training feature array shape exactly
        input_matrix = raw_input[FEATURE_ORDER]
        
        # Compute probabilities -> [0: Away Win, 1: Home Win, 2: Draw]
        probabilities = model.predict_proba(input_matrix)[0]
        
        # Render clean results layout
        st.success(f"### Simulation Summary ({'Neutral Venue' if is_neutral else 'True Home Advantage'}): {home_team} vs {away_team}")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric(f"🏠 {home_team} Win", f"{probabilities[1]:.1%}")
        res_col2.metric("🤝 Match Draw", f"{probabilities[2]:.1%}")
        res_col3.metric(f"🚀 {away_team} Win", f"{probabilities[0]:.1%}")