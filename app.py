import streamlit as st
import pandas as pd
from src.recommender import EmotionalArcRecommender
import os

st.set_page_config(page_title="Emotional Arc Music Recommender", layout="wide")

@st.cache_resource
def load_system():
    data_path = os.path.join("data", "music_data.pkl")
    if not os.path.exists(data_path):
        return None, None
    df = pd.read_pickle(data_path)
    rec_engine = EmotionalArcRecommender(df)
    return rec_engine, df

st.title("🎼 Emotional Arc Music Recommender")
st.markdown("### Recommend by emotional trajectory, not just audio similarity")

rec_engine, df = load_system()

if rec_engine is None:
    st.error("Data file not found!")
else:
    all_songs = df['title'] + " - " + df['artist']
    selected_song = st.selectbox("Search for a song you like:", all_songs)

    if st.button("Get Recommendations"):
        seed_track = df[df['title'] + " - " + df['artist'] == selected_song].iloc[0]
        
        st.divider()
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f"**Selected:** {seed_track['title']}")
            st.caption(f"{seed_track['artist']}") 
            st.caption(f"{seed_track['genre']}")
            st.caption(f"Arc: {seed_track['arc_type']}")
            
            st.line_chart(seed_track['arousal_trajectory'], use_container_width=True)
            st.caption("↑ Energy over time")
            
        with col2:
            st.subheader("Results")
            t1, t2 = st.tabs(["Vibe Match (Baseline)", "Story Match (Novel)"])
            
            def draw_song_card(r, is_story_mode=False, seed_arc=None, seed_genre=None):
                with st.container(border=True):
                    st.markdown(f"### {r['title']}")
                    
                    col_artist, col_genre, col_tags = st.columns([2, 1, 2])
                    
                    with col_artist:
                        st.caption(f"{r['artist']}")
                    
                    with col_genre:
                        st.caption(f"{r['genre']}")
                    
                    with col_tags:
                        if is_story_mode:
                            if r['arc_type'] == seed_arc:
                                st.success(f"{r['arc_type']}", icon="✅")
                            else:
                                st.warning(f"{r['arc_type']}", icon="⚠️")
                            
                            if r['genre'] != seed_genre:
                                st.info("New Genre", icon="🌟")
                        else:
                            st.metric("Match Score", f"{int(r['score']*100)}%")
            with t1:
                st.caption("Matches based on audio similarity.")
                for r in rec_engine.recommend(seed_track['track_id'], 'vibe'):
                    draw_song_card(r)

            with t2:
                st.caption("Matches based on emotional trajectory.")
                
                st.markdown(f"**Looking for:** `{seed_track['arc_type']}` arc")
                
                for r in rec_engine.recommend(seed_track['track_id'], 'story'):
                    draw_song_card(r, is_story_mode=True, 
                                seed_arc=seed_track['arc_type'], 
                                seed_genre=seed_track['genre'])