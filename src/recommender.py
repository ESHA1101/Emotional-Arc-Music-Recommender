import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmotionalArcRecommender:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        print(f"🔧 Initializing Recommender with {len(self.df)} tracks...")
        self.embedding_matrix = np.vstack(self.df['clap_embedding'].values)
        self.valence_matrix = np.vstack(self.df['valence_trajectory'].values)
        self.arousal_matrix = np.vstack(self.df['arousal_trajectory'].values)
        print("Recommender Ready.")
    
    def recommend(self, track_id, method='vibe', n=5):
        try:
            idx = self.df[self.df['track_id'] == track_id].index[0]
        except IndexError:
            return []
        
        if method == 'vibe':
            query = self.embedding_matrix[idx].reshape(1, -1)
            sim_scores = cosine_similarity(query, self.embedding_matrix)[0]
        elif method == 'story':
            v_query = self.valence_matrix[idx].reshape(1, -1)
            a_query = self.arousal_matrix[idx].reshape(1, -1)
            v_sim = cosine_similarity(v_query, self.valence_matrix)[0]
            a_sim = cosine_similarity(a_query, self.arousal_matrix)[0]
            sim_scores = (v_sim + a_sim) / 2
        
        top_indices = np.argsort(sim_scores)[::-1]
        top_indices = [i for i in top_indices if i != idx][:n]
        
        results = []
        for i in top_indices:
            row = self.df.iloc[i]
            results.append({
                'track_id': row['track_id'],
                'title': row['title'],
                'artist': row['artist'],
                'genre': row['genre'],
                'arc_type': row['arc_type'],
                'score': sim_scores[i]
            })
        return results