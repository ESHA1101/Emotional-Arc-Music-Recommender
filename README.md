# 🌊 Emotional Arc Music Recommender

A novel music recommendation system that matches songs based on **emotional trajectory** rather than just genre or audio similarity.

## The Problem

Traditional recommenders match songs by:
- Genre tags (limited, subjective)
- Audio similarity (finds similar-sounding tracks)

**They miss the emotional *story* of a song.**

## The Solution

This system extracts **temporal emotion features** from audio:
- **Valence trajectory**: Mood evolution (dark → bright)
- **Arousal trajectory**: Energy evolution (calm → intense)
- **Arc classification**: Builder, Fader, Explosive, Steady, Rollercoaster, Dynamic

Songs are matched by **how they make you feel over time**, not just how they sound.

## Live Demo

[Try it here](https://emotional-arc-music-recommender.streamlit.app/) 

## Performance

Evaluated on 8,000 FMA tracks:
- **Emotional Consistency**: 66.6% (matches correct arc type)
- **Genre Diversity**: 86.2% (discovers music across genres)

Example: A Hip-Hop track with an "Explosive" arc gets matched with:
- Folk songs (different genre, same emotional build-up)
- Electronic songs (different sound, same energy trajectory)

## Tech Stack

- **CLAP** (Contrastive Language-Audio Pretraining) for audio embeddings
- **Librosa** for temporal feature extraction
- **Streamlit** for UI
- **scikit-learn** for similarity computation

## Run Locally
```bash
# Clone
git clone https://github.com/ESHA1101/emotional-arc-music-recsys.git
cd emotional-arc-music-recsys

# Install
pip install -r requirements.txt

# Run
streamlit run app/app.py
```

## Project Structure

```text
├── app/
│   └── app.py                  # Streamlit Web App (Frontend)
│
├── src/
│   └── recommender.py          # Recommendation Logic (Inference)
│
├── preprocess/                 # 🔬 Research & Data Pipeline
│   ├── 01_download_data.ipynb  # FMA Dataset Downloader
│   ├── 02_feature_logic.ipynb  # CLAP + Librosa experimentation
│   └── 03_batch_process.ipynb  # Batch processing (8,000 tracks)
│
├── data/
│   └── music_data.pkl          # Processed Feature Database
│
├── requirements.txt            # App dependencies (Streamlit, Pandas)
└── README.md
```text

## Future Work

- [ ] Upload feature (live CLAP inference)
- [ ] Playlist generation based on emotional narratives
- [ ] API endpoint for third-party integration
- [ ] Multi-modal features (lyrics + audio)


## 📂 Data Source
This project uses the **FMA (Free Music Archive)** dataset:
> Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017). FMA: A Dataset for Music Analysis.

