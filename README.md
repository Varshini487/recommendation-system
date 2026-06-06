# 🎯 Recommendation System

A **hybrid AI Recommendation System** combining collaborative and content-based filtering.

## 🧩 Approaches
| Approach | How it works |
|---------|-------------|
| Collaborative Filtering | Finds similar users, recommends what they liked |
| Content-Based | Recommends items similar to what you've enjoyed |
| Hybrid | Combines both for best accuracy |
| Matrix Factorization | SVD to discover latent factors |

## 🛠️ Tech Stack
- **Surprise / LightFM** – CF algorithms
- **Sentence-Transformers** – content similarity
- **FastAPI** – recommendation API
- **Redis** – real-time caching
- **PostgreSQL** – interaction storage

## 🚀 Getting Started
```bash
git clone https://github.com/Varshini487/recommendation-system
cd recommendation-system
pip install -r requirements.txt
streamlit run app.py
```

## 💡 Use Cases
- Netflix / Spotify-style platforms
- E-commerce product recommendations
- News article personalization
- EdTech course suggestions
