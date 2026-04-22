"""
API REST Sparkle Movie — FastAPI
Endpoints :
  GET  /recommend/{user_id}  → top-N films recommandés pour un utilisateur
  GET  /metrics              → métriques du modèle (RMSE, couverture)
  GET  /health               → statut de l'API
  GET  /profiles             → profils fictifs (ALS, Content-Based, KNN)
  GET  /movies/search        → recherche de films par titre
  POST /custom-profile       → recommandations personnalisées pour un nouvel utilisateur
"""
import os
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECS_PATH     = os.path.join(BASE_DIR, "data", "recommendations.csv")
METRICS_PATH  = os.path.join(BASE_DIR, "data", "model_metrics.json")
PROFILES_PATH = os.path.join(BASE_DIR, "data", "profiles.json")
MOVIES_PATH          = os.path.join(BASE_DIR, "data", "movies.csv")
MOVIES_ENRICHED_PATH = os.path.join(BASE_DIR, "data", "movies_enriched.csv")

# ── Chargement des données au démarrage ───────────────────────────────────────
def load_recommendations() -> pd.DataFrame:
    if not os.path.exists(RECS_PATH):
        raise RuntimeError("Exécutez d'abord : python src/train_and_export.py")
    return pd.read_csv(RECS_PATH)

def load_metrics() -> dict:
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)

def load_profiles() -> dict:
    if not os.path.exists(PROFILES_PATH):
        return {}
    with open(PROFILES_PATH, encoding="utf-8") as f:
        return json.load(f)

def load_movies() -> pd.DataFrame:
    # Préférer la version enrichie (genres + tags) si disponible
    path = MOVIES_ENRICHED_PATH if os.path.exists(MOVIES_ENRICHED_PATH) else MOVIES_PATH
    if not os.path.exists(path):
        return pd.DataFrame(columns=["movieId", "title", "genres", "genres_clean", "tags_text", "features"])
    df = pd.read_csv(path)
    if "genres_clean" not in df.columns:
        df["genres_clean"] = df["genres"].fillna("").str.replace("|", " ", regex=False)
    if "tags_text" not in df.columns:
        df["tags_text"] = ""
    if "features" not in df.columns:
        df["features"] = df["genres_clean"]
    df["tags_text"] = df["tags_text"].fillna("")
    df["features"]  = df["features"].fillna("").str.strip()
    enriched = (df["tags_text"] != "").sum()
    print(f"Films charges : {len(df):,} ({enriched:,} enrichis avec tags)")
    return df

recs_df   = load_recommendations()
metrics   = load_metrics()
profiles  = load_profiles()
movies_df = load_movies()

# ── TF-IDF sur genres + tags (Content-Based) ──────────────────────────────────
tfidf     = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
tfidf_mat = tfidf.fit_transform(movies_df["features"]) if len(movies_df) else None
mid_to_idx  = {int(row["movieId"]): i for i, row in movies_df.iterrows()}
mid_to_tags = movies_df.set_index("movieId")["tags_text"].to_dict() if "tags_text" in movies_df.columns else {}

# ── Index genre par film pour les lookups rapides ─────────────────────────────
# Pré-calculer un set de genres par movieId depuis recs_df
recs_genres = recs_df.drop_duplicates("movieId").set_index("movieId")["genres"].fillna("")

# ── Modèles de réponse ────────────────────────────────────────────────────────
class MovieRecommendation(BaseModel):
    rank:    int
    movieId: int
    title:   str
    genres:  str
    score:   float

class RecommendationResponse(BaseModel):
    user_id:           int
    n_recommendations: int
    recommendations:   List[MovieRecommendation]

class HealthResponse(BaseModel):
    status:  str
    n_users: int
    n_recs:  int

class RatedMovie(BaseModel):
    movieId: int
    rating:  float

class CustomProfileRequest(BaseModel):
    name:    str
    genres:  List[str]
    ratings: List[RatedMovie]

class CustomProfileResponse(BaseModel):
    name:          str
    als:           List[dict]
    content_based: List[dict]
    knn:           List[dict]

# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sparkle Movie API",
    description="Système de recommandation de films basé sur ALS (MovieLens 32M)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Endpoints existants ───────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    return {
        "status":  "ok",
        "n_users": int(recs_df["userId"].nunique()),
        "n_recs":  len(recs_df),
    }

@app.get("/metrics", tags=["Monitoring"])
def get_metrics():
    if not metrics:
        raise HTTPException(status_code=404, detail="Métriques introuvables.")
    return metrics

@app.get("/recommend/{user_id}", response_model=RecommendationResponse, tags=["Recommandations"])
def recommend(
    user_id: int,
    n: int = Query(default=10, ge=1, le=50),
):
    user_recs = recs_df[recs_df["userId"] == user_id].head(n)
    if user_recs.empty:
        raise HTTPException(status_code=404, detail=f"Utilisateur {user_id} introuvable.")
    recommendations = [
        MovieRecommendation(
            rank=int(row["rank"]), movieId=int(row["movieId"]),
            title=str(row["title"]), genres=str(row["genres"]),
            score=round(float(row["score"]), 4),
        )
        for _, row in user_recs.iterrows()
    ]
    return RecommendationResponse(user_id=user_id, n_recommendations=len(recommendations), recommendations=recommendations)

@app.get("/profiles", tags=["Profils fictifs"])
def get_profiles():
    if not profiles:
        raise HTTPException(status_code=404, detail="Exécutez d'abord : python src/export_profiles.py")
    return profiles

@app.get("/profiles/{user_id}", tags=["Profils fictifs"])
def get_profile(user_id: int):
    if not profiles:
        raise HTTPException(status_code=404, detail="Exécutez d'abord : python src/export_profiles.py")
    user = next((u for u in profiles.get("users", []) if u["id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail=f"Profil {user_id} introuvable.")
    return user

# ── Recherche de films ────────────────────────────────────────────────────────
@app.get("/movies/search", tags=["Films"])
def search_movies(
    q: str = Query(..., min_length=2, description="Titre ou fragment de titre"),
    limit: int = Query(default=10, ge=1, le=30),
):
    """Recherche des films par titre (insensible à la casse)."""
    if movies_df.empty:
        raise HTTPException(status_code=503, detail="movies.csv non chargé.")
    mask = movies_df["title"].str.contains(q, case=False, na=False)
    results = movies_df[mask].head(limit)
    return [
        {"movieId": int(r["movieId"]), "title": r["title"], "genres": r["genres"]}
        for _, r in results.iterrows()
    ]

# ── Profil personnalisé ───────────────────────────────────────────────────────
@app.post("/custom-profile", response_model=CustomProfileResponse, tags=["Profil personnalisé"])
def custom_profile(req: CustomProfileRequest):
    """
    Génère des recommandations personnalisées (ALS proxy, Content-Based, KNN proxy)
    pour un nouvel utilisateur décrit par ses genres favoris et ses films notés.
    """
    seen_ids = {r.movieId for r in req.ratings}
    n = 10

    # ── Content-Based ─────────────────────────────────────────────────────────
    cb_list = []
    if tfidf_mat is not None:
        # Construire un vecteur "profil" depuis genres favoris + features des films aimés
        liked_features = " ".join(req.genres)
        for r in req.ratings:
            if r.rating >= 4.0 and r.movieId in mid_to_idx:
                idx = mid_to_idx[r.movieId]
                liked_features += " " + movies_df.iloc[idx]["features"]

        query_vec = tfidf.transform([liked_features])
        scores    = cosine_similarity(query_vec, tfidf_mat)[0]

        top_idx = np.argsort(scores)[::-1]
        for i in top_idx:
            row = movies_df.iloc[i]
            mid = int(row["movieId"])
            if mid not in seen_ids and scores[i] > 0:
                tags_raw = mid_to_tags.get(mid, "")
                top_tags = ", ".join(str(tags_raw).split()[:6]) if tags_raw else ""
                cb_list.append({
                    "rank": len(cb_list) + 1,
                    "movieId": mid,
                    "title": row["title"],
                    "genres": row["genres"],
                    "tags": top_tags,
                    "score": round(float(scores[i]), 4),
                })
            if len(cb_list) >= n:
                break

    # ── ALS : films les mieux scorés correspondant aux genres cibles ──────────────
    # Approche : parmi toutes les recommandations ALS du dataset, trouver les films
    # dont les genres correspondent aux préférences de l'utilisateur, triés par score moyen
    genre_set = set(req.genres)

    def genre_match(g_str):
        if not g_str:
            return False
        return bool(set(str(g_str).split("|")) & genre_set)

    mask_genre  = recs_df["genres"].apply(genre_match)
    mask_unseen = ~recs_df["movieId"].isin(seen_ids)
    als_pool    = recs_df[mask_genre & mask_unseen]

    als_agg = (
        als_pool.groupby(["movieId", "title", "genres"])
        .agg(score=("score", "mean"), votes=("score", "count"))
        .reset_index()
        .sort_values(["votes", "score"], ascending=False)
        .head(n)
    )

    als_list = []
    for _, row in als_agg.iterrows():
        als_list.append({
            "rank":    len(als_list) + 1,
            "movieId": int(row["movieId"]),
            "title":   str(row["title"]),
            "genres":  str(row["genres"]),
            "score":   round(min(float(row["score"]), 5.0), 4),
        })

    # ── KNN proxy : films ALS populaires dans les genres, hors top ALS ──────────
    # Même pool mais trié différemment : on favorise les films moins consensuels
    # (moins souvent recommandés mais très bien notés) pour diversifier des résultats ALS
    als_seen_ids = {m["movieId"] for m in als_list} | seen_ids

    knn_pool = recs_df[mask_genre & ~recs_df["movieId"].isin(als_seen_ids)]
    knn_agg  = (
        knn_pool.groupby(["movieId", "title", "genres"])
        .agg(score=("score", "mean"), votes=("score", "count"))
        .reset_index()
        .query("votes >= 3")                      # au moins 3 utilisateurs ont reçu ce film
        .sort_values(["score", "votes"], ascending=False)   # trier par score d'abord
        .head(n)
    )

    knn_list = []
    for _, row in knn_agg.iterrows():
        knn_list.append({
            "rank":    len(knn_list) + 1,
            "movieId": int(row["movieId"]),
            "title":   str(row["title"]),
            "genres":  str(row["genres"]),
            "score":   round(min(float(row["score"]), 5.0), 4),
            "votes":   int(row["votes"]),
        })

    return CustomProfileResponse(
        name=req.name,
        als=als_list,
        content_based=cb_list,
        knn=knn_list,
    )
