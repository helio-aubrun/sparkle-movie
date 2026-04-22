"""
Génère data/profiles.json pour les 5 utilisateurs fictifs.
Calcule ALS (depuis CSV), Content-Based (TF-IDF + cosine) et KNN (user-user CF).
Usage : python src/export_profiles.py
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import json, os

USER_IDS   = [1, 10, 500, 2000, 5000]
USER_NAMES = {1: "Akram", 10: "Giuliana", 500: "Hélio", 2000: "Victor", 5000: "Isabelle"}
N_RECS     = 10

# ── Chargement ────────────────────────────────────────────────────────────────
print("Chargement des données...")
# Préférer movies_enriched.csv (genres + tags) si disponible
if os.path.exists("data/movies_enriched.csv"):
    movies = pd.read_csv("data/movies_enriched.csv")
    print("  movies_enriched.csv charge (genres + tags)")
else:
    movies = pd.read_csv("ml-32m/movies.csv")
    movies["genres_clean"] = movies["genres"].fillna("").str.replace("|", " ", regex=False)
    movies["features"] = movies["genres_clean"]
    print("  movies.csv charge (genres seuls)")

ratings  = pd.read_csv("ml-32m/ratings.csv", usecols=["userId", "movieId", "rating"])
als_recs = pd.read_csv("data/recommendations.csv")
print(f"  {len(movies):,} films — {len(ratings):,} notes")

# ── Content-Based : TF-IDF sur genres + tags ──────────────────────────────────
print("Calcul TF-IDF...")
movies["features"] = movies["features"].fillna("").str.strip()
tfidf        = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(movies["features"])
mid_to_idx   = {int(row["movieId"]): i for i, row in movies.iterrows()}

def content_based_recs(fav_movie_id, seen_ids, n=N_RECS):
    if fav_movie_id not in mid_to_idx:
        return []
    idx = mid_to_idx[fav_movie_id]
    scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
    recs = []
    for i in np.argsort(scores)[::-1]:
        row = movies.iloc[i]
        mid = int(row["movieId"])
        if mid not in seen_ids:
            recs.append({
                "movieId": mid,
                "title":   row["title"],
                "genres":  row["genres"],
                "score":   round(float(scores[i]), 4),
            })
        if len(recs) >= n:
            break
    return recs

# ── KNN : user-user collaborative filtering ───────────────────────────────────
def knn_recs(uid, k=15, n=N_RECS):
    user_movies = set(ratings[ratings["userId"] == uid]["movieId"])
    # Trouver les utilisateurs ayant noté au moins 5 films en commun
    candidates = ratings[ratings["movieId"].isin(user_movies)]
    overlap    = candidates.groupby("userId")["movieId"].count()
    neighbors  = overlap[(overlap >= 5) & (overlap.index != uid)].index.tolist()

    if not neighbors:
        return []

    # Sous-ensemble : target user + voisins candidats sur les films communs
    sample = ratings[
        (ratings["userId"].isin([uid] + neighbors[:500])) &
        (ratings["movieId"].isin(user_movies))
    ]

    pivot = sample.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
    if uid not in pivot.index:
        return []

    user_vec    = pivot.loc[[uid]]
    neighbor_df = pivot.drop(index=uid, errors="ignore")
    if neighbor_df.empty:
        return []

    sims = cosine_similarity(user_vec, neighbor_df)[0]
    top_k_idx = np.argsort(sims)[::-1][:k]
    top_k_ids = neighbor_df.index[top_k_idx].tolist()

    # Films bien notés par les voisins que l'utilisateur n'a pas vus
    seen    = set(ratings[ratings["userId"] == uid]["movieId"])
    nb_rats = ratings[
        (ratings["userId"].isin(top_k_ids)) &
        (~ratings["movieId"].isin(seen)) &
        (ratings["rating"] >= 4.0)
    ]
    top_films = (
        nb_rats.groupby("movieId")
        .agg(votes=("rating", "count"), score=("rating", "mean"))
        .sort_values(["votes", "score"], ascending=False)
        .head(n)
        .reset_index()
        .merge(movies, on="movieId")
    )

    return [
        {
            "movieId": int(r["movieId"]),
            "title":   r["title"],
            "genres":  r["genres"],
            "score":   round(float(r["score"]), 4),
            "votes":   int(r["votes"]),
        }
        for _, r in top_films.iterrows()
    ]

# ── Génération des profils ────────────────────────────────────────────────────
profiles = []

for uid in USER_IDS:
    name = USER_NAMES[uid]
    print(f"\nProfil {name} (userId={uid})...")

    user_ratings = ratings[ratings["userId"] == uid].merge(movies, on="movieId")
    n_rated      = len(user_ratings)

    # Historique top-5
    top5 = user_ratings.nlargest(5, "rating")
    history = [
        {"movieId": int(r["movieId"]), "title": r["title"],
         "genres": r["genres"], "rating": float(r["rating"])}
        for _, r in top5.iterrows()
    ]

    # Genres favoris
    all_genres = []
    for g in user_ratings["genres"].dropna():
        all_genres.extend(g.split("|"))
    fav_genres = [g for g, _ in Counter(all_genres).most_common(3)]

    # ALS
    als = als_recs[als_recs["userId"] == uid].sort_values("rank").head(N_RECS)
    als_list = [
        {"rank": int(r["rank"]), "movieId": int(r["movieId"]),
         "title": r["title"], "genres": r["genres"], "score": round(float(r["score"]), 4)}
        for _, r in als.iterrows()
    ]

    # Content-Based
    seen_ids = set(user_ratings["movieId"])
    fav_id   = int(top5.iloc[0]["movieId"]) if len(top5) > 0 else None
    cb_list  = content_based_recs(fav_id, seen_ids) if fav_id else []
    print(f"  Content-Based : {len(cb_list)} recs")

    # KNN
    knn_list = knn_recs(uid)
    print(f"  KNN           : {len(knn_list)} recs")

    profiles.append({
        "id": uid, "name": name, "n_rated": n_rated,
        "favorite_genres": fav_genres,
        "history": history,
        "als": als_list,
        "content_based": cb_list,
        "knn": knn_list,
    })

os.makedirs("data", exist_ok=True)
with open("data/profiles.json", "w", encoding="utf-8") as f:
    json.dump({"users": profiles}, f, ensure_ascii=False, indent=2)

print("\nProfils exportes : data/profiles.json")
