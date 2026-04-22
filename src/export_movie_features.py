"""
Génère data/movies_enriched.csv en combinant genres + tags MovieLens.
Usage : python src/export_movie_features.py
Sortie : data/movies_enriched.csv (movieId, title, genres, tags_text, features)
"""
import os
import pandas as pd

MOVIES_PATH = "ml-32m/movies.csv"
TAGS_PATH   = "ml-32m/tags.csv"
OUTPUT      = "data/movies_enriched.csv"
MIN_TAG_COUNT = 3  # ignorer les tags anecdotiques (< 3 occurrences par film)

print("Chargement...")
movies = pd.read_csv(MOVIES_PATH).dropna(subset=["movieId", "title"])
tags   = pd.read_csv(TAGS_PATH).dropna(subset=["movieId", "tag"])

print(f"  {len(movies):,} films  |  {len(tags):,} tags bruts")

# Nettoyer les tags : minuscules, supprimer les doublons par (userId, movieId, tag)
tags["tag"] = tags["tag"].str.lower().str.strip()
tags = tags.drop_duplicates(subset=["userId", "movieId", "tag"])

# Garder seulement les tags qui apparaissent >= MIN_TAG_COUNT fois sur un même film
tag_counts = tags.groupby(["movieId", "tag"]).size().reset_index(name="n")
tag_counts = tag_counts[tag_counts["n"] >= MIN_TAG_COUNT]

# Agréger en une chaîne de texte par film
tags_agg = (
    tag_counts.groupby("movieId")["tag"]
    .apply(lambda t: " ".join(t.tolist()))
    .reset_index()
    .rename(columns={"tag": "tags_text"})
)

print(f"  {tags_agg['movieId'].nunique():,} films avec tags retenus (>= {MIN_TAG_COUNT} occ.)")

# Fusionner avec les films
df = movies.merge(tags_agg, on="movieId", how="left")
df["tags_text"] = df["tags_text"].fillna("")

# Construire la colonne features = genres nettoyés + tags
df["genres_clean"] = df["genres"].fillna("").str.replace("|", " ", regex=False)
df["features"]     = (df["genres_clean"] + " " + df["tags_text"]).str.strip()

os.makedirs("data", exist_ok=True)
df[["movieId", "title", "genres", "genres_clean", "tags_text", "features"]].to_csv(OUTPUT, index=False)

n_enriched = (df["tags_text"] != "").sum()
print(f"Exporte : {OUTPUT}")
print(f"  {n_enriched:,} films enrichis avec tags ({n_enriched/len(df)*100:.1f}% du catalogue)")
print(f"  {len(df) - n_enriched:,} films genres seuls (pas de tags suffisants)")
