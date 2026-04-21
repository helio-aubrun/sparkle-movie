"""
Tests automatisés — API Sparkle Movie
Exécution : pytest tests/ -v
"""
import json
import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient

# ── Fixtures ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECS_PATH    = os.path.join(BASE_DIR, "data", "recommendations.csv")
METRICS_PATH = os.path.join(BASE_DIR, "data", "model_metrics.json")
RMSE_THRESHOLD = 1.0

@pytest.fixture(scope="module")
def client():
    from api.main import app
    return TestClient(app)

@pytest.fixture(scope="module")
def recs_df():
    return pd.read_csv(RECS_PATH)

@pytest.fixture(scope="module")
def metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)

# ── Tests : Santé de l'API ────────────────────────────────────────────────────
def test_health_status(client):
    """L'API répond 200 et retourne status=ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_health_has_users(client):
    """L'API a bien chargé des données utilisateurs."""
    response = client.get("/health")
    assert response.json()["n_users"] > 0

# ── Tests : Recommandations ───────────────────────────────────────────────────
def test_recommend_valid_user(client, recs_df):
    """Un utilisateur valide reçoit des recommandations."""
    valid_user = int(recs_df["userId"].iloc[0])
    response = client.get(f"/recommend/{valid_user}")
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == valid_user
    assert body["n_recommendations"] > 0
    assert len(body["recommendations"]) > 0

def test_recommend_response_structure(client, recs_df):
    """Chaque recommandation contient les champs attendus."""
    valid_user = int(recs_df["userId"].iloc[0])
    response = client.get(f"/recommend/{valid_user}")
    rec = response.json()["recommendations"][0]
    assert "rank"    in rec
    assert "movieId" in rec
    assert "title"   in rec
    assert "genres"  in rec
    assert "score"   in rec

def test_recommend_n_param(client, recs_df):
    """Le paramètre n limite bien le nombre de résultats."""
    valid_user = int(recs_df["userId"].iloc[0])
    response = client.get(f"/recommend/{valid_user}?n=3")
    assert response.status_code == 200
    assert response.json()["n_recommendations"] <= 3

def test_recommend_unknown_user(client):
    """Un utilisateur inexistant retourne une erreur 404."""
    response = client.get("/recommend/999999999")
    assert response.status_code == 404

def test_recommend_invalid_n(client, recs_df):
    """Un paramètre n invalide (0 ou négatif) retourne une erreur 422."""
    valid_user = int(recs_df["userId"].iloc[0])
    response = client.get(f"/recommend/{valid_user}?n=0")
    assert response.status_code == 422

def test_recommend_scores_in_range(client, recs_df):
    """Les scores de recommandation sont des nombres positifs."""
    valid_user = int(recs_df["userId"].iloc[0])
    response = client.get(f"/recommend/{valid_user}")
    for rec in response.json()["recommendations"]:
        assert rec["score"] >= 0

def test_recommend_ranks_ordered(client, recs_df):
    """Les recommandations sont triées par rang croissant."""
    valid_user = int(recs_df["userId"].iloc[0])
    response = client.get(f"/recommend/{valid_user}")
    ranks = [r["rank"] for r in response.json()["recommendations"]]
    assert ranks == sorted(ranks)

# ── Tests : Métriques ─────────────────────────────────────────────────────────
def test_metrics_endpoint(client):
    """L'endpoint /metrics retourne 200."""
    response = client.get("/metrics")
    assert response.status_code == 200

def test_rmse_below_threshold(metrics):
    """Le RMSE du modèle doit rester sous le seuil défini."""
    rmse = metrics["rmse"]
    assert rmse < RMSE_THRESHOLD, (
        f"RMSE {rmse:.4f} dépasse le seuil {RMSE_THRESHOLD} — modèle à réentraîner."
    )

def test_metrics_fields(metrics):
    """Les métriques contiennent tous les champs attendus."""
    for field in ["rmse", "coverage_pct", "n_users", "n_movies", "trained_at"]:
        assert field in metrics, f"Champ manquant dans les métriques : {field}"

# ── Tests : Validation des données ───────────────────────────────────────────
def test_recommendations_no_duplicates(recs_df):
    """Un même film ne doit pas apparaître deux fois pour un même utilisateur."""
    dupes = recs_df.groupby(["userId", "movieId"]).size()
    assert dupes.max() == 1, "Des doublons (userId, movieId) ont été détectés."

def test_recommendations_ranks_valid(recs_df):
    """Les rangs doivent être positifs."""
    assert (recs_df["rank"] >= 1).all()

def test_recommendations_scores_positive(recs_df):
    """Les scores doivent être positifs."""
    assert (recs_df["score"] >= 0).all()
