# Sparkle Movie

Système de recommandation de films basé sur le dataset **MovieLens 32M**, développé avec **Apache Spark**, **PySpark MLlib**, et exposé via une application web complète (FastAPI + frontend SPA + Docker).

---

## Contexte

Une plateforme de streaming vidéo souhaite améliorer l'expérience utilisateur en proposant des recommandations de films personnalisées. L'enjeu est de traiter des dizaines de millions de notes utilisateurs à grande échelle, tout en couvrant un catalogue de films très large dont une grande partie reçoit peu d'évaluations.

**Problématique :**
> Comment recommander des films pertinents à un utilisateur, en s'appuyant sur ses préférences passées et sur le comportement d'utilisateurs similaires ?

---

## Structure du projet

```
sparkle-movie/
├── data/
│   ├── download_dataset.py      # Script de téléchargement du dataset MovieLens 32M
│   ├── recommendations.csv      # Top-10 ALS par utilisateur (2M lignes) — généré
│   ├── model_metrics.json       # RMSE, couverture, date d'entraînement — généré
│   └── profiles.json            # Profils des 5 utilisateurs fictifs — généré
├── notebooks/
│   └── 01_exploration.ipynb     # Exploration, modélisation (ALS, Content-Based, KNN) et évaluation
├── src/
│   ├── train_and_export.py      # Entraînement ALS Spark + export CSV/JSON
│   └── export_profiles.py       # Génération des profils fictifs (3 algos)
├── api/
│   ├── main.py                  # API REST FastAPI (5 endpoints)
│   └── Dockerfile               # Image Docker de l'API
├── frontend/
│   ├── index.html               # SPA 3 onglets (Dashboard, Profils, Recherche)
│   └── Dockerfile               # Image Docker nginx
├── tests/
│   └── test_api.py              # 14 tests Pytest
├── docker-compose.yml           # Orchestration des 2 containers
├── .github/
│   └── workflows/ci.yml         # Pipeline CI/CD GitHub Actions
├── requirements.txt
└── README.md
```

---

## Dataset

**MovieLens 32M** — GroupLens Research, Université du Minnesota

| Fichier | Contenu | Volume |
|---------|---------|--------|
| `ratings.csv` | Notes utilisateurs (userId, movieId, rating, timestamp) | 32 000 204 lignes |
| `movies.csv` | Métadonnées des films (movieId, title, genres) | 87 585 films |

Les données ne sont pas versionnées dans ce repository. Pour télécharger le dataset :

```bash
python data/download_dataset.py
```

### Analyse exploratoire — Résultats clés

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| Note moyenne globale | 3.54 / 5 | Biais de sélection positif |
| Médiane notes/utilisateur | 73 films | Historique suffisant pour ALS |
| Films avec < 5 notes | 48% du catalogue | Cold start massif côté films |
| Genre le plus noté | Drama (14M notes) | Genre dominant |
| Genre le mieux noté | Film-Noir (3.92 / 5) | Audience de niche très engagée |

**Cold Start Problem :**
- 0% des utilisateurs sont "froids" (dataset pré-filtré à 20 notes minimum)
- 48% des films ont moins de 5 notes → le Collaborative Filtering seul est insuffisant

---

## Algorithmes utilisés

### 1. ALS — Alternating Least Squares (Collaborative Filtering)

Algorithme natif de **Spark MLlib** qui décompose la matrice utilisateurs × films en deux matrices de facteurs latents. Il prédit les notes manquantes en exploitant les similarités de comportement entre utilisateurs.

**Hyperparamètres :**
- `rank` : dimension des vecteurs de facteurs latents (défaut : 10)
- `regParam` : régularisation pour éviter l'overfitting
- `maxIter` : nombre d'itérations
- `coldStartStrategy="drop"` : ignore les IDs inconnus lors de l'évaluation

**Résultats :**
- RMSE = **0.8112** sur le jeu de test (split 80/20)
- Couverture du catalogue = **0.91%** (ALS se concentre sur les films populaires)

**Avantages :** Scalable, natif Spark, très performant sur des données denses  
**Limites :** Couverture faible — aveugle sur les films peu notés

---

### 2. Content-Based Filtering (TF-IDF + Similarité Cosinus)

Recommande des films similaires à ceux qu'un utilisateur a aimés, en comparant leurs genres.

**Pipeline :**
1. Nettoyage des genres (`Action|Comedy` → `Action Comedy`)
2. **TF-IDF** (Term Frequency-Inverse Document Frequency) — vectorise les genres de chaque film
3. **Similarité cosinus** — mesure la proximité entre le film préféré de l'utilisateur et tous les autres films
4. Retourne les N films les plus proches non encore vus

**Avantages :** Résout le cold start côté films, indépendant des notes  
**Limites :** Limité aux genres disponibles, ne découvre pas de nouveaux profils

---

### 3. KNN — User-User Collaborative Filtering

Trouve les utilisateurs aux goûts similaires, puis recommande les films bien notés par ces voisins.

**Pipeline :**
1. Identification des candidats voisins (≥ 5 films en commun avec l'utilisateur cible)
2. Construction d'une matrice pivot utilisateurs × films notés
3. **Similarité cosinus** sur les vecteurs de notes → sélection des K=15 voisins les plus proches
4. Recommandation des films notés ≥ 4★ par ces voisins, non encore vus par l'utilisateur

**Avantages :** Intuitif, exploite les comportements réels des utilisateurs  
**Limites :** Sensible aux utilisateurs rares, coûteux en mémoire à grande échelle

---

## Évaluation comparative

Recommandations générées pour 5 utilisateurs fictifs :

| Utilisateur | ID | Genres favoris |
|-------------|-----|---------------|
| Akram | 1 | Drama, Comedy, Romance |
| Giuliana | 10 | Variable selon historique |
| Hélio | 500 | Variable selon historique |
| Victor | 2000 | Variable selon historique |
| Isabelle | 5000 | Variable selon historique |

| Approche | Précision (RMSE) | Couverture | Points forts |
|----------|-----------------|------------|--------------|
| **ALS** | 0.8112 | 0.91% | Personnalisation fine sur historique riche |
| **Content-Based** | N/A | ~100% | Couvre les films froids, indépendant des notes |
| **KNN user-user** | N/A | Dépend des voisins | Exploite les comportements similaires |

**Conclusion :** Aucune approche seule ne suffit. Un système hybride ALS + Content-Based offre la meilleure couverture du catalogue (réponse au cold start) tout en maintenant une précision élevée sur les utilisateurs actifs.

---

## État de l'art — Veille Technologique

### ALS vs Deep Learning

| Critère | ALS (Spark MLlib) | Deep Learning (NCF, BERT4Rec) |
|---------|-------------------|-------------------------------|
| Scalabilité | Excellente (distribué nativement) | Bonne (GPU requis) |
| Interprétabilité | Moyenne (facteurs latents) | Faible (boîte noire) |
| Cold start | Non résolu | Partiellement résolu |
| Coût infrastructure | Faible (CPU cluster) | Élevé (GPU) |
| Précision | Bonne sur données denses | Meilleure sur données riches |
| Temps d'entraînement | Rapide | Long |

**Choix retenu : ALS** — justifié par la taille du dataset (32M lignes), l'absence de GPU, et l'intégration native dans l'écosystème Spark.

### Pourquoi Apache Spark ?

Sur 32 millions de notes, les alternatives présentent des limitations critiques :

| Outil | Limite |
|-------|--------|
| Pandas | Charge le dataset en RAM — crash sur machine standard |
| Scikit-learn | Mono-thread, pas conçu pour du distribué |
| SQL classique | Pas de MLlib, pas de traitement distribué natif |
| **Spark** | Distribué, MLlib intégré, traitement en mémoire partitionné |

### Conformité RGPD

Le dataset MovieLens est **anonymisé** : les `userId` sont des identifiants numériques sans lien avec des données personnelles réelles. Toutefois, en contexte de production réelle :

- Les recommandations personnalisées constituent un **traitement de données à caractère personnel** (article 4 RGPD)
- L'utilisateur doit être informé de l'usage de ses données (article 13)
- Un droit d'opposition au profilage doit être prévu (article 21)
- Les modèles entraînés sur des données utilisateurs doivent être supprimables sur demande (**droit à l'oubli**, article 17)

---

## Installation & Déploiement

### Prérequis

- Python 3.11+
- Java 21 (requis par PySpark)
- Docker Desktop

```bash
# Windows — installer Java via winget
winget install Microsoft.OpenJDK.21
```

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Télécharger le dataset

```bash
python data/download_dataset.py
```

### 3. Entraîner le modèle et exporter les recommandations

```bash
python src/train_and_export.py
```

Génère :
- `data/recommendations.csv` — top-10 ALS pour chaque utilisateur (2M lignes)
- `data/model_metrics.json` — RMSE, couverture, date d'entraînement

### 4. Générer les profils fictifs

```bash
python src/export_profiles.py
```

Génère :
- `data/profiles.json` — recommandations ALS + Content-Based + KNN pour 5 utilisateurs fictifs

### 5. Lancer l'application via Docker

```bash
docker-compose up -d
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| API REST | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |

### 6. Lancer les notebooks

```bash
jupyter notebook notebooks/
```

---

## Application Web

### Frontend — SPA 3 onglets (`frontend/index.html`)

Servie via **nginx** sur le port 3000.

| Onglet | Contenu |
|--------|---------|
| **Dashboard** | 5 stat cards (32M notes, 87K films, 200K users, RMSE, couverture) + tableau comparatif des 3 algorithmes + conclusion |
| **Profils fictifs** | 5 utilisateurs (Akram, Giuliana, Hélio, Victor, Isabelle) — clic → comparaison côte à côte ALS / Content-Based / KNN + historique noté |
| **Recherche libre** | Saisie d'un userId, slider 1–10 résultats, grille de recommandations ALS |

### API REST (`api/main.py`)

Construite avec **FastAPI**, servie via **uvicorn** sur le port 8000.

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/health` | Statut de l'API + nb utilisateurs/recommandations |
| GET | `/metrics` | RMSE, couverture, date d'entraînement |
| GET | `/recommend/{user_id}?n=10` | Top-N films ALS (max 10) |
| GET | `/profiles` | Les 5 profils fictifs complets (3 algos) |
| GET | `/profiles/{user_id}` | Profil individuel |

### Exemple de réponse

```
GET /recommend/42?n=3
```

```json
{
  "user_id": 42,
  "n_recommendations": 3,
  "recommendations": [
    { "rank": 1, "movieId": 318, "title": "Shawshank Redemption, The (1994)", "genres": "Crime|Drama", "score": 4.87 },
    { "rank": 2, "movieId": 858, "title": "Godfather, The (1972)", "genres": "Crime|Drama", "score": 4.82 },
    { "rank": 3, "movieId": 527, "title": "Schindler's List (1993)", "genres": "Drama|War", "score": 4.79 }
  ]
}
```

---

## Docker

Deux containers orchestrés via **docker-compose** :

| Container | Image | Port | Rôle |
|-----------|-------|------|------|
| `sparkle-api` | `python:3.11-slim` | 8000 | FastAPI + uvicorn |
| `sparkle-frontend` | `nginx:alpine` | 3000 | Serveur HTTP statique |

Le dossier `./data` est monté en volume dans l'API — une mise à jour de `profiles.json` ne nécessite qu'un `docker restart sparkle-api`, sans rebuild de l'image.

---

## Tests

```bash
pytest tests/ -v
```

14 tests couvrant :
- Santé de l'API (`/health`, `/metrics`)
- Validité des recommandations (structure, ordre, unicité)
- Validation des paramètres (n invalide → 422, user inconnu → 404)
- Performance modèle (RMSE < 1.0 — Green Build)
- Intégrité des données (pas de doublons, scores positifs)

---

## Intégration Continue (CI)

GitHub Actions exécute automatiquement à chaque push sur `main` :
1. Installation des dépendances
2. Vérification de la présence du CSV de recommandations
3. Lancement de la suite de tests Pytest
4. Vérification que le RMSE reste sous le seuil défini

---

## Monitoring

`data/model_metrics.json` est généré à chaque entraînement et expose :
- RMSE sur le jeu de test
- Couverture du catalogue (% de films recommandés)
- Date et heure d'entraînement

**Stratégie de détection du Data Drift :**
La distribution des notes et la popularité des genres sont mesurées périodiquement. Un écart significatif par rapport aux statistiques d'entraînement déclenche un réentraînement via `src/train_and_export.py`.

---

## Références

- [MovieLens Dataset — GroupLens](https://grouplens.org/datasets/movielens/)
- [Apache Spark MLlib — ALS](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context.* ACM TIIS.
