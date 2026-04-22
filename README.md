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
│   ├── movies.csv               # Catalogue 87 585 films (copie locale pour l'API)
│   ├── movies_enriched.csv      # Catalogue enrichi avec tags utilisateurs — généré
│   ├── recommendations.csv      # Top-10 ALS par utilisateur (~2M lignes) — généré
│   ├── model_metrics.json       # RMSE, couverture, hyperparamètres — généré
│   └── profiles.json            # Profils des 5 utilisateurs fictifs — généré
├── notebooks/
│   └── 01_exploration.ipynb     # Exploration, modélisation (ALS, Content-Based, KNN) et évaluation
├── src/
│   ├── train_and_export.py      # Entraînement ALS Spark + export CSV/JSON
│   ├── export_movie_features.py # Agrégation des tags MovieLens → movies_enriched.csv
│   └── export_profiles.py       # Génération des profils fictifs (3 algos)
├── api/
│   ├── main.py                  # API REST FastAPI (7 endpoints)
│   └── Dockerfile               # Image Docker de l'API
├── frontend/
│   ├── index.html               # SPA 4 onglets (Dashboard, Profils, Mon Profil, Recherche)
│   └── Dockerfile               # Image Docker nginx
├── tests/
│   └── test_api.py              # Tests Pytest
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
| `tags.csv` | Tags libres posés par les utilisateurs sur les films | 2 000 072 lignes |
| `links.csv` | Correspondances movieId ↔ IMDb / TMDB | 87 585 films |

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

**Hyperparamètres retenus (version finale) :**

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `rank` | 20 | Dimension des vecteurs latents — 20 permet +28% de films distincts vs 10, sans saturer la RAM |
| `regParam` | 0.1 | Régularisation — 0.3 sur-régularisait et uniformisait les vecteurs |
| `nonnegative` | `False` | Facteurs libres (positifs et négatifs) → vraie personnalisation par profil |
| `maxIter` | 10 | Convergence suffisante sur ce dataset |
| `coldStartStrategy` | `drop` | Ignore les IDs inconnus lors de l'évaluation |
| Filtre popularité | ≥ 50 notes | Élimine 71 551 films bruités dont les facteurs sont mal contraints |

**Résultats (modèle v5) :**
- RMSE = **0.7993** sur le jeu de test (split 80/20)
- Couverture = **3.39%** du catalogue (2 969 films distincts recommandés)
- 200 947 utilisateurs couverts, 2 009 470 recommandations générées

**Avantages :** Scalable, natif Spark, très performant sur des données denses
**Limites :** Couverture faible — aveugle sur les films peu notés (< 50 notes)

---

### 2. Content-Based Filtering (TF-IDF + Similarité Cosinus)

Recommande des films similaires au profil d'un utilisateur, en comparant leurs genres **et leurs tags**.

**Pipeline :**
1. Nettoyage des genres (`Action|Comedy` → `Action Comedy`)
2. Agrégation des tags utilisateurs par film (`src/export_movie_features.py`) — tags apparaissant ≥ 3 fois conservés
3. Construction de `features = genres + tags` pour chaque film
4. **TF-IDF** avec bigrammes — vectorise les features de 87 585 films au démarrage de l'API
5. Construction du vecteur profil utilisateur : genres favoris (1 fois) + features des films notés (répétées selon la note : un 5★ compte 5×, un 2★ compte 2×)
6. **Similarité cosinus** — score de proximité entre le profil et chaque film du catalogue
7. Retourne les N films les plus proches non encore vus

**Enrichissement par les tags :**
- 2 000 072 tags bruts → 10 032 films enrichis (11.5% du catalogue)
- Exemples : *The Matrix* → "cyberpunk mindfuck bullet time dystopia thought-provoking virtual reality"
- Résultat : fan de Matrix + Sci-Fi → Terminator, Blade Runner, Dark City (au lieu de films génériques)

**Avantages :** Résout le cold start côté films, indépendant des notes, couvre 100% du catalogue
**Limites :** Signal limité aux genres/tags disponibles, ne découvre pas de nouveaux profils utilisateur

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

## Calibrage du modèle ALS — Historique des runs

Cinq entraînements successifs ont permis d'identifier les hyperparamètres optimaux :

| Run | `nonnegative` | `regParam` | `rank` | Filtre | RMSE | Films distincts | Problème |
|-----|--------------|------------|--------|--------|------|-----------------|---------|
| v1 | `True` | 0.1 | 10 | aucun | 0.8112 | 37 | Scores > 5.0, biais popularité extrême |
| v2 | `True` | 0.3 | 10 | ≥ 50 | 0.9011 | 38 | Sur-régularisation → vecteurs uniformes |
| v3 | `False` | 0.1 | 10 | ≥ 50 | 0.8003 | 2 309 | Bonne qualité, couverture limitée |
| v4 | `False` | 0.1 | 20 | ≥ 20 | 0.7992 | 2 780 | Films de niche bruités (peu notés) |
| **v5** | **`False`** | **0.1** | **20** | **≥ 50** | **0.7993** | **2 969** | **Optimal** ✅ |

**Leçons apprises :**
- `nonnegative=True` sur données creuses force tous les facteurs latents dans le même sens → convergence vers les mêmes 37-38 films populaires pour tous les utilisateurs
- `regParam=0.3` pousse les vecteurs vers zéro → perte de personnalisation
- Un filtre popularité trop bas (≥ 20) introduit des films aux facteurs bruités (anime obscur, comédiens inconnus) en tête de recommandations
- `rank=20` apporte +28% de films distincts vs `rank=10` sans dégrader la qualité

---

## Évaluation comparative

Recommandations générées pour 5 utilisateurs fictifs :

| Utilisateur | ID | Profil |
|-------------|-----|--------|
| Akram | 1 | Cinéma d'auteur classique (Drama, Film-Noir) |
| Giuliana | 10 | Variable selon historique |
| Hélio | 500 | Variable selon historique |
| Victor | 2000 | Blockbusters (Action, Sci-Fi, Fantasy) |
| Isabelle | 5000 | Variable selon historique |

| Approche | RMSE | Couverture | Personnalisation | Cold Start films |
|----------|------|------------|-----------------|-----------------|
| **ALS** | **0.7993** | 3.39% (2 969 films) | Élevée (profils distincts) | Non résolu |
| **Content-Based** | N/A | ~100% | Basée sur genres + tags | Résolu |
| **KNN** | N/A | Variable | Films de niche | Partiel |

**Conclusion :** Un système hybride ALS + Content-Based offre la meilleure couverture du catalogue tout en maintenant une précision élevée sur les utilisateurs actifs. Le KNN enrichit les profils avec des découvertes inattendues.

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

### 3. Générer le catalogue enrichi (genres + tags)

```bash
python src/export_movie_features.py
```

Génère :
- `data/movies_enriched.csv` — 87 585 films avec genres + tags agrégés (10 032 enrichis)

### 4. Entraîner le modèle et exporter les recommandations

```bash
python src/train_and_export.py
```

Durée : ~45 minutes sur machine standard (4 GB RAM alloués à Spark).

Génère :
- `data/recommendations.csv` — top-10 ALS pour chaque utilisateur (~2M lignes, ~150 MB)
- `data/model_metrics.json` — RMSE, couverture, hyperparamètres, date d'entraînement

### 5. Générer les profils fictifs

```bash
python src/export_profiles.py
```

Génère :
- `data/profiles.json` — recommandations ALS + Content-Based (genres+tags) + KNN pour 5 utilisateurs fictifs

### 6. Lancer l'application via Docker

```bash
docker-compose up --build -d
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| API REST | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |

### 7. Lancer les notebooks

```bash
jupyter notebook notebooks/
```

---

## Application Web

### Frontend — SPA 4 onglets (`frontend/index.html`)

Servie via **nginx** sur le port 3000.

| Onglet | Contenu |
|--------|---------|
| **Dashboard** | Stat cards dynamiques (RMSE, couverture, films distincts, films populaires) + tableau comparatif des 3 algorithmes avec détails hyperparamètres + conclusion |
| **Profils fictifs** | 5 utilisateurs (Akram, Giuliana, Hélio, Victor, Isabelle) — clic → comparaison côte à côte ALS / Content-Based / KNN + historique noté |
| **Mon Profil** | Prénom + genres favoris (facultatif, max 3) + notation de films via autocomplétion → recommandations personnalisées via les 3 algorithmes |
| **Recherche libre** | Saisie d'un userId, slider 1–10 résultats, suggestions de profils connus, grille de recommandations ALS avec score prédit |

**Score affiché en violet :** prédiction personnalisée du modèle (0–5 pour ALS/KNN, similarité cosinus 0–1 pour Content-Based). Ce n'est pas la note moyenne IMDb mais une estimation de combien l'utilisateur apprécierait ce film.

### API REST (`api/main.py`)

Construite avec **FastAPI**, servie via **uvicorn** sur le port 8000. TF-IDF (genres + tags) et index calculés au démarrage sur les 87 585 films.

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/health` | Statut de l'API + nb utilisateurs/recommandations |
| GET | `/metrics` | RMSE, couverture, hyperparamètres, date d'entraînement |
| GET | `/recommend/{user_id}?n=10` | Top-N films ALS pour un utilisateur (max 10) |
| GET | `/profiles` | Les 5 profils fictifs complets (3 algos) |
| GET | `/profiles/{user_id}` | Profil individuel par userId |
| GET | `/movies/search?q=...` | Recherche de films par titre (autocomplétion) |
| POST | `/custom-profile` | Recommandations personnalisées pour un nouvel utilisateur |

### Endpoint `/custom-profile`

Accepte un profil utilisateur en JSON et retourne les recommandations des 3 algorithmes. Les genres sont **facultatifs** — les films notés suffisent à générer des recommandations.

```json
POST /custom-profile
{
  "name": "Marie",
  "genres": ["Sci-Fi", "Thriller"],
  "ratings": [
    { "movieId": 2571, "rating": 5.0 }
  ]
}
```

**Pondération du vecteur profil Content-Based :**
- Genres favoris : ajoutés **1 fois** (signal léger)
- Films notés : features répétées selon la note (5★ → 5×, 2★ → 2×) → les films aimés dominent le profil

**Algorithmes retournés :**
- **ALS proxy** : agrège les recommandations existantes par genre, triées par score moyen et nombre de votes
- **Content-Based** : similarité cosinus TF-IDF (genres + tags) entre le profil et les 87 585 films
- **KNN proxy** : films bien notés dans les genres cibles, diversifié par rapport à l'ALS

### Exemple de réponse `/recommend`

```
GET /recommend/2000?n=3
```

```json
{
  "user_id": 2000,
  "n_recommendations": 3,
  "recommendations": [
    { "rank": 1, "movieId": 122914, "title": "Avengers: Infinity War - Part II (2019)", "genres": "Action|Adventure|Sci-Fi", "score": 4.6652 },
    { "rank": 2, "movieId": 122912, "title": "Avengers: Infinity War - Part I (2018)", "genres": "Action|Adventure|Sci-Fi", "score": 4.6409 },
    { "rank": 3, "movieId": 89745,  "title": "Avengers, The (2012)", "genres": "Action|Adventure|Sci-Fi|IMAX", "score": 4.5722 }
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

Après un réentraînement complet :

```bash
python src/export_movie_features.py  # rapide (~1 min)
python src/train_and_export.py       # ~45 min
python src/export_profiles.py        # ~5 min
docker-compose up --build -d
```

---

## Tests

```bash
pytest tests/ -v
```

Tests couvrant :
- Santé de l'API (`/health`, `/metrics`)
- Validité des recommandations (structure, ordre, unicité)
- Validation des paramètres (n invalide → 422, user inconnu → 404)
- Performance modèle (RMSE < 1.0 — Green Build)
- Intégrité des données (pas de doublons, scores positifs)

---

## Intégration Continue (CI)

GitHub Actions exécute automatiquement à chaque push sur `main` :

1. Vérification de la présence de tous les fichiers requis
2. Validation syntaxique Python (`py_compile`)
3. Création de données mock (CSV minimal + metrics JSON)
4. Lancement de la suite de tests Pytest
5. Vérification que le RMSE reste sous le seuil défini (< 1.0)

---

## Monitoring

`data/model_metrics.json` est généré à chaque entraînement et expose :

```json
{
  "rmse": 0.7993,
  "coverage_pct": 3.39,
  "n_users": 200947,
  "n_movies": 87585,
  "n_popular_films": 16034,
  "min_ratings_filter": 50,
  "reg_param": 0.1,
  "rank": 20,
  "n_recommendations": 2009470,
  "trained_at": "2026-04-22T00:05:56.726527"
}
```

Ces métriques sont chargées dynamiquement par le dashboard à chaque ouverture de l'application.

**Stratégie de détection du Data Drift :**
La distribution des notes et la popularité des genres sont mesurées périodiquement. Un écart significatif par rapport aux statistiques d'entraînement déclenche un réentraînement via `src/train_and_export.py`.

---

## Références

- [MovieLens Dataset — GroupLens](https://grouplens.org/datasets/movielens/)
- [Apache Spark MLlib — ALS](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context.* ACM TIIS.
