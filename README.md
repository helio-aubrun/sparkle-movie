# Sparkle Movie

Système de recommandation de films basé sur le dataset **MovieLens 32M**, développé avec **Apache Spark** et **PySpark MLlib**.

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
│   └── download_dataset.py      # Script de téléchargement du dataset MovieLens 32M
├── notebooks/
│   ├── 01_exploration.ipynb     # Analyse exploratoire des données (EDA)
│   └── 02_modelisation.ipynb    # Modélisation : ALS, Content-Based, KNN
├── src/                         # Scripts Python d'entraînement et de nettoyage
├── api/                         # API REST FastAPI
├── tests/                       # Tests automatisés Pytest
├── .github/
│   └── workflows/               # Configuration CI/CD
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

**Hyperparamètres ajustés :**
- `rank` : dimension des vecteurs de facteurs latents
- `regParam` : régularisation pour éviter l'overfitting
- `maxIter` : nombre d'itérations

**Évaluation :** RMSE (Root Mean Square Error)

**Avantages :** Scalable, natif Spark, très performant sur des données denses
**Limites :** Aveugle sur les films froids (< 5 notes)

---

### 2. Content-Based Filtering

Crée des profils de films à partir de leurs genres, puis recommande des films similaires à ceux qu'un utilisateur a aimés. Utilise :
- **TF-IDF** pour vectoriser les genres
- **Similarité cosinus** pour mesurer la proximité entre films

**Avantages :** Résout le cold start côté films, indépendant des notes
**Limites :** Limité aux genres disponibles, ne découvre pas de nouveaux profils

---

### 3. KNN — K-Nearest Neighbors

Trouve les K utilisateurs dont le profil de notes est le plus proche de l'utilisateur cible, puis recommande les films bien notés par ces voisins.

**Avantages :** Intuitif, efficace pour les utilisateurs avec peu d'historique
**Limites :** Coûteux à l'échelle, sensible aux outliers

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

**Choix retenu : ALS** — justifié par la taille du dataset (32M lignes), l'absence de GPU, et l'intégration native dans l'écosystème Spark déjà utilisé.

### Pourquoi Apache Spark ?

Sur 32 millions de notes, les alternatives présentent des limitations critiques :

| Outil | Limite |
|-------|--------|
| Pandas | Charge le dataset en RAM — crash ou swap sur une machine standard |
| Scikit-learn | Mono-thread, pas conçu pour du distribué |
| SQL classique | Pas de MLlib, pas de traitement distribué natif |
| **Spark** | Distribué, MLlib intégré, traitement en mémoire partitionné |

### Conformité RGPD

Le dataset MovieLens est **anonymisé** : les `userId` sont des identifiants numériques sans lien avec des données personnelles réelles. Toutefois, en contexte de production réelle :

- Les recommandations personnalisées constituent un **traitement de données à caractère personnel** au sens de l'article 4 du RGPD
- L'utilisateur doit être informé de l'usage de ses données (article 13)
- Un droit d'opposition au profilage doit être prévu (article 21)
- Les modèles entraînés sur des données utilisateurs doivent être supprimables sur demande (**droit à l'oubli**, article 17)

---

## Installation

### Prérequis

- Python 3.11+
- Java 21 (requis par PySpark)

```bash
# Windows — installer Java via winget
winget install Microsoft.OpenJDK.21
```

### Dépendances Python

```bash
pip install -r requirements.txt
```

### Télécharger le dataset

```bash
python data/download_dataset.py
```

### Lancer les notebooks

```bash
jupyter notebook notebooks/
```

---

## API REST

L'API expose les recommandations via un endpoint REST documenté avec Swagger.

### Démarrer l'API

```bash
uvicorn api.main:app --reload
```

### Endpoint principal

```
GET /recommend/{user_id}
```

**Réponse :**
```json
{
  "user_id": 42,
  "recommendations": [
    { "movieId": 318, "title": "Shawshank Redemption, The (1994)", "score": 4.87 },
    { "movieId": 858, "title": "Godfather, The (1972)", "score": 4.82 }
  ]
}
```

Documentation Swagger disponible sur `http://localhost:8000/docs`

---

## Tests

```bash
pytest tests/ -v
```

Les tests couvrent :
- Validation des types de données en entrée
- Vérification que le RMSE reste sous le seuil défini
- Tests de l'API (statut, format de réponse)

---

## Monitoring

Le système de logging enregistre :
- Les prédictions effectuées (userId, films recommandés, timestamp)
- Les temps de réponse de l'API

**Stratégie de détection du Data Drift :**
La distribution des notes et la popularité des genres sont mesurées périodiquement. Un écart significatif par rapport aux statistiques d'entraînement (distribution des notes, médiane des votes par film) déclenche un réentraînement du modèle.

---

## Conclusion

Les trois approches se complètent :

- **ALS** couvre efficacement les 200 948 utilisateurs avec un historique riche
- **Content-Based** résout le cold start pour les 48% de films peu notés
- **KNN** affine les recommandations pour les profils atypiques

L'évaluation comparative (RMSE, précision, couverture) démontre qu'aucune approche seule ne suffit : un système hybride combinant ALS et Content-Based offre la meilleure couverture du catalogue tout en maintenant une précision élevée.

---

## Références

- [MovieLens Dataset — GroupLens](https://grouplens.org/datasets/movielens/)
- [Apache Spark MLlib — ALS](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context.* ACM TIIS.
