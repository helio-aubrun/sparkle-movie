"""
Entraîne le modèle ALS sur MovieLens 32M et exporte les recommandations.
Usage : python src/train_and_export.py
Sortie : data/recommendations.csv  (userId, rank, movieId, title, score)
         data/model_metrics.json   (RMSE, couverture, date)
"""
import os
import sys
import json
import datetime

os.environ["JAVA_HOME"] = "C:/Program Files/Microsoft/jdk-21.0.10.7-hotspot"
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# ── Constantes ────────────────────────────────────────────────────────────────
DATA_DIR       = "ml-32m"
OUTPUT_RECS    = "data/recommendations.csv"
OUTPUT_METRICS = "data/model_metrics.json"
N_RECS         = 10
RMSE_THRESHOLD = 1.0

# ── Session Spark ─────────────────────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("SparkleMovie-Train")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
print("Session Spark démarrée.")

# ── Chargement ────────────────────────────────────────────────────────────────
df_ratings = spark.read.csv(f"{DATA_DIR}/ratings.csv", header=True, inferSchema=True)
df_movies  = spark.read.csv(f"{DATA_DIR}/movies.csv",  header=True, inferSchema=True)

df_ratings = (
    df_ratings
    .na.drop(subset=["userId", "movieId", "rating"])
    .dropDuplicates(["userId", "movieId"])
    .filter("rating >= 0 AND rating <= 5")
)
df_movies = df_movies.na.drop(subset=["movieId", "title"]).dropDuplicates(["movieId"])
print(f"Données chargées : {df_ratings.count():,} notes — {df_movies.count():,} films.")

# ── Entraînement ALS ──────────────────────────────────────────────────────────
train, test = df_ratings.randomSplit([0.8, 0.2], seed=42)

als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    rank=10,
    maxIter=10,
    regParam=0.1,
    nonnegative=True,
    implicitPrefs=False,
    coldStartStrategy="drop",
)
print("Entraînement ALS en cours...")
model = als.fit(train)

# ── Évaluation ────────────────────────────────────────────────────────────────
predictions = model.transform(test)
evaluator   = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse        = evaluator.evaluate(predictions)
print(f"RMSE = {rmse:.4f}")

if rmse > RMSE_THRESHOLD:
    print(f"ERREUR : RMSE {rmse:.4f} dépasse le seuil {RMSE_THRESHOLD}. Export annulé.")
    spark.stop()
    sys.exit(1)

# ── Export des recommandations ────────────────────────────────────────────────
print(f"Génération des top-{N_RECS} recommandations pour tous les utilisateurs...")
all_recs = model.recommendForAllUsers(N_RECS)

recs_flat = (
    all_recs
    .select("userId", F.posexplode("recommendations").alias("rank", "rec"))
    .select(
        F.col("userId"),
        (F.col("rank") + 1).alias("rank"),
        F.col("rec.movieId").alias("movieId"),
        F.col("rec.rating").alias("score"),
    )
    .join(df_movies.select("movieId", "title", "genres"), on="movieId", how="left")
    .select("userId", "rank", "movieId", "title", "genres", "score")
    .orderBy("userId", "rank")
)

os.makedirs("data", exist_ok=True)
recs_pd = recs_flat.toPandas()
recs_pd.to_csv(OUTPUT_RECS, index=False)
print(f"Recommandations exportées → {OUTPUT_RECS} ({len(recs_pd):,} lignes)")

# ── Export des métriques ──────────────────────────────────────────────────────
recommended_movies = recs_flat.select("movieId").distinct().count()
total_movies       = df_movies.count()
coverage           = round(recommended_movies / total_movies * 100, 2)

metrics = {
    "rmse":            round(rmse, 4),
    "coverage_pct":    coverage,
    "n_users":         df_ratings.select("userId").distinct().count(),
    "n_movies":        int(total_movies),
    "n_recommendations": len(recs_pd),
    "trained_at":      datetime.datetime.now().isoformat(),
}
with open(OUTPUT_METRICS, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Métriques exportées → {OUTPUT_METRICS}")
print(json.dumps(metrics, indent=2))

spark.stop()
print("Terminé.")
