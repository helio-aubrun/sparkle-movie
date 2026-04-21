FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn pandas pydantic httpx scikit-learn

COPY api/ ./api/
COPY data/recommendations.csv ./data/recommendations.csv
COPY data/model_metrics.json  ./data/model_metrics.json
COPY data/profiles.json       ./data/profiles.json
COPY data/movies.csv          ./data/movies.csv

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
