FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY api/ ./api/
COPY run_pipeline.py .
COPY data/raw/ingredient_taste_lexicon.csv ./data/raw/ingredient_taste_lexicon.csv
COPY data/raw/ingredient_aliases.csv ./data/raw/ingredient_aliases.csv
COPY data/dataset_manifest.json ./data/dataset_manifest.json
COPY data/processed/recipe_taste_features.csv ./data/processed/recipe_taste_features.csv
COPY models/taste_model_weights.json ./models/taste_model_weights.json
COPY results/ ./results/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
