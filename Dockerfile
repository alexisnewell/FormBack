# ---- 1. build the frontend ----
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- 2. runtime: API + built UI in one image ----
FROM python:3.11-slim

# opencv (pulled in by mediapipe) needs these shared libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY src/ src/
COPY models/ models/
COPY NOTICE ./
COPY THIRD_PARTY_LICENSES/ THIRD_PARTY_LICENSES/
COPY --from=frontend /app/frontend/dist frontend/dist

ENV MPLBACKEND=Agg
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
