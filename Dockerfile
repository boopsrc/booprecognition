FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY config.yaml ./
RUN pip install --no-cache-dir ".[all]"

# Baixa o modelo ArcFace no build para evitar latência na primeira requisição
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', allowed_modules=['detection','recognition']).prepare(ctx_id=0)"

EXPOSE 8000
CMD ["booprecognition", "serve", "--port", "8000"]
