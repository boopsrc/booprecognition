FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY config.yaml ./
RUN pip install --no-cache-dir ".[all]"

# Baixa os modelos no build para evitar latência na primeira requisição:
# buffalo_l (SCRFD-10G + ArcFace R50) e antelopev2 (ArcFace R100)
RUN python -c "from booprecognition.infrastructure.encoders.insightface_encoder import InsightFaceEncoder; InsightFaceEncoder(model_names=['buffalo_l', 'antelopev2'])"

EXPOSE 8000
CMD ["booprecognition", "serve", "--port", "8000"]
