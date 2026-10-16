# booprecognition

Motor agnóstico de reconhecimento facial. Guarda **somente embeddings** (ArcFace, 512-d) no Firestore — nunca imagens — e identifica pessoas por KNN usando a busca vetorial nativa do Firestore.

## Serviços

| Serviço | HTTP | CLI | Comportamento |
|---|---|---|---|
| Identificar | `POST /identify` (multipart `image`) | `booprecognition identify foto.jpg` | Retorna a pessoa ou `null`. Se encontrada, salva o novo embedding no histórico (máx. 10, o mais antigo é substituído). |
| Cadastrar | `POST /persons` (multipart `image`) | `booprecognition register foto.jpg` | Se já existir, retorna a existente (`200`, `created: false`); senão cria com ID aleatório (`201`). |
| Remover | `DELETE /persons/{id}` | `booprecognition remove 123456789` | Remove a pessoa e todos os embeddings (`404` se não existir). |

Saída:
```json
{ "identification": 123123123, "precision": 0.87, "history": [213, 245, 345, 634, 346] }
```
- `precision` (0–1): similaridade média dos vizinhos vencedores × fração dos K votos.
- `history`: IDs dos embeddings vizinhos que sustentaram a decisão.

Erros: `422` sem rosto/imagem inválida, `404` pessoa inexistente.

## Arquitetura

```
src/booprecognition/
  domain/          entidades, value objects, KnnClassifier (Python puro)
  application/     portas (Protocols), DTOs, casos de uso
  infrastructure/  InsightFace, Firestore, in-memory, gerador de IDs
  interfaces/      FastAPI e CLI (frameworks ficam só aqui)
  config/          Settings (.env) e composition root (container.py)
```
Dependências apontam para dentro: domínio não conhece aplicação; aplicação não conhece infraestrutura nem framework. Trocar o modelo facial ou o banco = novo adaptador + `container.py`.

## Firestore

```
persons/{personId}                          identification, created_at, updated_at, embedding_ids[≤10]
persons/{personId}/embeddings/{embeddingId} embedding_id, person_id, vector (Vector 512), created_at
```

Crie o índice vetorial (uma vez por nome de coleção):
```bash
gcloud firestore indexes composite create --collection-group=embeddings --query-scope=COLLECTION_GROUP --field-config=field-path=vector,vector-config='{"dimension":"512","flat":"{}"}'
```

## Setup

```bash
py -m venv .venv
.venv/Scripts/python -m pip install -e ".[all,dev]"
cp .env.example .env   # configure credenciais e projeto
.venv/Scripts/booprecognition serve
```

`BOOP_STORAGE=memory` roda sem Firestore (dev local).

## Testes

```bash
.venv/Scripts/python -m pytest
```
Integração (Firestore + modelo reais, coleções `persons_it`/`embeddings_it`, exige o índice para `embeddings_it`):
```bash
.venv/Scripts/python -m pytest -m integration
```
Variáveis: `BOOP_TEST_FACE_A1`, `BOOP_TEST_FACE_A2` (mesma pessoa), `BOOP_TEST_FACE_B`.

## Docker

```bash
docker build -t booprecognition .
```
