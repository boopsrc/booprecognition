# 📡 booprecognition: exemplos de `curl`

Todos os comandos da API HTTP, com o que enviar e o que esperar de volta.

## Variáveis usadas nos exemplos

```bash
export BOOP_URL=http://localhost:8000
export BOOP_CLIENT_ID=boop-admin
export BOOP_CLIENT_SECRET=troque-este-segredo   # valor de BOOP_CLIENT_ADMIN_SECRET
```

> Requisitos: API rodando (`.venv/Scripts/booprecognition serve`), `BOOP_JWT_SECRET` e o segredo do cliente definidos no `.env`, e um cliente cadastrado em `auth.clients` no `config.yaml`.

---

## 1. 🔑 Obter token: `POST /oauth/token`

Grant OAuth 2.0 `client_credentials`. O corpo vai como `application/x-www-form-urlencoded`.

### 1.1 Credenciais via Basic auth (recomendado)

```bash
curl -s -X POST "$BOOP_URL/oauth/token" \
  -u "$BOOP_CLIENT_ID:$BOOP_CLIENT_SECRET" \
  -d grant_type=client_credentials
```

### 1.2 Credenciais no corpo do formulário

```bash
curl -s -X POST "$BOOP_URL/oauth/token" \
  -d grant_type=client_credentials \
  -d client_id="$BOOP_CLIENT_ID" \
  -d client_secret="$BOOP_CLIENT_SECRET"
```

### 1.3 Pedir só alguns escopos

Separe os escopos por espaço. Só é possível pedir escopos que o cliente tem em `auth.clients[].scopes`.

```bash
curl -s -X POST "$BOOP_URL/oauth/token" \
  -u "$BOOP_CLIENT_ID:$BOOP_CLIENT_SECRET" \
  -d grant_type=client_credentials \
  --data-urlencode "scope=identify persons:write"
```

**Resposta `200`:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "identify persons:write"
}
```

### 1.4 Guardar o token numa variável

Requer `jq`:

```bash
export BOOP_TOKEN=$(curl -s -X POST "$BOOP_URL/oauth/token" \
  -u "$BOOP_CLIENT_ID:$BOOP_CLIENT_SECRET" \
  -d grant_type=client_credentials | jq -r .access_token)
```

Sem `jq` (Python):

```bash
export BOOP_TOKEN=$(curl -s -X POST "$BOOP_URL/oauth/token" \
  -u "$BOOP_CLIENT_ID:$BOOP_CLIENT_SECRET" \
  -d grant_type=client_credentials | py -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
```

**Erros (formato RFC 6749):**

| Status | `error` | Quando |
|---|---|---|
| `400` | `unsupported_grant_type` | `grant_type` diferente de `client_credentials` |
| `400` | `invalid_scope` | escopo pedido não permitido para o cliente |
| `401` | `invalid_client` | `client_id`/`client_secret` errados ou ausentes |

```json
{ "error": "invalid_client", "error_description": "Credenciais do cliente inválidas" }
```

---

## 2. 🔍 Identificar pessoa: `POST /identify`

**Escopo:** `identify` · **Corpo:** multipart com o campo `image`.

```bash
curl -s -X POST "$BOOP_URL/identify" \
  -H "Authorization: Bearer $BOOP_TOKEN" \
  -F image=@foto.jpg
```

**Resposta `200` (pessoa encontrada):**

```json
{ "identification": 482913057, "precision": 0.87, "history": [213, 245, 345, 634, 346] }
```

**Resposta `200` (ninguém reconhecido):** `null`

Quando a pessoa é encontrada, o embedding novo entra no histórico dela (máx. 10, FIFO).

---

## 3. ➕ Cadastrar pessoa: `POST /persons`

**Escopo:** `persons:write` · **Corpo:** multipart com o campo `image`.

```bash
curl -s -i -X POST "$BOOP_URL/persons" \
  -H "Authorization: Bearer $BOOP_TOKEN" \
  -F image=@foto.jpg
```

**Resposta `201`** (com o header `Location: /persons/482913057`):

```json
{ "id": 482913057, "identification": 482913057, "precision": 1.0, "history": [213], "created": true }
```

Guarde o `id`: é ele que identifica a pessoa em `/identify` e em `DELETE /persons/{id}`.

| Status | Significado |
|---|---|
| `201` | pessoa criada (`created: true`) |
| `200` | a pessoa já existia (`created: false`) e o ID existente é devolvido |

---

## 4. 🗑️ Remover pessoa: `DELETE /persons/{id}`

**Escopo:** `persons:delete`.

```bash
curl -s -X DELETE "$BOOP_URL/persons/482913057" \
  -H "Authorization: Bearer $BOOP_TOKEN"
```

Apaga a pessoa e todos os embeddings dela. Devolve `404` se o ID não existir.

---

## 5. ⚠️ Erros comuns nas rotas protegidas

| Status | `detail` | Causa |
|---|---|---|
| `400` | `Imagem vazia` | arquivo enviado sem conteúdo |
| `401` | `Token de acesso ausente` | faltou o header `Authorization: Bearer ...` |
| `401` | `Token inválido` | assinatura, emissor ou audiência errados, ou token malformado |
| `401` | `Token expirado` | passou de `expires_in`: peça um token novo |
| `403` | `Escopo insuficiente: ...` | o token não tem o escopo da rota |
| `404` | — | pessoa não encontrada |
| `422` | — | nenhum rosto detectado ou imagem inválida |

Para ver o desafio `WWW-Authenticate` devolvido em `401`/`403`, use `-i`:

```bash
curl -s -i -X POST "$BOOP_URL/identify" -F image=@foto.jpg
```

---

## 6. 🔁 Fluxo completo

```bash
BOOP_TOKEN=$(curl -s -X POST "$BOOP_URL/oauth/token" -u "$BOOP_CLIENT_ID:$BOOP_CLIENT_SECRET" -d grant_type=client_credentials | jq -r .access_token)
PERSON_ID=$(curl -s -X POST "$BOOP_URL/persons" -H "Authorization: Bearer $BOOP_TOKEN" -F image=@foto1.jpg | jq -r .id)
curl -s -X POST "$BOOP_URL/identify" -H "Authorization: Bearer $BOOP_TOKEN" -F image=@foto2.jpg
curl -s -X DELETE "$BOOP_URL/persons/$PERSON_ID" -H "Authorization: Bearer $BOOP_TOKEN"
```

> 🪟 **PowerShell:** use `curl.exe` (não o alias `curl`), troque `export VAR=...` por `$env:VAR = "..."` e `\` no fim da linha por `` ` ``.
