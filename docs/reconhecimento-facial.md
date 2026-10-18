# Como funciona o reconhecimento facial no booprecognition

Este documento explica, passo a passo, o que acontece com uma foto desde que ela
chega na API até a resposta `identification` / `precision` / `history`.

> Versão visual: [reconhecimento-facial.html](reconhecimento-facial.html)

---

## Visão geral

```
foto ──► decodificar ──► detectar rosto ──► checar qualidade ──► alinhar 112×112
     ──► embedding (ArcFace, + espelhado, + ensemble) ──► busca KNN no banco
     ──► votação + travas ──► resposta  (e, se muito parecido, salva no histórico)
```

A foto **nunca é salva**. Só o vetor numérico do rosto (embedding) vai para o banco.

---

## Modelos usados

Todos vêm do [InsightFace](https://github.com/deepinsight/insightface) e rodam via ONNX Runtime (CPU por padrão).

| Pacote | Detector | Reconhecimento | Treino | Vetor |
|---|---|---|---|---|
| `buffalo_l` (padrão) | SCRFD-10G (`det_10g.onnx`) | ArcFace ResNet-50 (`w600k_r50.onnx`) | WebFace600K | 512-d |
| `antelopev2` (ensemble opcional) | SCRFD-10G | ArcFace ResNet-100 (`glintr100.onnx`) | Glint360K | 512-d |

- **SCRFD** encontra os rostos e devolve, para cada um, a caixa (bbox), uma
  confiança (`det_score`) e 5 pontos de referência: olho esquerdo, olho direito,
  nariz e os dois cantos da boca.
- **ArcFace** transforma o rosto alinhado num vetor de 512 números. Ele foi
  treinado com *additive angular margin loss*: fotos da mesma pessoa ficam com
  ângulo pequeno entre os vetores, pessoas diferentes ficam afastadas. Por isso
  comparamos com **similaridade de cosseno** (1 = idêntico, ~0 = sem relação).

O modelo é configurado em `config.yaml`:

```yaml
setup-alg:
  encoder_models: [buffalo_l]               # um modelo, 512-d
  # encoder_models: [buffalo_l, antelopev2] # ensemble, 1024-d
```

---

## Passo a passo

### 1. Decodificação e tamanho
`InsightFaceEncoder._decode` lê os bytes com OpenCV (já aplicando a orientação
EXIF). Depois `_normalize_size` reduz fotos com lado maior que **1600 px** e
amplia imagens com lado menor que **320 px**.

### 2. Detecção com tentativas progressivas
O detector roda em várias versões da imagem e para na primeira que acha um rosto:

1. imagem original (confiança mínima `0.5`);
2. com borda de 25% (rostos em close encostados na borda);
3. contraste equalizado (CLAHE, para fotos escuras ou contraluz);
4. borda + contraste;
5. ampliada 2x (rostos pequenos);
6. rotações de 90°, 180° e 270° (EXIF errado).

Da 2ª tentativa em diante a confiança mínima cai para `0.3`. Entre os rostos
encontrados, fica o **maior** com pelo menos 40 px de lado.

### 3. Filtro de qualidade
Rostos ruins geram vetores "genéricos", que parecem com todo mundo. Por isso o
encoder recusa (erro `LowQualityFaceError`):

- `det_score < min_det_score` (padrão `0.6`): rosto borrado, artificial ou parcial;
- rosto de perfil: mede o deslocamento horizontal do nariz em relação ao centro
  dos olhos, dividido pela distância entre os olhos. Acima de `max_yaw_ratio`
  (padrão `0.45`) é recusado.

### 4. Alinhamento
Com os 5 pontos, `face_align.norm_crop` aplica uma transformação de similaridade
que leva olhos, nariz e boca para posições fixas num recorte de **112×112**.
Isso é o que o ArcFace espera; sem alinhar, a precisão cai muito.

### 5. Embedding
Para cada modelo configurado:

1. calcula o vetor do recorte **e do recorte espelhado** e soma os dois
   (*test-time augmentation*, deixa o vetor mais estável);
2. normaliza o vetor (norma L2 = 1);
3. multiplica por `1/√N` (N = número de modelos).

Os vetores dos modelos são concatenados. Com esse peso, o cosseno do vetor final
é exatamente a **média dos cossenos de cada modelo**: um rosto só passa se os
dois modelos concordarem.

### 6. Busca KNN
O vetor vai para o índice vetorial do banco, que devolve os `knn_k` (padrão 5)
embeddings mais parecidos, cada um com `person_id` e similaridade:

| Banco | Busca |
|---|---|
| Firestore | `find_nearest` (vector search nativo) |
| MongoDB | `$vectorSearch` (Atlas, HNSW) ou força bruta em Python |
| SQLite | numpy em memória ou extensão `sqlite-vec` |

### 7. Votação e travas (`KnnClassifier`)
1. descarta vizinhos com similaridade `< min_neighbor_similarity` (`0.4`);
2. agrupa por pessoa; vence quem tem a **maior soma** de similaridades;
3. `precision = média das similaridades do vencedor × (votos dele / votos válidos)`;
4. rejeita se `precision < match_threshold` (`0.5`);
5. rejeita se o vizinho mais próximo do vencedor for `< min_top_similarity` (`0.55`);
6. rejeita se o vencedor não superar a 2ª pessoa por `min_margin` (`0.08`).

As travas 5 e 6 existem para casos como avatares/personagens de jogo: vários
rostos medianamente parecidos não somam um reconhecimento.

### 8. Resposta e aprendizado
- **Identificar** (`IdentifyPerson`): devolve a pessoa. O rosto novo só entra no
  histórico dela se a similaridade for `>= update_threshold` (`0.65`). O
  histórico é FIFO com até `max_embeddings_person` (10) vetores.
- **Cadastrar** (`RegisterPerson`): se o rosto já é reconhecido, ou se o vizinho
  mais próximo tem similaridade `>= duplicate_threshold` (`0.45`), devolve a
  pessoa existente em vez de criar outra. Senão gera um ID aleatório de 9
  dígitos e salva o primeiro embedding.

Exemplo de resposta:

```json
{ "identification": 482913057, "precision": 0.87, "history": [213, 245, 345] }
```

---

## Parâmetros (`setup-alg`)

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `knn_k` | 5 | vizinhos consultados |
| `min_neighbor_similarity` | 0.4 | abaixo disso o vizinho não vota |
| `match_threshold` | 0.5 | precision mínima |
| `min_top_similarity` | 0.55 | melhor vizinho do vencedor |
| `min_margin` | 0.08 | distância para a 2ª pessoa |
| `update_threshold` | 0.65 | mínimo para salvar no histórico |
| `duplicate_threshold` | 0.45 | trava de cadastro duplicado |
| `max_embeddings_person` | 10 | tamanho do histórico |
| `encoder_models` | `[buffalo_l]` | modelos do ensemble |
| `min_det_score` | 0.6 | qualidade mínima da detecção |
| `max_yaw_ratio` | 0.45 | limite de perfil |

**Mais falsos positivos?** Suba `min_top_similarity` (0.6) e `min_margin`, ou
ligue o ensemble. **Pessoa real não reconhecida?** Cadastre mais fotos dela ou
desça `min_top_similarity` com cuidado.

> Ao trocar `encoder_models` a dimensão muda (512 × N). Ajuste
> `sqlite.vector_search.dimension`, recrie o índice vetorial do Mongo/Firestore e
> recadastre as pessoas: vetores de modelos diferentes não são comparáveis.

## Limitações
- Não há anti-spoofing: uma foto de foto ou um render 3D muito realista não é
  bloqueado por um modelo dedicado, apenas pelas travas de similaridade.
- Os limites padrão são ponto de partida; calibre com fotos reais do seu uso.
