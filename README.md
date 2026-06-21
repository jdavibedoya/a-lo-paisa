---
title: A lo Paisa
emoji: 🎙️
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Tu voz, pero hablando a lo paisa de Antioquia.
---

# A lo Paisa 🎙️

Pipeline de voz (no en tiempo real) que **re-dice un audio en español paisa de Antioquia,
con tu voz clonada**:

```
audio → STT → (normalización a español, si hace falta) → reescritura a paisa (RAG) → TTS voz clonada → audio
```

Corre de punta a punta (voz a voz), con UI web (Gradio) y empaquetado para **Hugging Face
Spaces con Docker** (CPU).

## Cómo funciona

El usuario fija **tres diales** antes de hablar:

- **Idioma de entrada** — `español` (whisper fuerza `es`, no se normaliza) · `inglés`
  (fuerza `en`, se normaliza a español) · `otro` (autodetecta y normaliza).
- **Exageración** `1–3` — gobierna **el texto Y la voz** (a más exageración, más jerga y
  más expresividad).
- **Registro** — `urbano` (parlache de Medellín) · `montañero` (rural/tradicional).

Las etapas:

1. **STT** con faster-whisper (int8, CPU), dirigido por el dial de idioma.
2. **Normalización** a español neutro (Gemini) — solo si el dial es `inglés`/`otro`.
3. **Reescritura a paisa** con Gemini, anclada por **RAG semántico** sobre un glosario
   curado (embeddings + similitud coseno).
4. **TTS** con voz clonada: Chatterbox V3 + Language Pack es-MX (LatAm).

Un **"portero"** garantiza que si el texto falla, **nunca se sintetiza un error**.

## Requisitos

- [uv](https://docs.astral.sh/uv/) como gestor de paquetes · Python 3.11 (fijado en
  `.python-version`; uv lo instala solo).
- Una **API key de Gemini** (capa gratuita): la usan la normalización, la reescritura y los
  embeddings (el STT no la necesita).
- Un audio de **tu voz** en `data/voice_reference.aif` (para clonar el timbre).

## Puesta en marcha

```bash
uv python install 3.11                       # uv gestiona el Python; no toca el del sistema
uv sync                                      # entorno + deps (la 1ª vez baja torch/Chatterbox)
cp .env.example .env                         # rellená GEMINI_API_KEY (HF_TOKEN opcional)
uv run python scripts/build_embeddings.py    # índice del glosario (1 vez, y cada vez que lo edites)
```

## Uso

**Interfaz web** (Gradio):

```bash
uv run python app.py        # abre en http://localhost:7860
```

**Pipeline por CLI** (voz → voz):

```bash
uv run python scripts/cli.py --audio tu_voz.wav --idioma español
uv run python scripts/cli.py --idioma inglés --exageracion 3 --registro urbano   # graba del micrófono
```

**Etapas sueltas**:

```bash
uv run python -m a_lo_paisa.transcribe audio.wav   # solo STT
uv run python -m a_lo_paisa.paisa_transform        # solo la reescritura (texto)
uv run python -m a_lo_paisa.synthesize             # solo TTS
```

## Estructura

| Ruta | Qué hace |
|------|----------|
| `src/a_lo_paisa/config.py` | Bootstrap del entorno: `.env`, secretos, rutas, `require()`. |
| `src/a_lo_paisa/llm.py` | Cliente Gemini + IDs de modelo + `TransformacionError`. |
| `src/a_lo_paisa/transcribe.py` | STT con faster-whisper (int8, CPU). |
| `src/a_lo_paisa/retrieve.py` | RAG: glosario + embeddings + `lookup_paisa()`. |
| `src/a_lo_paisa/paisa_transform.py` | Reescritura a paisa (Gemini + RAG) + normalización a español. |
| `src/a_lo_paisa/synthesize.py` | TTS con voz clonada (Chatterbox V3 + Language Pack es-MX). |
| `src/a_lo_paisa/pipeline.py` | Orquestación reusable sin UI (la comparten CLI y app). |
| `scripts/build_embeddings.py` | (Offline) construye el índice de embeddings del glosario. |
| `scripts/cli.py` | Entrypoint CLI del pipeline completo. |
| `app.py` | UI web (Gradio). |
| `Dockerfile` | Imagen del Space (uv, CPU). |
| `data/paisa_glossary.json` | Glosario paisa curado (neutro → términos con exageración, ejemplos, notas). |

## Decisiones de diseño

- **RAG semántico, no léxico.** La búsqueda por significado recupera "amigo"/"dinero" desde
  "mi compañero anda sin billete", cosa que el match léxico no logra. (Lo verifiqué con un
  experimento comparando ambos enfoques.)
- **Auto-reflexión: probada y descartada.** Construí un paso de auto-crítica/corrección, lo
  A/B-testeé y validé el crítico; el single-pass ya cumplía las reglas, así que no shipeé la
  complejidad extra.
- **Dispositivos.** STT y TTS en CPU (el Space es CPU; con `"auto"` tomarían una GPU NVIDIA
  si la hubiera). En Apple Silicon el TTS va en CPU igual: MPS tiene un bug de torchaudio.
- **Dos temperaturas.** Normalización `0.2` (traducción fiel, casi determinista);
  transformación `0.7` (paisa con chispa pero controlada).

## Modelos

| Etapa | Modelo |
|-------|--------|
| STT | faster-whisper `small`, int8 |
| Normalización / reescritura | Gemini `2.5-flash` |
| Embeddings (RAG) | `gemini-embedding-001`, dim 768 |
| TTS | Chatterbox V3 (multilingüe) + Language Pack es-MX (LatAm) |
