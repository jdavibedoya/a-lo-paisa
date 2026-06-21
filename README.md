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

# A lo Paisa 🎙️🇨🇴

Pipeline de voz (no en tiempo real) que **re-dice un audio en español paisa de
Antioquia, con tu voz clonada**:

```
audio → STT → (normalización a español, si hace falta) → reescritura a paisa (RAG) → TTS voz clonada → audio
```

Construido por **etapas verificables**. Corre de punta a punta (voz a voz), tiene UI web
(Gradio) y está empaquetado para **Hugging Face Spaces con Docker** (CPU).

> **No es un agente autónomo**: es un **workflow determinista** (pasos fijos
> orquestados por código). El LLM no decide qué herramientas usar ni planifica.
> Probé un paso de auto-reflexión y, con datos, decidí no incluirlo (ver
> [`archive/`](archive/)).

## Cómo funciona

El usuario fija **tres diales** antes de hablar:

- **Idioma de entrada** — `español` (whisper fuerza `es`, no se normaliza) ·
  `inglés` (fuerza `en`, se normaliza a español) · `otro` (autodetecta y normaliza).
- **Exageración** `1–3` — gobierna **el texto Y la voz** (a más exageración, más
  jerga y más expresividad en el TTS).
- **Registro** — `urbano` (parlache de Medellín) · `montañero` (rural/tradicional).

Las etapas:

1. **STT** con faster-whisper (int8, CPU), dirigido por el dial de idioma.
2. **Normalización** a español neutro (Gemini) — solo si el dial es `inglés`/`otro`.
3. **Reescritura a paisa** con Gemini, anclada por **RAG semántico** sobre un
   glosario curado (embeddings + similitud coseno).
4. **TTS** con voz clonada: Chatterbox V3 + Language Pack es-MX (LatAm).

Un **"portero"** garantiza que si el texto falla, **nunca se sintetiza un error**: el
TTS solo corre si la transformación tuvo éxito.

## Requisitos

- [uv](https://docs.astral.sh/uv/) como gestor de paquetes · Python 3.11 (fijado en
  `.python-version`; uv lo instala solo).
- Una **API key de Gemini** (capa gratuita): STT no la necesita, pero la
  normalización, la reescritura y los embeddings sí.
- Un audio de **tu voz** en `data/voice_reference.aif` (para clonar el timbre en el TTS).

## Puesta en marcha

```bash
# 1. Python 3.11 (uv lo gestiona; no toca tu Python del sistema).
uv python install 3.11

# 2. Entorno + dependencias. La 1ª vez es una descarga grande (torch, Chatterbox
#    desde GitHub, etc.); genera .venv/ y uv.lock con versiones exactas.
uv sync

# 3. Secretos: copiá la plantilla y rellená GEMINI_API_KEY (HF_TOKEN es opcional).
cp .env.example .env

# 4. Construí el índice de embeddings del glosario (1 vez, y cada vez que lo edites).
uv run python scripts/build_embeddings.py
```

## Correr el pipeline completo (voz → voz)

```bash
# Español (no normaliza), montañero, exageración 2:
uv run python scripts/cli.py --audio tu_voz.wav --idioma español

# Inglés (fuerza 'en' y normaliza a español antes del agente):
uv run python scripts/cli.py --audio data/english_input.wav --idioma inglés --exageracion 3 --registro urbano
```

Si omitís `--audio`, **graba del micrófono** hasta que presiones Enter. La salida va a
`outputs/salida_pipeline.wav` (configurable con `--salida`).

### Probar etapas sueltas

```bash
uv run python -m a_lo_paisa.transcribe audio.wav   # solo STT
uv run python -m a_lo_paisa.paisa_transform        # solo el agente (texto)
uv run python -m a_lo_paisa.synthesize             # solo TTS (con voz_reference.aif)
```

`uv run` ejecuta dentro del entorno del proyecto sin activar el venv a mano.

## Estructura

| Ruta | Estado | Qué hace |
|------|--------|----------|
| `src/a_lo_paisa/config.py` | ✅ | Bootstrap del entorno: carga `.env`, secretos, rutas, `require()`. |
| `src/a_lo_paisa/llm.py` | ✅ | Cliente Gemini + IDs de modelo (generación/embeddings) + `TransformacionError`. |
| `src/a_lo_paisa/transcribe.py` | ✅ | STT con faster-whisper int8 en CPU, dirigido por el dial de idioma. |
| `src/a_lo_paisa/retrieve.py` | ✅ | RAG: glosario + embeddings + `lookup_paisa()` (búsqueda semántica). |
| `src/a_lo_paisa/paisa_transform.py` | ✅ | Reescritura a paisa (Gemini + RAG) + normalización a español. Workflow determinista. |
| `src/a_lo_paisa/synthesize.py` | ✅ | TTS con voz clonada (Chatterbox V3 + Language Pack es-MX LatAm). |
| `src/a_lo_paisa/pipeline.py` | ✅ | Orquestación reusable (sin UI): `audio_a_paisa()` = STT → normalización → paisa. |
| `scripts/build_embeddings.py` | ✅ | (Offline) construye el índice de embeddings del glosario. |
| `scripts/cli.py` | ✅ | Entrypoint CLI: graba/lee audio, llama a `pipeline`, presenta el texto y sintetiza (con el "portero"). |
| `data/paisa_glossary.json` | ✅ | Glosario paisa curado (neutro → términos con exageración, ejemplos, notas). |
| `app.py` | ⏳ | UI Gradio (etapa final, reusará `pipeline.audio_a_paisa()`). |
| `Dockerfile` | ⏳ | Imagen para el Hugging Face Space (etapa final). |

## Decisiones de diseño (el porqué)

- **Workflow, no agente.** El flujo es fijo y determinista; un agente autónomo
  agregaría costo e imprevisibilidad sin beneficio. `lookup_paisa()` quedó
  *tool-shaped* por si algún día una necesidad real de autonomía lo justifica.
- **RAG semántico, no léxico.** Lo medí (ver `archive/experimento_paisa.py`): la
  búsqueda por significado recupera "amigo"/"dinero" desde "mi compañero anda sin
  billete", cosa que el match léxico no logra.
- **Auto-reflexión: probada y descartada.** Construí un paso de crítica/corrección,
  lo A/B-testeé y validé el crítico (ver `archive/experimento_reflexion.py`): el
  single-pass ya cumple las reglas, así que no se justifica la complejidad extra.
- **Dispositivos.** STT en **CPU** a propósito (en HF Spaces la GPU es escasa y se
  reserva para el TTS, lo pesado). TTS en **CUDA** en producción y **CPU** en local
  (MPS de Apple Silicon tiene un bug de torchaudio que lo hace inservible).
- **Dos temperaturas.** Normalización `0.2` (traducción fiel, casi determinista);
  transformación `0.7` (paisa con chispa pero controlada).

## Modelos

| Etapa | Modelo |
|-------|--------|
| STT | faster-whisper `small`, cuantización int8 |
| Normalización / reescritura | Gemini `2.5-flash` (capa gratuita) |
| Embeddings (RAG) | `gemini-embedding-001`, dim 768 |
| TTS | Chatterbox V3 (multilingüe) + Language Pack es-MX LatAm |
