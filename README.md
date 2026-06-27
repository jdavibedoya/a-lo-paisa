---
title: A lo Paisa
emoji: ⛰️🫓
colorFrom: green
colorTo: white
sdk: docker
app_port: 7860
pinned: false
short_description: "Voice pipeline with RAG: STT → Paisa transformation → TTS with cloned voice."
---

# ⛰️🫓 A lo Paisa
> Voice-to-Voice Pipeline for Antioquian Spanish

[![Hugging Face Spaces](https://img.shields.io/badge/HF-A_lo_Paisa-FFD21E?logo=huggingface)](REEMPLAZAR_CON_LINK_DE_HUGGINGFACE)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?logo=pytorch&logoColor=white)
![Google GenAI](https://img.shields.io/badge/LLM-Google_GenAI-4285F4?logo=google&logoColor=white)
![Gradio](https://img.shields.io/badge/UI-Gradio_6.8-FF7C00?logo=gradio&logoColor=white)
![Faster Whisper](https://img.shields.io/badge/STT-Faster_Whisper-10A37F)
![Chatterbox](https://img.shields.io/badge/TTS-Chatterbox_V3-000000)

End-to-end voice pipeline transforming speech audio into Paisa Spanish (Antioquia, Colombia) with a cloned voice.

<div align="center">
  <figure>
    <img src="assets/architecture_diagram.webp" alt="A lo Paisa - Architecture Workflow" width="100%">
    <figcaption>
      <b>Architecture Workflow</b><br>
      <small>Illustrated with ChatGPT</small>
    </figcaption>
  </figure>
</div>


## 🎧 Examples

| Dials&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Input | Output |
| :--- | :--- | :--- |
| *Idioma:* `español`<br>*Exageración:* `2`<br>*Registro:* `urbano` | [▶️](https://github.com/jdavibedoya/a-lo-paisa/raw/refs/heads/main/assets/examples/spanish_input.wav)<br><small>"Hoy es un día tranquilo en la montaña con algo de lluvia y un cielo gris me gusta correr temprano y escuchar música"</small> | [▶️](https://github.com/jdavibedoya/a-lo-paisa/raw/refs/heads/main/assets/examples/spanish_output.wav)<br><small>"Hoy está el día como tranquilito por acá en la montaña, con una agüita y el cielo medio gris. A mí me gusta, pues, correr bien tempranito y escuchar música."</small> |
| *Idioma:* `inglés`<br>*Exageración:* `2`<br>*Registro:* `montañero` | [▶️](https://github.com/jdavibedoya/a-lo-paisa/raw/refs/heads/main/assets/examples/english_input.wav)<br><small>"She can scoop these things into three red bags and we will go meet her Wednesday at the train station."</small> | [▶️](https://github.com/jdavibedoya/a-lo-paisa/raw/refs/heads/main/assets/examples/english_output.wav)<br><small>"Ella puede meter esas cositas en tres bolsas rojas y nos pillamos con ella el miércoles en la estación del tren, ¿sí o qué?"</small> |

---
## How it Works
### Dials
- *Idioma de entrada:* `español` | `inglés` | `otro`
- *Exageración:* `1` (suave) | `2` (cotidiano) | `3` (recargado)
- *Registro:* `urbano` (parlache) | `montañero` (rural / tradicional)

### Pipeline Stages
1. **Input Voice**
2. **STT (Speech-to-Text):** Automatically detects the language if set to `otro`.
3. **(Translation):** Handled by an LLM (triggered only if the *Idioma de entrada* dial is `inglés` or `otro`).
4. **Paisa Rewriting:** An LLM rewrites the text, enriched by **semantic RAG** against a curated glossary (Embeddings + Cosine Similarity). This stage is modulated by the *Exageración* and *Registro* dials.
5. **TTS (Text-to-Speech):** Synthesizes the text with a cloned voice. The *Exageración* dial influences the model's prosodic variance and acoustic expressiveness.
6. **Output Voice**

## Setup & Installation
```bash
uv python install 3.11                       # Python 3.11 for the virtual environment
uv sync                                      # Create venv and install dependencies
cp .env.example .env                         # Setup your credentials
```

## Usage
### [Web UI (Gradio) - Hugging Face Spaces](REEMPLAZAR_CON_LINK_DE_HUGGINGFACE)

<div align="center">
  <img src="assets/gradio_ui.webp" alt="Gradio Web UI Screenshot" width="600">
</div>

### Web UI (Gradio) - Local:
```bash
uv run python app.py        # Runs on http://localhost:7860
```

### CLI:
```bash
uv run python scripts/cli.py --audio input_voice.wav --idioma español
```

---
## Architectural Decisions
- **Pre-processing:** A speech enhancement model for background noise and reverberation was initially considered. However, testing revealed that `faster-whisper` is sufficiently robust on its own.
- **Workflow over Agentic Loop:** A/B testing showed that autonomous processes (self-critique loops or autonomous tool-calling for RAG) did not yield tangible improvements. A sequential workflow was chosen for reliability and lower latency.
- **Data Curation:** Informed by known dictionaries (e.g. Julio C. Jaramillo R., Medellín.travel) and social polls, this compact, manually curated glossary consistently improves rewriting performance during testing.
- **Semantic RAG:** Retrieval relies on vector embeddings rather than strict keyword lookups to ensure accurate contextual matching.
- **Model Agnosticism:** Gemini was selected for its balance of performance, free-tier availability, and included embedding model. Still, the architecture is model-agnostic; testing another LLM only requires updating the `LLM_API_KEY` and the `provider.py` module.
- **Asymmetric Temperatures:** The translation step runs at `0.2` (prioritizing deterministic accuracy), while the Paisa transformation step runs at `0.7` (encouraging creative and expressive phrasing).
- **TTS Package:** Using Chatterbox V3 for its Latin American Spanish capabilities. Since this version is currently unavailable on PyPI, the dependency is pulled directly from the Resemble GitHub repository.
- **Hardware Targeting:** Designed to run efficiently on CPU-only Hugging Face Free Spaces. The system automatically detects and leverages CUDA if available. Apple Silicon (MPS) falls back to CPU due to `torchaudio` constraints in the TTS module.
- **Cold Start Optimization:** Model weights are baked directly into the Docker image to eliminate downloads on runtime. Additionally, a background thread warms up the TTS model at startup to reduce first-request latency.

## Models
| Stage | Model |
|-------|-------|
| **STT** | faster-whisper `small`, int8 |
| **Translation / Rewriting** | Gemini `2.5-flash` |
| **Embeddings (RAG)** | `gemini-embedding-001`, dim 768 |
| **TTS** | Chatterbox V3 (Multilingual) + es-MX Language Pack |

## Project Structure
Each stage can also be executed as a standalone module. Check the docstrings or run `--help` on individual files for details.

| Path | Purpose |
|------|---------|
| `src/a_lo_paisa/config.py` | Environment bootstrap and configuration. |
| `src/a_lo_paisa/provider.py` | Model IDs and LLM client setup. |
| `src/a_lo_paisa/transcribe.py` | STT transcription. |
| `src/a_lo_paisa/retrieve.py` | RAG implementation. |
| `src/a_lo_paisa/paisa_transform.py` | LLM translation + LLM Paisa rewriting. |
| `src/a_lo_paisa/synthesize.py` | TTS synthesis with voice cloning. |
| `src/a_lo_paisa/pipeline.py` | Core end-to-end orchestrator. |
| `scripts/build_embeddings.py` | (Offline) Generates vector embeddings. |
| `scripts/cli.py` | Command-line entrypoint. |
| `scripts/prefetch.py` | Bakes model weights into the Docker image. |
| `app.py` | Gradio Web UI. |
| `Dockerfile` | Hugging Face Space image definition. |
| `data/paisa_glossary.json` | Curated Paisa dataset. |

---
**Note on language:** Code comments and docstrings are in **Spanish**.

*Developed by [David Bedoya](https://github.com/jdavibedoya)*
