# Imagen del Hugging Face Space (SDK docker, CPU). Usa uv + uv.lock para deps 100%
# reproducibles, y HORNEA los modelos en la imagen (prefetch) para que el runtime no
# descargue nada: cold-start rápido y sin re-bajar los pesos cuando el Space despierta.

# Imagen oficial de uv con Python 3.11 (Debian slim). Trae uv preinstalado.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# git: chatterbox-tts y resemble-perth se instalan DESDE GitHub.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces corre el contenedor como usuario NO-root (uid 1000).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    UV_LINK_MODE=copy
# Gradio escucha en 0.0.0.0:7860 (host/puerto que expone el Space); por env, no hardcodeado.
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /home/user/app

# 1) Dependencias (capa cacheable: solo cambia si cambian los manifests).
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# 2) Paquete + PREFETCH de los pesos (TTS + STT) en la imagen. Va ANTES de copiar el
#    resto para que tweaks en app.py/data NO re-disparen la descarga de modelos.
COPY --chown=user README.md ./
COPY --chown=user src/ ./src/
RUN uv sync --frozen
COPY --chown=user scripts/prefetch.py ./scripts/prefetch.py
RUN uv run python scripts/prefetch.py

# 3) El resto del proyecto (app.py, data, scripts).
COPY --chown=user . .

EXPOSE 7860
# app.py lanza Gradio en 0.0.0.0:7860 (lo que el Space espera).
CMD ["uv", "run", "python", "app.py"]
