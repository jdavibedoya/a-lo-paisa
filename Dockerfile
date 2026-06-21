# Imagen del Hugging Face Space (SDK docker, CPU). Usa uv + uv.lock para instalar las deps
# 100% reproducibles: el mismo lock probado en local se instala acá.

# Imagen oficial de uv con Python 3.11 (Debian slim). Trae uv preinstalado.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# git: chatterbox-tts y resemble-perth se instalan DESDE GitHub, así que uv necesita
# git para clonarlos. Lo instalamos como root, antes de bajar de privilegios.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces corre el contenedor como usuario NO-root (uid 1000). Lo creamos y nos
# cambiamos a él; todo lo de abajo escribe en su HOME (escribible).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    # Caché de modelos de HF en el HOME del user. Los pesos de Chatterbox se descargan
    # en el PRIMER request y quedan cacheados mientras el contenedor viva.
    HF_HOME=/home/user/.cache/huggingface \
    # En contenedor, copiar en vez de symlinkear evita problemas entre filesystems.
    UV_LINK_MODE=copy

# Gradio escucha en 0.0.0.0:7860 dentro del contenedor (host/puerto que expone el Space).
# Lo fijamos por env (Gradio las respeta solo) en vez de hardcodear en app.py, así el
# launch() local sigue eligiendo un puerto libre sin chocar.
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /home/user/app

# 1) Solo los manifiestos primero => capa de DEPENDENCIAS cacheable: si después solo
#    cambia el código, Docker reusa esta capa y no reinstala todo. `--frozen` exige
#    que uv.lock esté en sync (instala las versiones EXACTAS del lock, reproducible).
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# 2) El resto del proyecto + instalar el paquete a_lo_paisa en el venv.
COPY --chown=user . .
RUN uv sync --frozen

EXPOSE 7860
# app.py lanza Gradio en 0.0.0.0:7860 (lo que el Space espera).
CMD ["uv", "run", "python", "app.py"]
