"""Bootstrap del entorno: carga el .env y expone secretos, rutas y require().

La config de cada modelo vive en su módulo (STT en transcribe.py, TTS en synthesize.py,
EMBED y LLM en provider.py); la lista de variables de entorno está en .env.example.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Raíz del proyecto (desde este archivo, no del cwd).
ENV_PATH = PROJECT_ROOT / ".env"

# Carga el .env al importar este módulo.
# override=False: una variable ya presente en el entorno gana sobre el .env (útil en Spaces, que inyectan los secretos por entorno).
load_dotenv(dotenv_path=ENV_PATH, override=False)

# Secretos: se validan bajo demanda con require().
# HF_TOKEN es opcional (los pesos se bajan de repos públicos de HF; evita rate y bandwidth limits).
LLM_API_KEY: str | None = os.getenv("LLM_API_KEY")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")


def require(name: str) -> str:
    """Devuelve una variable de entorno o falla con un mensaje accionable."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Falta la variable de entorno '{name}'. Copiá .env.example a .env y asigná '{name}'."
        )
    return value
