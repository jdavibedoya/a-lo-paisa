"""Bootstrap del entorno: carga el .env y expone secretos, rutas y require().

Solo plomería transversal. La config de cada modelo vive en SU módulo (STT en
transcribe.py, TTS en synthesize.py, LLM en llm.py); la lista completa de variables de
entorno está documentada en .env.example.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Raíz del proyecto, calculada explícitamente (no depende del directorio de ejecución):
# este archivo es src/a_lo_paisa/config.py, así que subimos tres niveles.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Carga el .env al importar este módulo (por eso otros módulos lo importan, aunque sea
# por efecto). override=False: una variable ya presente en el entorno real gana sobre el
# .env (útil en Spaces, que inyectan secretos por entorno).
load_dotenv(dotenv_path=ENV_PATH, override=False)

# Secretos: pueden ser None (no toda etapa los necesita); se validan bajo demanda con
# require(). HF_TOKEN es opcional (Chatterbox baja de un repo público; evita rate limits).
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")


def require(name: str) -> str:
    """Devuelve una variable de entorno obligatoria, o falla con un mensaje accionable."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Falta la variable de entorno obligatoria '{name}'. Copia .env.example a .env y rellena '{name}'."
        )
    return value
