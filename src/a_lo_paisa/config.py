"""Bootstrap del ENTORNO: carga el .env y expone secretos, rutas y require().

Foco deliberado: este módulo es SOLO la plomería de entorno/secretos. La config de
cada modelo vive en SU módulo —el STT en transcribe.py, el TTS en synthesize.py, el
LLM en llm.py—, donde es más localizable. Acá quedan únicamente las cosas
transversales: cargar el .env una sola vez, las claves, las rutas del proyecto, y
require() para fallar claro si falta una clave obligatoria.

(La lista documentada de TODAS las variables de entorno que el proyecto lee está en
.env.example, no acá: por eso distribuir la lectura por módulo no pierde el punto de
auditoría.)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# Rutas del proyecto.
#
# Calculamos la raíz de forma EXPLÍCITA en vez de confiar en "el directorio actual",
# porque desde dónde ejecutes varía (terminal, IDE, Docker). __file__ es este archivo;
# subimos tres niveles  src/a_lo_paisa/config.py -> src/a_lo_paisa -> src -> raíz.
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# load_dotenv() vuelca el contenido del .env en os.environ; a partir de ahí os.getenv()
# lo ve. Se ejecuta al importar este módulo, así que cualquier módulo que vaya a leer
# env vars debe importar config primero (aunque sea por este efecto).
# override=False: si una variable ya existe en el entorno real, esa gana sobre el
# .env (útil en Spaces, donde los secretos se inyectan por entorno).
load_dotenv(dotenv_path=ENV_PATH, override=False)


# ──────────────────────────────────────────────────────────────────────────────
# Claves / tokens (secretos). Pueden ser None si no están definidas: no reventamos
# acá, porque hay etapas que no las necesitan. La validación se hace bajo demanda con
# require() solo cuando un módulo realmente la necesita.
#
# HF_TOKEN: token de Hugging Face. Chatterbox descarga sus pesos de un repo PÚBLICO,
# así que no es obligatorio, pero evita límites de descarga y hace falta para Spaces
# privados.
# ──────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")


def require(name: str) -> str:
    """Devuelve el valor de una variable de entorno obligatoria o falla claro.

    Úsalo cuando un módulo SÍ necesita la clave para funcionar. Por ejemplo, llm.py
    hará `config.require("GEMINI_API_KEY")` y, si no está, verás un error explicando
    exactamente qué configurar, en vez de un fallo confuso más adelante en la cadena.

    Args:
        name: nombre de la variable de entorno (p. ej. "GEMINI_API_KEY").

    Returns:
        El valor de la variable como str.

    Raises:
        RuntimeError: si la variable no está definida o está vacía.
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Falta la variable de entorno obligatoria '{name}'. Copia .env.example a .env y rellena '{name}'."
        )
    return value
