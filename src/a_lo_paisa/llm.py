"""Acceso al LLM: cliente del proveedor, identidad de los modelos y excepción común.

Único punto donde vive el proveedor (hoy Gemini): cambiar de proveedor o de modelo se
toca acá. Vive aparte de retrieve.py y paisa_transform.py (que lo usan) para evitar un
import circular; solo depende de config.
"""

from functools import lru_cache

from google import genai

from a_lo_paisa import config

# Identidad de los modelos (centralizada: cambiar de modelo es un cambio de un solo sitio).
MODEL = "gemini-2.5-flash"            # generación (capa gratuita)
EMBED_MODEL = "gemini-embedding-001"  # embeddings (capa gratuita)
# EMBED_DIM debe ser IGUAL al construir el índice y al consultar, o las similitudes no
# tendrían sentido. 768 sobra para ~85 vectores y es más liviano que el máximo (3072).
EMBED_DIM = 768


class TransformacionError(Exception):
    """Falla NO recuperable al hablar con el LLM (o al embeber en la recuperación).

    Lleva un mensaje apto para el usuario. El orquestador la atrapa (el "portero") para no
    confundirla con el texto generado. La usan lookup_paisa, transformar_a_paisa y
    normalizar_a_espanol.
    """


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """Crea el cliente de Gemini una sola vez (cacheado) y lo reusa en toda la corrida."""
    return genai.Client(api_key=config.require("GEMINI_API_KEY"))
