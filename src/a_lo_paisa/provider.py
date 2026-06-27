"""Identidad de los modelos, excepción común y cliente del proveedor."""

from functools import lru_cache

from google import genai

from a_lo_paisa import config

MODEL = "gemini-2.5-flash"            # generación (capa gratuita)
EMBED_MODEL = "gemini-embedding-001"  # embeddings (capa gratuita)
EMBED_DIM = 768  # truncado de 3072 (gemini-embedding-001 lo permite)


class TransformacionError(Exception):
    """Falla NO recuperable al embeber en la recuperación o al hablar con el LLM."""


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """Crea el cliente del LLM una sola vez (cacheado) y lo reusa en toda la corrida."""
    return genai.Client(api_key=config.require("LLM_API_KEY"))
