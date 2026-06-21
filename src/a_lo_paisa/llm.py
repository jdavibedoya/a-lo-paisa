"""Acceso al LLM: cliente del proveedor + identidad de los modelos + excepción común.

(Renombrado de gemini.py: este es el ÚNICO punto donde vive el proveedor —hoy
Gemini—, así que un nombre neutro deja claro que acá se concentraría el cambio el día
que se use otro proveedor.)

Acá viven:
  - get_client(): el cliente del proveedor (hoy google-genai), cacheado.
  - los IDENTIFICADORES de modelo que usamos (generación y embeddings), juntos para
    que cambiar de modelo sea un cambio de UN solo archivo.
  - TransformacionError: la excepción común del pipeline.

Vive aparte para evitar un import circular: lo necesitan tanto la recuperación
(retrieve.py) como la generación (paisa_transform.py), y este módulo solo depende de
config.
"""

from functools import lru_cache

from google import genai

from a_lo_paisa import config

# ──────────────────────────────────────────────────────────────────────────────
# Identidad de los modelos. Centralizada acá: cambiar de modelo se toca en UN sitio.
# ──────────────────────────────────────────────────────────────────────────────
# Modelo de GENERACIÓN: gemini-2.5-flash (capa gratuita, disponibilidad decente).
MODEL = "gemini-2.5-flash"

# Modelo de EMBEDDINGS: gemini-embedding-001, en free tier. La dimensión es
# configurable (128..3072); 768 sobra para ~85 vectores y es más liviano que 3072.
# EMBED_DIM debe ser IGUAL al construir el índice y al consultar, o las similitudes
# no tendrían sentido.
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768


class TransformacionError(Exception):
    """Falla NO recuperable al hablar con el LLM.

    Se levanta cuando, tras los reintentos, el modelo no entrega un resultado usable
    (429/5xx persistente, respuesta vacía o bloqueada por filtros, u otro error de
    API), o cuando falla el embed de la recuperación. Lleva un mensaje APTO PARA EL
    USUARIO. La idea es que el orquestador la atrape (el "portero") y NUNCA la deje
    salir confundida con el texto generado —en vez de devolver un string "[ERROR...]"
    que se podría imprimir como si fuera la respuesta paisa. La usan por igual la
    recuperación (lookup_paisa), transformar_a_paisa y normalizar_a_espanol.
    """


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """Crea el cliente de Gemini una sola vez (lee la API key del .env vía config).

    Es @lru_cache para reusar el MISMO cliente entre embeddings y generación en toda
    la corrida, en vez de abrir uno por módulo.
    """
    return genai.Client(api_key=config.require("GEMINI_API_KEY"))
