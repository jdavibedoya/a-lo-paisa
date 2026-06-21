"""Capa de RECUPERACIÓN (RAG): glosario + embeddings + búsqueda semántica.

Aísla "dado un texto, traé las entradas del glosario relevantes". El índice se precomputa
offline (scripts/build_embeddings.py); acá se carga y se embebe la consulta en caliente.
"""

import json
import textwrap
from functools import lru_cache

import numpy as np
from google import genai
from google.genai import errors, types

from a_lo_paisa import config
from a_lo_paisa.llm import EMBED_DIM, EMBED_MODEL, TransformacionError, get_client

GLOSSARY_PATH = config.PROJECT_ROOT / "data" / "paisa_glossary.json"
EMBEDDINGS_PATH = config.PROJECT_ROOT / "data" / "glossary_embeddings.npz"
K_DEFAULT = 5      # cuántas entradas recupera el RAG por defecto
EMBED_BATCH = 100  # textos por request al embeber (las ~85 entran en una sola)


def cargar_glosario() -> dict:
    """Lee data/paisa_glossary.json y devuelve el dict completo (_meta + entries)."""
    with open(GLOSSARY_PATH, encoding="utf-8") as f:
        return json.load(f)


def embeber(client: genai.Client, textos: list[str], task_type: str) -> np.ndarray:
    """Embebe `textos` y devuelve una matriz (n × EMBED_DIM) L2-normalizada.

    task_type: "RETRIEVAL_DOCUMENT" para los documentos del índice (offline) o
    "RETRIEVAL_QUERY" para la consulta (online); usar el correcto en cada lado mejora la
    recuperación. Normalizamos acá porque así la similitud coseno se reduce a un producto
    punto, y porque gemini-embedding no normaliza al truncar la dimensión a <3072 (768).
    """
    vectores: list[list[float]] = []
    for inicio in range(0, len(textos), EMBED_BATCH):
        lote = textos[inicio : inicio + EMBED_BATCH]
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=lote,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM),
        )
        vectores.extend(e.values for e in resp.embeddings)

    matriz = np.asarray(vectores, dtype=np.float32)
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    normas[normas == 0] = 1.0  # evita división por cero (no debería pasar)
    return matriz / normas


def formatear_contexto(entradas: list[dict]) -> str:
    """Convierte entradas del glosario en texto para inyectar al prompt.

    Por entrada: neutro -> términos paisa (con su exageración 'exa'), notas y hasta 2
    ejemplos (límite para no inflar tokens). Formato:
        - neutro: término1 (exa 2), término2 (exa 3) | notas: ... | ej: "frase1"; "frase2"
    """
    lineas = []
    for e in entradas:
        terminos = ", ".join(f"{p['t']} (exa {p['exa']})" for p in e.get("paisa", []))
        linea = f"- {e.get('neutro')}: {terminos}"
        if e.get("notas"):
            linea += f" | notas: {e['notas']}"
        ejemplos = e.get("ejemplos", [])[:2]
        if ejemplos:
            ej_txt = "; ".join(f'"{x}"' for x in ejemplos)
            linea += f" | ej: {ej_txt}"
        lineas.append(linea)
    return "\n".join(lineas)


@lru_cache(maxsize=1)
def _cargar_indice() -> tuple[np.ndarray, dict]:
    """Carga la matriz de embeddings + el glosario y VALIDA que correspondan (cacheado).

    Falla con mensajes accionables si el índice no existe o quedó desfasado del glosario.
    """
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            textwrap.dedent(
                f"""\
                No existe el índice de embeddings en '{EMBEDDINGS_PATH}'.
                Corré primero: uv run python scripts/build_embeddings.py"""
            )
        )

    datos = np.load(EMBEDDINGS_PATH, allow_pickle=False)
    matriz = datos["embeddings"].astype(np.float32)
    glosario = cargar_glosario()
    entradas = glosario.get("entries", [])

    # El índice debe ser del glosario actual: si lo editaste sin regenerar, los vectores
    # ya no representan los textos (y la alineación por índice se rompe).
    version_idx = str(datos["glossary_version"].item())
    version_actual = str(glosario.get("_meta", {}).get("version", "desconocida"))
    if version_idx != version_actual:
        raise RuntimeError(
            textwrap.dedent(
                f"""\
                El índice de embeddings es del glosario v{version_idx}, pero el actual es v{version_actual}.
                Regeneralo: uv run python scripts/build_embeddings.py"""
            )
        )

    # Alineación por índice: fila i del vector <-> entries[i].
    if matriz.shape[0] != len(entradas):
        raise RuntimeError(
            textwrap.dedent(
                f"""\
                El índice tiene {matriz.shape[0]} vectores pero el glosario tiene {len(entradas)} entradas.
                Regeneralo: uv run python scripts/build_embeddings.py"""
            )
        )

    return matriz, glosario


def lookup_paisa(texto: str, k: int = K_DEFAULT) -> list[dict]:
    """Devuelve las `k` entradas del glosario más cercanas a `texto` (búsqueda semántica).

    Embebe `texto` (RETRIEVAL_QUERY) y rankea por similitud coseno contra la matriz
    precomputada. Con ~85 vectores no hace falta base vectorial: producto punto + argsort.

    Raises:
        TransformacionError: si falla el embed de la consulta. Lo plegamos a esa excepción
            para que el portero del orquestador cubra todo el pipeline con un solo except.
    """
    matriz, glosario = _cargar_indice()
    client = get_client()

    try:
        consulta = embeber(client, [texto], task_type="RETRIEVAL_QUERY")[0]
    except errors.APIError as e:
        # Solo plegamos acá (ruta en caliente); build_embeddings.py maneja sus errores aparte.
        raise TransformacionError(
            "No se pudo consultar el glosario (falló el servicio de embeddings). Intentá de nuevo."
        ) from e

    similitudes = matriz @ consulta  # coseno = producto punto (todo normalizado)
    k = max(1, min(k, similitudes.shape[0]))
    mejores = np.argsort(-similitudes)[:k]
    return [glosario["entries"][i] for i in mejores]
