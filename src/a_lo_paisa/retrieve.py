"""Capa de RECUPERACIÓN (RAG): glosario + embeddings + búsqueda semántica.

Aísla todo lo de "dado un texto, traé las entradas del glosario relevantes" del resto
del agente (paisa_transform.py), que se queda solo con la generación. El índice de
embeddings se precomputa offline (scripts/build_embeddings.py); acá se carga y se
embebe la consulta en caliente.
"""

import json
import textwrap
from functools import lru_cache

import numpy as np
from google import genai
from google.genai import types, errors

from a_lo_paisa import config
from a_lo_paisa.llm import EMBED_DIM, EMBED_MODEL, TransformacionError, get_client

# ──────────────────────────────────────────────────────────────────────────────
# Rutas y constantes de recuperación.
# ──────────────────────────────────────────────────────────────────────────────
GLOSSARY_PATH = config.PROJECT_ROOT / "data" / "paisa_glossary.json"
EMBEDDINGS_PATH = config.PROJECT_ROOT / "data" / "glossary_embeddings.npz"

# Cuántas entradas del glosario recupera el RAG por defecto.
K_DEFAULT = 5

# EMBED_BATCH: cuántos textos embebemos por request (las ~85 entran en una sola). Los
# IDENTIFICADORES del modelo (EMBED_MODEL/EMBED_DIM) viven en llm.py y se importan arriba.
EMBED_BATCH = 100


def cargar_glosario() -> dict:
    """Lee data/paisa_glossary.json y devuelve el dict completo (con _meta y entries)."""
    with open(GLOSSARY_PATH, encoding="utf-8") as f:
        return json.load(f)


def embeber(client: genai.Client, textos: list[str], task_type: str) -> np.ndarray:
    """Embebe una lista de textos y devuelve una matriz (n x EMBED_DIM) L2-normalizada.

    Args:
        client: cliente de google-genai ya creado.
        textos: textos a embeber, en orden.
        task_type: "RETRIEVAL_DOCUMENT" para los documentos del índice (offline) o
            "RETRIEVAL_QUERY" para la consulta (online). Usar el task_type correcto
            en cada lado mejora la calidad de recuperación: el modelo coloca query y
            documento en el mismo espacio pensado para búsqueda.

    Devuelve vectores UNITARIOS (norma 1). ¿Por qué normalizar acá? Porque así la
    similitud coseno entre dos vectores se reduce a su producto punto (más simple y
    rápido), y porque gemini-embedding NO normaliza automáticamente cuando se trunca
    la dimensión a menos de 3072 (nuestro caso, 768): hay que hacerlo a mano.
    """
    vectores: list[list[float]] = []
    # Procesamos por lotes por si la lista excede el máximo por request.
    for inicio in range(0, len(textos), EMBED_BATCH):
        lote = textos[inicio : inicio + EMBED_BATCH]
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=lote,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBED_DIM,
            ),
        )
        # resp.embeddings viene alineado con el orden de `lote`.
        vectores.extend(e.values for e in resp.embeddings)

    matriz = np.asarray(vectores, dtype=np.float32)
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    normas[normas == 0] = 1.0  # evita división por cero en un vector nulo (no debería pasar)
    return matriz / normas


def formatear_contexto(entradas: list[dict]) -> str:
    """Convierte entradas del glosario en texto legible para inyectar al prompt.

    Cada línea muestra: neutro -> términos paisa con su EXAGERACIÓN ('exa'), las
    notas, y hasta 2 EJEMPLOS de uso. Los ejemplos son fuente importante de estilo
    (le muestran al modelo cómo suena el término en una frase real), pero limitamos a
    2 por entrada para no inflar tokens del prompt. La exageración por término ('exa')
    le permite al modelo elegir acorde al nivel pedido.

    Formato por entrada:
        - neutro: término1 (exa 2), término2 (exa 3) | nota: ... | ej: "frase1"; "frase2"
    """
    lineas = []
    for e in entradas:
        terminos = ", ".join(f"{p['t']} (exa {p['exa']})" for p in e.get("paisa", []))
        linea = f"- {e.get('neutro')}: {terminos}"
        if e.get("notas"):
            linea += f" | notas: {e['notas']}"
        # Máximo 2 ejemplos por entrada (control de tokens).
        ejemplos = e.get("ejemplos", [])[:2]
        if ejemplos:
            ej_txt = "; ".join(f'"{x}"' for x in ejemplos)
            linea += f" | ej: {ej_txt}"
        lineas.append(linea)
    return "\n".join(lineas)


@lru_cache(maxsize=1)
def _cargar_indice() -> tuple[np.ndarray, dict]:
    """Carga la matriz de embeddings + el glosario, y VALIDA que correspondan.

    Devuelve (matriz_normalizada, glosario). Se cachea: el índice no cambia durante
    la corrida. Falla con mensajes accionables si el índice no existe o quedó
    desfasado del glosario actual.
    """
    if not EMBEDDINGS_PATH.exists():
        # dedent: la 2ª línea va indentada en el código por prolijidad; dedent le
        # quita esa sangría para que NO se cuele en el mensaje.
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

    # Validación 1: la versión del glosario del índice debe coincidir con la actual.
    # Si editaste el glosario y no regeneraste, los vectores ya no representan los
    # textos correctos (y la alineación por índice puede romperse).
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

    # Validación 2: alineación por índice (fila i del vector <-> entries[i]).
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
    """HERRAMIENTA: devuelve las `k` entradas del glosario más cercanas a `texto`.

    Recuperación SEMÁNTICA: embebe `texto` con el mismo modelo de embeddings que el
    índice (task_type RETRIEVAL_QUERY) y rankea por similitud coseno contra la
    matriz precomputada. Como son ~85 vectores, no hace falta base vectorial: un
    producto punto sobre vectores normalizados (coseno) y un argsort bastan.

    Es la pieza pensada como TOOL de un futuro agente: entra texto, salen entradas;
    sin estado mutable ni efectos colaterales.

    Args:
        texto: la frase de entrada (en español neutro).
        k: cuántas entradas devolver.

    Returns:
        Lista de hasta `k` entradas del glosario (dicts), de más a menos similar.

    Raises:
        TransformacionError: si falla el embed de la consulta. Lo plegamos a esta
            excepción para que el portero del orquestador atrape el fallo igual que
            uno de generación (un solo except cubre TODO el pipeline).
    """
    matriz, glosario = _cargar_indice()
    client = get_client()

    # La consulta se embebe como RETRIEVAL_QUERY (los documentos se embebieron como
    # RETRIEVAL_DOCUMENT): así query y documentos quedan en el mismo espacio de
    # búsqueda. embeber() ya devuelve el vector normalizado.
    try:
        consulta = embeber(client, [texto], task_type="RETRIEVAL_QUERY")[0]
    except errors.APIError as e:
        # OJO: solo plegamos acá (ruta "en caliente" del agente). El builder offline
        # (build_embeddings.py) llama a embeber() directo y maneja sus errores
        # aparte, así que ese script NO se ve afectado.
        raise TransformacionError(
            "No se pudo consultar el glosario (falló el servicio de embeddings). Intentá de nuevo."
        ) from e

    # Coseno = producto punto (todo está normalizado a norma 1).
    similitudes = matriz @ consulta

    # Índices de las k mayores similitudes, de mayor a menor.
    k = max(1, min(k, similitudes.shape[0]))
    mejores = np.argsort(-similitudes)[:k]
    return [glosario["entries"][i] for i in mejores]
