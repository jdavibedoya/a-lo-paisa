"""Construye OFFLINE el índice de embeddings del glosario paisa.

Se corre a mano cada vez que cambie el glosario:
    uv run python scripts/build_embeddings.py

Precomputar aparte (en vez de embeber al vuelo) evita re-embeber el glosario estático en
cada arranque: en caliente el pipeline solo embebe la consulta (1 vector) y compara contra
la matriz ya lista. Embebemos SOLO 'neutro' + 'notas', no la jerga paisa: el modelo de
embeddings entiende español estándar y la jerga ('parce', 'guaro') mete ruido; el
significado en español es lo que hace que 'mi compañero sin billete' caiga cerca de
'amigo'/'dinero' por sentido y no por las palabras.
"""

import sys

import numpy as np
from google import genai
from google.genai import errors

from a_lo_paisa import config, retrieve as r


def texto_a_embeber(entrada: dict) -> str:
    """Texto solo-neutro de una entrada: 'neutro' + 'notas' (ambos en español)."""
    neutro = (entrada.get("neutro") or "").strip()
    notas = (entrada.get("notas") or "").strip()
    return f"{neutro}. {notas}".strip() if notas else neutro


def main() -> None:
    client = genai.Client(api_key=config.require("GEMINI_API_KEY"))

    glosario = r.cargar_glosario()
    entradas = glosario.get("entries", [])
    version = glosario.get("_meta", {}).get("version", "desconocida")

    # Un texto por entrada, EN ORDEN: la fila i de la matriz queda alineada con entries[i].
    textos = [texto_a_embeber(e) for e in entradas]
    print(f"Glosario v{version}: {len(textos)} entradas a embeber con {r.EMBED_MODEL} (dim {r.EMBED_DIM})...")

    try:
        matriz = r.embeber(client, textos, task_type="RETRIEVAL_DOCUMENT")
    except errors.ClientError as e:
        code = getattr(e, "code", None)
        if code == 429:
            print("\n❌ Límite de tasa (429) del free tier. Espera a que se renueve la cuota y reintenta.")
        else:
            print(f"\n❌ Error de cliente {code}: {getattr(e, 'message', str(e))}")
        sys.exit(1)
    except errors.APIError as e:
        print(f"\n❌ Error de la API de embeddings: {getattr(e, 'message', str(e))}")
        sys.exit(1)

    # Metadatos para que el pipeline valide que el índice corresponde al glosario actual.
    # Arrays 0-d porque np.savez no admite dicts (y se lee con allow_pickle=False).
    np.savez(
        r.EMBEDDINGS_PATH,
        embeddings=matriz,
        embedding_model=np.array(r.EMBED_MODEL),
        dim=np.array(r.EMBED_DIM),
        glossary_version=np.array(version),
        n=np.array(len(textos)),
    )
    print(f"✅ Guardados {matriz.shape[0]} vectores ({matriz.shape[1]} dims) en: {r.EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
