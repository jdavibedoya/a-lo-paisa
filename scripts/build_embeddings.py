"""Construye offline el índice de embeddings del glosario paisa.

Se debe correr cada vez que cambie el glosario: uv run python scripts/build_embeddings.py
Se embebe solo 'neutro' + 'notas', no la jerga paisa.
"""

import sys

import numpy as np
from google.genai import errors

from a_lo_paisa import provider, retrieve as r


def texto_a_embeber(entrada: dict) -> str:
    """Texto a embeber de una entrada: 'neutro' + 'notas'."""
    neutro = (entrada.get("neutro") or "").strip()
    notas = (entrada.get("notas") or "").strip()
    return f"{neutro}. {notas}".strip() if notas else neutro


def main() -> None:
    client = provider.get_client()

    glosario = r.cargar_glosario()
    entradas = glosario.get("entries", [])
    version = glosario.get("_meta", {}).get("version")

    # Un texto por entrada. Ordenado: la fila i de la matriz queda alineada con entries[i].
    textos = [texto_a_embeber(e) for e in entradas]
    print(f"Glosario v{version}: {len(textos)} entradas a embeber con {provider.EMBED_MODEL} (dim {provider.EMBED_DIM})...")

    try:
        matriz = r.embeber(client, textos, "RETRIEVAL_DOCUMENT")
    except errors.ClientError as e:
        code = getattr(e, "code", None)
        if code == 429:
            print("\n❌ Límite de tasa (429) del free tier. Esperá un momento y reintentá.")
        else:
            print(f"\n❌ Error de cliente {code}: {getattr(e, 'message', str(e))}")
        sys.exit(1)
    except errors.APIError as e:
        print(f"\n❌ Error de la API de embeddings: {getattr(e, 'message', str(e))}")
        sys.exit(1)

    # Guardar con metadatos para que el pipeline valide que el índice corresponde al glosario actual.
    # Arrays 0-d porque np.savez no admite dicts y se lee con allow_pickle=False
    np.savez(
        r.EMBEDDINGS_PATH,
        embeddings=matriz,
        embedding_model=np.array(provider.EMBED_MODEL),
        dim=np.array(provider.EMBED_DIM),
        glossary_version=np.array(version),
        n=np.array(len(textos)),
    )
    print(f"✅ Guardados {matriz.shape[0]} vectores ({matriz.shape[1]} dims) en: {r.EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
