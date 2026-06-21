"""Construye, OFFLINE y una sola vez, el índice de embeddings del glosario paisa.

Se corre a mano cada vez que cambie el glosario:
    uv run python scripts/build_embeddings.py

¿Por qué PRECOMPUTAR aparte y no embeber al vuelo dentro del agente?
- El glosario es estático entre ediciones (las entradas no cambian). Embeberlas en
  cada arranque del agente sería lento y gastaría cuota repetida en vectores
  idénticos. Calculándolos una vez y cacheándolos en disco, el agente en caliente
  solo embebe la consulta del usuario (1 vector) y compara contra la matriz ya lista.

¿Por qué embeber SOLO 'neutro' + 'notas' (y NO los términos paisa ni los ejemplos)?
- El modelo de embeddings entiende español estándar; la jerga paisa ('parce',
  'guaro', 'achilao') la modela mal y mete RUIDO en el espacio vectorial. El 'neutro'
  (el concepto) y las 'notas' (su definición en español) describen el SIGNIFICADO de
  la entrada en lenguaje que el modelo sí captura, que es justo lo que hace falta
  para que una consulta neutra ('mi compañero anda sin billete') caiga cerca de la
  entrada correcta ('amigo', 'dinero') por sentido, no por las palabras paisas.
"""

import sys

import numpy as np

from a_lo_paisa import config
from a_lo_paisa import retrieve as r
from google import genai
from google.genai import errors


def texto_a_embeber(entrada: dict) -> str:
    """Arma el texto SOLO-NEUTRO de una entrada: 'neutro' + 'notas' (ambos español).

    Deliberadamente NO incluye paisa[].t ni ejemplos (ver el porqué en el docstring
    del módulo). Si no hay notas, basta el neutro.
    """
    neutro = (entrada.get("neutro") or "").strip()
    notas = (entrada.get("notas") or "").strip()
    return f"{neutro}. {notas}".strip() if notas else neutro


def main() -> None:
    api_key = config.require("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    glosario = r.cargar_glosario()
    entradas = glosario.get("entries", [])
    version = glosario.get("_meta", {}).get("version", "desconocida")

    # Un texto por entrada, EN ORDEN: la fila i de la matriz quedará alineada con
    # entries[i]. Por eso embebemos TODAS las entradas (ninguna se salta).
    textos = [texto_a_embeber(e) for e in entradas]
    print(f"Glosario v{version}: {len(textos)} entradas a embeber con {r.EMBED_MODEL} (dim {r.EMBED_DIM})...")

    try:
        # task_type RETRIEVAL_DOCUMENT: estos son los "documentos" del índice.
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

    # Metadatos mínimos para que el agente VALIDE que el índice corresponde al
    # glosario actual y se construyó con el mismo modelo/dim. Los guardamos como
    # arrays 0-d (np.savez no admite dicts; allow_pickle se mantiene en False al leer).
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
