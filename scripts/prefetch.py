"""Pre-descarga los modelos (TTS + STT) — se corre en el BUILD del Docker.

Hornea los pesos en la imagen para que el RUNTIME no descargue nada: cold-start rápido y,
sobre todo, el Space no vuelve a bajar varios GB cada vez que despierta.

    uv run python scripts/prefetch.py
"""

import glob
import os

from a_lo_paisa import synthesize, transcribe


def _podar_s3gen_v3_source() -> None:
    """Borra el s3gen_v3 source del cache HF: solo sirvió para construir el s3gen combinado;
    el runtime lee el combinado, no el source (~1 GB de más).

    Va en el MISMO proceso que el ensamblado A PROPÓSITO: en Docker, borrar en una capa
    POSTERIOR no achica las anteriores, así que la descarga y el borrado deben quedar en la
    misma RUN para que la imagen realmente baje. (Nada en el dir ensamblado apunta a este
    source —el combinado es un archivo real—, así que borrarlo no rompe ningún symlink.)
    """
    cache = os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
    patron = os.path.join(cache, "models--ResembleAI--Chatterbox-Multilingual-es-mx-latam", "snapshots", "*", "s3gen_v3.pt")
    for link in glob.glob(patron):
        blob = os.path.realpath(link)  # el snapshot es un symlink al blob real
        os.remove(link)
        if os.path.isfile(blob):
            os.remove(blob)  # libera el GB
        print("   podado: s3gen_v3.pt (source, ya combinado)")


def main() -> None:
    # TTS: ensambla el Language Pack LatAm (descarga sus pesos y construye el s3gen
    # combinado). En runtime, from_local lee de esa carpeta sin bajar nada.
    synthesize._preparar_ckpt_latam()
    _podar_s3gen_v3_source()
    # STT: instancia el modelo whisper, lo que dispara y cachea su descarga.
    transcribe._get_model(transcribe.STT_MODEL_SIZE, "cpu", transcribe.STT_COMPUTE_TYPE)
    print("✅ Modelos horneados en la imagen (TTS + STT).")


if __name__ == "__main__":
    main()
