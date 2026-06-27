"""STT (Speech-To-Text) con faster-whisper.

faster-whisper corre Whisper sobre CTranslate2 (motor optimizado): mismo modelo, mucho
más rápido y liviano en CPU. La cuantización int8 baja memoria y acelera en CPU, con una
pérdida de precisión despreciable para transcribir.
"""

from functools import lru_cache

from faster_whisper import WhisperModel, download_model

STT_MODEL_SIZE = "small"    # tiny < base < small < medium < large-v3.
STT_DEVICE = "auto"         # ctranslate2 elige solo (CPU sin GPU, CUDA si la hay), sin torch.
STT_COMPUTE_TYPE = "int8"   # cuantización ligera para CPU.


@lru_cache(maxsize=1)
def _get_stt(model_size: str, device: str, compute_type: str) -> WhisperModel:
    """Carga el modelo una sola vez (cacheado) y lo reusa."""
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def _descargar_pesos(model_size: str) -> None:
    """Baja los pesos del modelo a la cache, sin instanciarlo (para hornear la imagen Docker)."""
    download_model(model_size)


def transcribir(ruta_audio: str, idioma_codigo: str | None) -> tuple[str, str, float]:
    """Transcribe un audio, dirigido por el DIAL de idioma de entrada.

    Args:
        ruta_audio: archivo de audio (wav, mp3, m4a, ogg...; whisper usa ffmpeg debajo).
        idioma_codigo: "es"/"en" para forzar ese idioma; None para autodetectar ('otro').

    Returns:
        (texto, idioma_detectado, confianza)
    """
    model = _get_stt(STT_MODEL_SIZE, STT_DEVICE, STT_COMPUTE_TYPE)
    # segments es un generador perezoso (la transcripción final ocurre al iterarlo).
    segments, info = model.transcribe(ruta_audio, language=idioma_codigo, beam_size=5)
    texto = " ".join(segment.text.strip() for segment in segments).strip()
    print(f"Transcripción STT [{info.language} {info.language_probability:.2f}]: {texto}")
    return texto, info.language, info.language_probability


# Prueba independiente:  uv run python -m a_lo_paisa.transcribe <ruta_audio> <es|en|otro>
if __name__ == "__main__":
    import os
    import sys

    CODIGOS = {"es", "en", "otro"}  # es/en fuerzan idioma; otro = autodetectar

    if len(sys.argv) < 3:
        print("Uso: uv run python -m a_lo_paisa.transcribe <ruta_audio> <es|en|otro>", file=sys.stderr)
        sys.exit(1)

    ruta, codigo = sys.argv[1], sys.argv[2]
    if not os.path.isfile(ruta):
        print(f"Error: no existe el archivo de audio '{ruta}'.", file=sys.stderr)
        sys.exit(1)
    if codigo not in CODIGOS:
        print(f"Error: código de idioma inválido '{codigo}'. Usá es, en u otro.", file=sys.stderr)
        sys.exit(1)

    idioma_codigo = None if codigo == "otro" else codigo  # otro -> None (autodetectar)
    print(f"Transcribiendo: {ruta}")
    texto, _idioma, _conf = transcribir(ruta, idioma_codigo)
    print("\n--- Transcripción ---")
    print(texto)
