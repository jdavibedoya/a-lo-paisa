"""STT (Speech-To-Text) con faster-whisper.

faster-whisper corre Whisper sobre CTranslate2 (motor optimizado): mismo modelo, mucho
más rápido y liviano en CPU. La cuantización int8 baja memoria y acelera en CPU, con una
pérdida de precisión despreciable para transcribir.
"""

import os
from functools import lru_cache

from faster_whisper import WhisperModel

# config se importa por su EFECTO (ejecuta load_dotenv), para que las env vars de abajo
# vean el .env. No se usa directamente, de ahí el noqa.
from a_lo_paisa import config  # noqa: F401

# Perillas del STT (overrideables por env var; ver .env.example).
#   MODEL_SIZE: tiny < base < small < medium < large-v3 (más grande = más preciso/lento).
#   DEVICE: "auto" = ctranslate2 elige solo (CPU sin GPU, CUDA si la hay), sin torch.
#   COMPUTE_TYPE: "int8" = cuantización ligera para CPU.
STT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "small")
STT_DEVICE = os.getenv("STT_DEVICE", "auto")
STT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "int8")


@lru_cache(maxsize=1)
def _get_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    """Carga el modelo una sola vez (cacheado por sus parámetros) y lo reusa."""
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribir(ruta_audio: str, idioma_codigo: str | None) -> tuple[str, str, float]:
    """Transcribe un audio, dirigido por el DIAL de idioma de entrada.

    Args:
        ruta_audio: archivo de audio (wav, mp3, m4a, ogg...; whisper usa ffmpeg debajo).
        idioma_codigo: "es"/"en" para FORZAR ese idioma; None para autodetectar ('otro').

    Returns:
        (texto, idioma_detectado, confianza). La confianza es solo informativa; quien
        decide el flujo es el dial, no ella.
    """
    model = _get_model(STT_MODEL_SIZE, STT_DEVICE, STT_COMPUTE_TYPE)
    # segments es un generador perezoso (la transcripción ocurre al iterarlo). beam_size=5
    # explora varias hipótesis y se queda con la mejor (buena calidad a coste moderado).
    segments, info = model.transcribe(ruta_audio, language=idioma_codigo, beam_size=5)
    texto = " ".join(segment.text.strip() for segment in segments).strip()
    print(f"STT escuchó [{info.language} {info.language_probability:.2f}]: {texto}")
    return texto, info.language, info.language_probability


# Prueba manual:  uv run python -m a_lo_paisa.transcribe <ruta_audio>
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python -m a_lo_paisa.transcribe <ruta_audio>")
        sys.exit(1)
    ruta = sys.argv[1]
    print(f"Transcribiendo: {ruta}")
    texto, _idioma, _conf = transcribir(ruta, None)
    print("\n--- Transcripción ---")
    print(texto)
