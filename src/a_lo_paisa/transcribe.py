"""Etapa 1 del pipeline: STT (Speech-To-Text) con faster-whisper.

Qué hace este módulo:
- Carga un modelo Whisper cuantizado a int8 que corre en CPU.
- Expone `transcribir(ruta_audio, idioma_codigo)` que devuelve
  (texto, idioma_detectado, confianza).
- El idioma lo dirige el DIAL de entrada: "es"/"en" se le fuerzan a whisper;
  None deja que autodetecte (caso 'otro').

Conceptos clave (porque estás aprendiendo):

* faster-whisper vs. Whisper "original":
  faster-whisper reimplementa Whisper sobre CTranslate2, un motor de inferencia
  optimizado. Resultado: mismo modelo, pero varias veces más rápido y con menos
  memoria, especialmente en CPU. Por eso lo elegimos aquí (no asumimos GPU).

* Cuantización int8 (compute_type="int8"):
  El modelo guarda sus "pesos" como números. Por defecto serían float32 (32 bits
  cada uno). int8 los aproxima a enteros de 8 bits. Beneficios: ~4x menos memoria
  y más velocidad en CPU. Coste: una pérdida de precisión casi siempre
  imperceptible para transcribir. Es el punto dulce para correr en CPU sin GPU.
"""

import os
from functools import lru_cache

from faster_whisper import WhisperModel

# config se importa por su EFECTO: al cargarse ejecuta load_dotenv(), así las env vars
# del STT de abajo ven lo del .env. No usamos config en sí (de ahí el noqa).
from a_lo_paisa import config  # noqa: F401

# ──────────────────────────────────────────────────────────────────────────────
# Config del STT (perillas de entorno; el .env las puede sobreescribir).
#
# MODEL_SIZE: tamaño del modelo Whisper (tiny < base < small < medium < large-v3).
#   Más grande = más preciso pero más lento. "small" es buen punto de partida en CPU.
# DEVICE: "auto" = ctranslate2 elige solo (CPU si no hay GPU; CUDA si la hay), SIN
#   depender de torch (tiene su propia detección). Hoy el Space va en CPU, así que
#   resuelve a CPU; el día de una GPU dedicada el STT la tomaría solo, sin tocar código.
# COMPUTE_TYPE: "int8" = la cuantización ligera para CPU.
# ──────────────────────────────────────────────────────────────────────────────
STT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "small")
STT_DEVICE = os.getenv("STT_DEVICE", "auto")
STT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "int8")


@lru_cache(maxsize=1)
def _get_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    """Carga el modelo una sola vez y lo reutiliza.

    Cargar Whisper (descargar pesos la primera vez + inicializar) es costoso. Con
    @lru_cache, la primera llamada construye el modelo y las siguientes devuelven
    el MISMO objeto cacheado. Así, si transcribes varios audios en una sesión, no
    pagas el arranque cada vez. Es lazy: no se carga hasta que se usa de verdad.

    Recibimos model_size, device y compute_type COMO ARGUMENTOS (no leemos config
    aquí dentro) por dos motivos: el llamador los toma de config en un único sitio,
    y así esos valores son la CLAVE del @lru_cache —pedir la misma combinación
    reutiliza el modelo cacheado; cambiarla carga uno nuevo.
    """
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribir(ruta_audio: str, idioma_codigo: str | None) -> tuple[str, str, float]:
    """Transcribe un audio a texto, dirigido por el DIAL de idioma de entrada.

    Args:
        ruta_audio: ruta a un archivo de audio (wav, mp3, m4a, ogg...).
            faster-whisper usa ffmpeg por debajo, así que acepta muchos formatos.
        idioma_codigo: "es" o "en" para FORZAR ese idioma en whisper; None para que
            AUTODETECTE (caso 'otro' del dial). Forzar el idioma cuando ya lo sabemos
            evita que whisper se equivoque (común en audios cortos o ruidosos).

    Returns:
        (texto, idioma_detectado, confianza):
          - texto: la transcripción completa.
          - idioma_detectado: info.language (el forzado, o el hallado si autodetecta).
          - confianza: info.language_probability (0..1). Es solo INFORMATIVA —se
            imprime—, NO decide nada en el pipeline (eso lo fija el dial).
    """
    model = _get_model(STT_MODEL_SIZE, STT_DEVICE, STT_COMPUTE_TYPE)

    # model.transcribe() devuelve:
    #   - segments: un GENERADOR de trozos (perezoso: el trabajo real ocurre al iterar).
    #   - info: metadatos, incluido el idioma detectado y su probabilidad.
    #
    # language=idioma_codigo: si el dial dio "es"/"en", se lo pasamos FIJO y whisper
    # no autodetecta; si es None, whisper detecta el idioma solo.
    #
    # beam_size=5: "beam search" explora varias hipótesis y se queda con la mejor.
    # 5 es el valor habitual: mejor calidad que 1 (voraz) a un coste moderado.
    segments, info = model.transcribe(ruta_audio, language=idioma_codigo, beam_size=5)

    # Concatenamos el texto de todos los segmentos (al iterar se ejecuta la
    # transcripción de verdad). .strip() limpia espacios sobrantes.
    texto = " ".join(segment.text.strip() for segment in segments).strip()

    # Print de verificación pedido: 'STT escuchó [es 0.98]: <texto>'. La confianza
    # es solo informativa para el operador; no cambia el flujo.
    print(f"STT escuchó [{info.language} {info.language_probability:.2f}]: {texto}")

    return texto, info.language, info.language_probability


# ──────────────────────────────────────────────────────────────────────────────
# Prueba manual desde la terminal.
#
# Este bloque SOLO se ejecuta si corres el archivo directamente, no al importarlo.
# Uso:
#   uv run python -m a_lo_paisa.transcribe ruta/al/audio.wav
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python -m a_lo_paisa.transcribe <ruta_audio>")
        sys.exit(1)

    ruta = sys.argv[1]
    print(f"Transcribiendo: {ruta}")
    # Para la prueba manual autodetectamos el idioma (idioma_codigo=None).
    texto, _idioma, _conf = transcribir(ruta, None)
    print("\n--- Transcripción ---")
    print(texto)
