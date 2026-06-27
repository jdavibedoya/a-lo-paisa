"""Pipeline de voz con RAG: STT → transformación a paisa → TTS con voz clonada.

Lo usan los entrypoints (CLI, Gradio); emite el resultado de cada paso.
"""

from dataclasses import dataclass

from a_lo_paisa.transcribe import transcribir
from a_lo_paisa.paisa_transform import traducir_a_espanol, reescribir_a_paisa
from a_lo_paisa.synthesize import sintetizar

DIAL_A_CODIGO = {"español": "es", "inglés": "en", "otro": None}  # Dial de idioma -> código de whisper.


@dataclass
class Paso:
    """Un paso terminado del pipeline.

    etapa: 'stt' | 'traducción' | 'reescritura' | 'tts'.
    texto: resultado (texto en 'stt', 'traducción' y 'reescritura', ruta de salida en 'tts').
    """

    etapa: str
    texto: str


def pasos_pipeline(ruta_audio: str, idioma: str, exageracion: int, registro: str, ruta_salida: str):
    """Generador: STT -> (traducción) -> reescritura -> TTS, produciendo cada paso.

    Raises:
        ValueError: si `idioma` no es un dial válido.
        TransformacionError: si falla la traducción o la reescritura (RAG incluido).
        SintesisError: si falla la síntesis de voz.
    """
    if idioma not in DIAL_A_CODIGO:
        raise ValueError(f"idioma '{idioma}' inválido; usá uno de {list(DIAL_A_CODIGO)}.")

    codigo = DIAL_A_CODIGO[idioma]
    transcripcion, _idioma_det, _conf = transcribir(ruta_audio, codigo)
    yield Paso("stt", transcripcion)

    if idioma == "español":  # Quién decide traducir es el DIAL
        texto_es = transcripcion
    else:
        texto_es = traducir_a_espanol(transcripcion)
        yield Paso("traducción", texto_es)

    texto_paisa = reescribir_a_paisa(texto_es, exageracion, registro)
    yield Paso("reescritura", texto_paisa)

    ruta = sintetizar(texto_paisa, ruta_salida, exageracion)
    yield Paso("tts", ruta)
