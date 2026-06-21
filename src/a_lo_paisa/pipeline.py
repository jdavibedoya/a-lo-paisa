"""Orquestación reusable del pipeline (capa de librería, sin UI): audio -> texto paisa.

Encadena STT -> normalización condicional -> reescritura. Es lo que comparten los
entrypoints (scripts/cli.py y app.py), por eso vive en el paquete. A PROPÓSITO no imprime
ni sintetiza voz: el TTS (lento) lo agrega el entrypoint, después de mostrar el texto.

Dos formas de consumirla, misma orquestación:
  - pasos_audio_a_paisa(): generador que emite cada etapa al terminarla (para UIs con
    progreso, como Gradio).
  - audio_a_paisa(): conveniencia no-streaming que devuelve el resultado completo (el CLI).

El "portero" (nunca sintetizar un error) es estructural: si el texto falla, el generador
levanta TransformacionError antes de emitir 'paisa', así que el caller no llega al TTS.
"""

from dataclasses import dataclass

from a_lo_paisa.paisa_transform import normalizar_a_espanol, transformar_a_paisa
from a_lo_paisa.transcribe import transcribir

# Dial de idioma (lo que ve el usuario) -> código de whisper. Vive acá (es lógica del
# pipeline, no de la UI); los entrypoints lo reusan para las opciones. None = autodetectar.
DIAL_A_CODIGO = {"español": "es", "inglés": "en", "otro": None}


@dataclass
class PasoPipeline:
    """Una etapa terminada del pipeline de texto (para UIs con progreso).

    etapa: 'stt' | 'normalizado' (solo si se normalizó) | 'paisa'. texto: su resultado.
    """

    etapa: str
    texto: str


@dataclass
class ResultadoTexto:
    """Resultado completo de la parte de texto (para callers que no streamean)."""

    transcripcion: str    # lo que oyó el STT, en el idioma original
    texto_es: str         # español neutro (== transcripcion si el dial ya era español)
    paisa: str            # el texto reescrito en paisa (lo que se sintetiza)
    se_normalizo: bool    # True si hubo normalización (dial inglés/otro)


def pasos_audio_a_paisa(ruta_audio: str, idioma: str, exageracion: int, registro: str):
    """Generador: corre STT -> (normalización) -> reescritura, emitiendo cada etapa.

    Quién decide normalizar es el DIAL, no la confianza del STT: 'español' no se normaliza;
    'inglés'/'otro' sí.

    Yields:
        PasoPipeline por etapa ('stt', 'normalizado'?, 'paisa').

    Raises:
        ValueError: si `idioma` no es un dial válido.
        TransformacionError: si falla la normalización o la reescritura (incl. el RAG).
    """
    if idioma not in DIAL_A_CODIGO:
        raise ValueError(f"idioma '{idioma}' inválido; usá uno de {list(DIAL_A_CODIGO)}.")

    codigo = DIAL_A_CODIGO[idioma]
    transcripcion, _idioma_det, _conf = transcribir(ruta_audio, codigo)
    yield PasoPipeline("stt", transcripcion)

    if idioma == "español":
        texto_es = transcripcion  # ya está en español; no se emite paso
    else:
        texto_es = normalizar_a_espanol(transcripcion)
        yield PasoPipeline("normalizado", texto_es)

    paisa = transformar_a_paisa(texto_es, exageracion, registro)
    yield PasoPipeline("paisa", paisa)


def audio_a_paisa(ruta_audio: str, idioma: str, exageracion: int, registro: str) -> ResultadoTexto:
    """Conveniencia no-streaming: consume pasos_audio_a_paisa() y devuelve todo el resultado."""
    transcripcion = texto_es = paisa = ""
    se_normalizo = False
    for paso in pasos_audio_a_paisa(ruta_audio, idioma, exageracion, registro):
        if paso.etapa == "stt":
            transcripcion = texto_es = paso.texto
        elif paso.etapa == "normalizado":
            texto_es = paso.texto
            se_normalizo = True
        elif paso.etapa == "paisa":
            paisa = paso.texto
    return ResultadoTexto(transcripcion, texto_es, paisa, se_normalizo)
