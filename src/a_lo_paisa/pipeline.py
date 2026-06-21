"""Orquestación REUSABLE del pipeline (capa de LIBRERÍA, sin UI): audio -> texto paisa.

Dado un audio y los diales, encadena STT -> normalización condicional -> reescritura.
Es la parte que comparten TODOS los entrypoints, así que vive en el paquete (no en un
script): hoy la usan scripts/cli.py y app.py (Gradio).

A PROPÓSITO esta capa NO imprime, NO graba micrófono y NO sintetiza voz. El TTS (lo
lento y lo único que necesita GPU) lo agrega el ENTRYPOINT después, porque cada UI
presenta distinto y todas quieren MOSTRAR el texto paisa ANTES de lanzar la voz.

Dos formas de consumirla, MISMA orquestación (sin duplicar):
  - pasos_audio_a_paisa(): GENERADOR que emite cada etapa al terminarla; lo usa la UI
    con progreso (Gradio) para ir mostrando STT -> Normalizado -> Paisa.
  - audio_a_paisa(): conveniencia NO-streaming; consume el generador y devuelve el
    resultado completo de una. La usa el CLI.

El "portero" (NUNCA sintetizar un error) queda garantizado de forma estructural: si el
texto falla, el generador levanta TransformacionError ANTES de emitir el paso 'paisa',
y el caller nunca llega al TTS —de hecho NO PUEDE, porque necesita el texto paisa que
solo sale si todo fue bien—.
"""

from dataclasses import dataclass

from a_lo_paisa.paisa_transform import normalizar_a_espanol, transformar_a_paisa
from a_lo_paisa.transcribe import transcribir

# Mapeo del DIAL de idioma (lo que ve el usuario) al código de whisper. Vive ACÁ porque
# es lógica del pipeline (qué idioma forzar y si hay que normalizar), no de la UI; los
# entrypoints lo reusan para validar/ofrecer las opciones. None = autodetección ('otro').
DIAL_A_CODIGO = {"español": "es", "inglés": "en", "otro": None}


@dataclass
class PasoPipeline:
    """Una etapa intermedia YA TERMINADA del pipeline de texto, para UIs con progreso.

    `etapa` es una de: 'stt' (lo que transcribió), 'normalizado' (español neutro, solo
    si hubo que normalizar) y 'paisa' (el texto final). `texto` es su resultado.
    """

    etapa: str
    texto: str


@dataclass
class ResultadoTexto:
    """Resultado COMPLETO de la parte de texto (para callers que no streamean)."""

    transcripcion: str    # lo que oyó el STT, en el idioma original
    texto_es: str         # español neutro (== transcripcion si el dial ya era español)
    paisa: str            # el texto reescrito en paisa: esto es lo que se sintetiza
    se_normalizo: bool    # True si hubo paso de normalización (dial inglés/otro)


def pasos_audio_a_paisa(ruta_audio: str, idioma: str, exageracion: int, registro: str):
    """Generador: corre STT -> (normalización) -> reescritura, EMITIENDO cada etapa.

    Es la fuente ÚNICA de orquestación del texto. Es DETERMINISTA en quién decide
    normalizar: el DIAL de idioma, no la confianza del STT (solo informativa). 'español'
    no se normaliza; 'inglés'/'otro' sí.

    Yields:
        PasoPipeline por cada etapa terminada ('stt', 'normalizado'?, 'paisa').

    Raises:
        ValueError: si `idioma` no es un dial válido (error de programación del caller).
        TransformacionError: si falla la normalización o la reescritura (incl. el RAG).
    """
    if idioma not in DIAL_A_CODIGO:
        raise ValueError(f"idioma '{idioma}' inválido; usá uno de {list(DIAL_A_CODIGO)}.")

    # 1) STT dirigido por el dial. transcribir() imprime 'STT escuchó [es 0.98]: ...'.
    codigo = DIAL_A_CODIGO[idioma]
    transcripcion, _idioma_det, _conf = transcribir(ruta_audio, codigo)
    yield PasoPipeline("stt", transcripcion)

    # 2) Normalización DETERMINISTA por el dial (no por la confianza del STT).
    if idioma == "español":
        texto_es = transcripcion  # ya está en español; no se toca ni se emite paso.
    else:
        texto_es = normalizar_a_espanol(transcripcion)  # inglés/otro -> español neutro.
        yield PasoPipeline("normalizado", texto_es)

    # 3) Reescritura a paisa (Gemini + RAG). Puede levantar TransformacionError.
    paisa = transformar_a_paisa(texto_es, exageracion, registro)
    yield PasoPipeline("paisa", paisa)


def audio_a_paisa(ruta_audio: str, idioma: str, exageracion: int, registro: str) -> ResultadoTexto:
    """Conveniencia NO-streaming: corre el pipeline y devuelve el resultado completo.

    Consume pasos_audio_a_paisa() (la orquestación vive ahí, sin duplicar). El portero
    se respeta igual: si el texto falla, propaga TransformacionError antes de devolver.
    """
    transcripcion = texto_es = paisa = ""
    se_normalizo = False
    for paso in pasos_audio_a_paisa(ruta_audio, idioma, exageracion, registro):
        if paso.etapa == "stt":
            transcripcion = texto_es = paso.texto  # default: texto_es == transcripcion
        elif paso.etapa == "normalizado":
            texto_es = paso.texto
            se_normalizo = True
        elif paso.etapa == "paisa":
            paisa = paso.texto
    return ResultadoTexto(transcripcion, texto_es, paisa, se_normalizo)
