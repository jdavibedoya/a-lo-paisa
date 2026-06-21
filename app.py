"""Interfaz Gradio de 'A lo Paisa'. Se empaqueta como Space Docker (CPU) en Hugging Face.

El handler es un GENERADOR: emite los pasos intermedios (STT -> Normalizado -> Paisa) a
medida que salen, y al final el audio, así el usuario ve el texto rápido antes de la parte
lenta (la voz). El "portero" se respeta: si el texto falla, no se sintetiza.

El TTS se precarga al arrancar (warm-up en un hilo, no bloquea la UI) y se reusa cacheado
con lock; ver synthesize._get_tts.
"""

import tempfile
import threading

import gradio as gr

from a_lo_paisa import pipeline, synthesize
from a_lo_paisa.llm import TransformacionError
from a_lo_paisa.synthesize import SintesisError, sintetizar

# Plantilla del panel de resultado al arrancar (para que no se vea vacío).
PLACEHOLDER = "🗣️🎙️ **Transcripción STT:** …\n\n⛰️🫓 **Paisa LLM:** …\n\n 💬🤖 **Síntesis TTS**:"


def _sintetizar_a_temp(paisa: str, exageracion: int) -> str:
    """Sintetiza `paisa` a un .wav temporal y devuelve su ruta."""
    tmp = tempfile.NamedTemporaryFile(prefix="paisa_", suffix=".wav", delete=False)
    return sintetizar(paisa, tmp.name, exageracion)


def procesar(audio_path, idioma, exageracion, registro):
    """Handler generador: emite (pasos, audio, listo) en cada yield.

    El audio queda en None hasta que la voz está lista; 'listo' (el "✅ ¡Listo!" debajo del
    audio) hasta el final. Portero: si el texto falla, cortamos antes del TTS; si falla la
    voz, el texto ya quedó mostrado arriba.
    """
    if not audio_path:
        yield "⚠️ **Subí o grabá un audio primero.**", None, ""
        return

    exageracion = int(exageracion)  # el slider entrega float
    lineas: list[str] = []

    def render(estado: str = "") -> str:
        cuerpo = "\n\n".join(lineas)
        if estado:
            cuerpo = f"{cuerpo}\n\n{estado}" if cuerpo else estado
        return cuerpo

    # 1) TEXTO: STT -> (normalización) -> paisa, mostrando cada paso al salir.
    paisa = None
    try:
        yield render("⏳ Transcribiendo…"), None, ""
        for paso in pipeline.pasos_audio_a_paisa(audio_path, idioma, exageracion, registro):
            if paso.etapa == "stt":
                lineas.append(f"🗣️🎙️ **Transcripción STT:** {paso.texto}")
                yield render("⏳ Procesando…"), None, ""
            elif paso.etapa == "normalizado":
                lineas.append(f"✨ **Traducción:** {paso.texto}")
                yield render("⏳ Transformando…"), None, ""
            elif paso.etapa == "paisa":
                lineas.append(f"⛰️🫓 **Paisa LLM:** {paso.texto}")
                paisa = paso.texto
                yield render(" ⏳ Sintetizando…"), None, ""
    except TransformacionError as e:
        lineas.append(f"⚠️ **No se pudo procesar el mensaje:** {e}")
        yield render(), None, ""
        return

    # 2) VOZ. El texto ya se mostró; si la voz falla, igual queda arriba.
    try:
        ruta = _sintetizar_a_temp(paisa, exageracion)
    except SintesisError as e:
        lineas.append(f"⚠️ **La síntesis de voz falló**: {e}")
        yield render(), None, ""
        return

    # Todo cargó: header de síntesis en el panel + el audio + "✅ ¡Listo!" debajo.
    yield render("💬🤖 **Síntesis TTS:**"), ruta, "✅ **¡Listo!**"


with gr.Blocks(title="A lo Paisa") as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("Grabá o subí un audio y te lo devuelvo **a lo paisa**.")
            audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Voz de entrada")
            idioma = gr.Dropdown(choices=list(pipeline.DIAL_A_CODIGO), value="español", label="Idioma de entrada")
            registro = gr.Dropdown(choices=["montañero", "urbano"], value="montañero", label="Registro")
            exageracion = gr.Slider(1, 3, value=2, step=1, label="Exageración (1 suave - 2 cotidiano - 3 recargado)")
            boton = gr.Button("¡Hágale pues!", variant="primary")
        with gr.Column():
            pasos = gr.Markdown(value=PLACEHOLDER)
            audio_out = gr.Audio(label="Voz de salida", type="filepath")
            listo = gr.Markdown()  # "✅ ¡Listo!" debajo del audio, solo al terminar

    # concurrency_limit=1: procesamos un audio a la vez. Ya es el default de Gradio, pero
    # explícito deja clara la intención (y no depende de que ese default no cambie).
    boton.click(
        procesar,
        [audio_in, idioma, exageracion, registro],
        [pasos, audio_out, listo],
        concurrency_limit=1,
    )


if __name__ == "__main__":
    # Warm-up: cargamos el TTS al arrancar en un hilo aparte (NO bloquea la UI), para
    # aprovechar el rato que el usuario tarda grabando. El lock en _get_tts evita que este
    # hilo y el primer request inicien dos cargas a la vez.
    threading.Thread(target=lambda: synthesize._get_tts(synthesize._device_tts()), daemon=True).start()
    # launch() sin host/puerto fijos: en local Gradio busca un puerto libre (no choca con
    # otra instancia). En el Space (Docker) el host/puerto los fija el Dockerfile con
    # GRADIO_SERVER_NAME / GRADIO_SERVER_PORT (0.0.0.0:7860), que Gradio respeta solo.
    demo.launch()
