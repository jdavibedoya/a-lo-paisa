"""Interfaz Gradio de 'A lo Paisa'. Se empaqueta como Space Docker (CPU) en Hugging Face.

El handler es un generador que emite resultados en cada paso.
El TTS se calienta en otro hilo antes de lanzar la interfaz
y se reusa cacheado con lock; ver synthesize._get_tts.
"""

import tempfile

import gradio as gr

from a_lo_paisa import pipeline, synthesize
from a_lo_paisa.provider import TransformacionError
from a_lo_paisa.synthesize import SintesisError

PLACEHOLDER = "🗣️🎙️ **Transcripción STT:** …\n\n⛰️🫓 **Paisa LLM:** …\n\n 💬🤖 **Síntesis TTS**:"  # Plantilla del panel de resultado al arrancar


def procesar(audio_path, idioma, exageracion, registro):
    """Handler generador: emite resultados en cada paso."""
    if not audio_path:
        yield "⚠️ **Subí o grabá un audio primero.**", None, ""
        return

    exageracion = int(exageracion)
    lineas: list[str] = []

    def render(estado: str = "") -> str:
        cuerpo = "\n\n".join(lineas)
        if estado:
            cuerpo = f"{cuerpo}\n\n{estado}" if cuerpo else estado
        return cuerpo

    ruta_salida = tempfile.NamedTemporaryFile(prefix="paisa_", suffix=".wav", delete=False).name

    # Consumimos el pipeline, mostrando cada paso al correr.
    # Portero: si el texto falla, no se llega al TTS.
    try:
        yield render("⏳ Transcribiendo…"), None, ""
        for paso in pipeline.pasos_pipeline(audio_path, idioma, exageracion, registro, ruta_salida):
            if paso.etapa == "stt":
                lineas.append(f"🗣️🎙️ **Transcripción STT:** {paso.texto}")
                yield render("⏳ Procesando…"), None, ""
            elif paso.etapa == "traducción":
                lineas.append(f"✨ **Traducción:** {paso.texto}")
                yield render("⏳ Transformando…"), None, ""
            elif paso.etapa == "reescritura":
                lineas.append(f"⛰️🫓 **Paisa LLM:** {paso.texto}")
                yield render(" ⏳ Sintetizando…"), None, ""
            elif paso.etapa == "tts":
                yield render("💬🤖 **Síntesis TTS:**"), paso.texto, "✅ **¡Listo!**"
    except TransformacionError as e:
        lineas.append(f"⚠️ **No se pudo procesar el mensaje:** {e}")
        yield render(), None, ""
    except SintesisError as e:
        lineas.append(f"⚠️ **La síntesis de voz falló**: {e}")
        yield render(), None, ""


with gr.Blocks(title="A lo Paisa") as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("Grabá o subí un audio y te lo devuelvo **a lo paisa**.")
            audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Voz de entrada")
            idioma = gr.Dropdown(choices=list(pipeline.DIAL_A_CODIGO), value="español", label="Idioma de entrada")
            registro = gr.Dropdown(choices=["montañero", "urbano"], value="montañero", label="Registro")
            exageracion = gr.Slider(1, 3, value=2, step=1, label="Exageración (1 suave | 2 cotidiano | 3 recargado)")
            boton = gr.Button("¡Hágale pues!", variant="primary")
        with gr.Column():
            pasos = gr.Markdown(value=PLACEHOLDER)
            audio_out = gr.Audio(label="Voz de salida", type="filepath")
            listo = gr.Markdown()  # "✅ ¡Listo!" debajo del audio, solo al terminar

    # concurrency_limit=1: un request a la vez. Explícito aunque ya es el default de Gradio.
    boton.click(
        procesar,
        [audio_in, idioma, exageracion, registro],
        [pasos, audio_out, listo],
        concurrency_limit=1,
    )


if __name__ == "__main__":
    synthesize.warm_up()  # precarga el TTS en un hilo (no bloquea la UI) antes de servir la interfaz.
    # launch() sin host/puerto fijos: en local Gradio busca un puerto libre.
    # En el Space (Docker) host/puerto los fija el Dockerfile.
    demo.launch()
