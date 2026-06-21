"""Interfaz Gradio del pipeline 'A lo Paisa', pensada para Hugging Face Spaces (ZeroGPU).

Diseño pensado para ZeroGPU (GPU compartida que se asigna por-llamada):
  - SOLO el TTS necesita GPU. El STT (CPU) y la normalización/reescritura (API Gemini)
    NO la necesitan, así que corren FUERA de @spaces.GPU. Envolvemos únicamente la
    síntesis, de modo que la GPU se pide solo durante la voz (menos cuota, mejor
    prioridad en la cola). Así conviven las tres etapas sin pelearse por el dispositivo.
  - El modelo TTS se PRECARGA a 'cuda' a NIVEL DE MÓDULO (al arrancar), no perezosamente
    dentro de la función GPU: fuera de @spaces.GPU hay una emulación de CUDA que permite
    esa colocación en startup, que es donde ZeroGPU optimiza la transferencia.
  - El handler es un GENERADOR: emite los pasos intermedios (STT -> Normalizado -> Paisa)
    a medida que salen, y al final el audio. El usuario ve el progreso del texto antes
    de la parte lenta (la voz).

Local vs Space: @spaces.GPU es efecto-nulo fuera de ZeroGPU, así que el MISMO código
corre en tu Mac (en CPU) y en el Space (en GPU).
"""

import os
import tempfile

# En un Space (ZeroGPU) el TTS va en GPU: fijamos el dispositivo ANTES de importar
# synthesize (que lee TTS_DEVICE en su import). En local no tocamos nada -> queda 'auto'
# (CPU en Mac). SPACE_ID lo setea Hugging Face automáticamente en cualquier Space.
if os.environ.get("SPACE_ID") and not os.environ.get("TTS_DEVICE"):
    os.environ["TTS_DEVICE"] = "cuda"

# `spaces` habilita la emulación de CUDA y el decorador @spaces.GPU en el Space. Fuera
# de ZeroGPU puede no estar instalado: si falta, definimos un decorador NO-OP para que
# la app corra igual en local (en CPU). Importarlo ANTES de precargar el modelo es lo
# que activa la emulación que permite colocar en 'cuda' al arrancar.
try:
    import spaces

    gpu = spaces.GPU
except ImportError:  # entorno local sin `spaces`
    def gpu(*args, **kwargs):
        # Soporta tanto @gpu como @gpu(duration=...): si lo llaman directo con la
        # función, la devuelve tal cual; si lo llaman con kwargs, devuelve el wrapper.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn

import gradio as gr

from a_lo_paisa import pipeline, synthesize
from a_lo_paisa.llm import TransformacionError
from a_lo_paisa.synthesize import SintesisError

# Precarga del modelo TTS SOLO en el Space (patrón ZeroGPU: colocar en 'cuda' al
# arrancar). En local lo dejamos perezoso para no pagar la carga pesada en cada
# reinicio durante el desarrollo de la UI.
if os.environ.get("SPACE_ID"):
    synthesize.precargar_tts()

# Plantilla del panel de resultado al arrancar (para que no se vea vacío): muestra el
# formato de lo que va a aparecer.
PLACEHOLDER = "🗣️🎙️ **Transcripción STT:** …\n\n⛰️🫓 **Paisa LLM:** …\n\n 💬🤖 **Síntesis TTS**:"


@gpu(duration=120)
def _sintetizar_gpu(paisa: str, exageracion: int) -> str:
    """ÚNICO paso en GPU: la síntesis de voz. Escribe un .wav temporal y devuelve su ruta.

    El modelo ya está cargado (precarga a nivel de módulo en el Space), así que acá solo
    corre generate() —la operación que de verdad usa la GPU—. duration=120 da margen para
    textos de varias frases; bajarlo mejora la prioridad en la cola de ZeroGPU.
    """
    tmp = tempfile.NamedTemporaryFile(prefix="paisa_", suffix=".wav", delete=False)
    return synthesize.sintetizar(paisa, tmp.name, exageracion)


def procesar(audio_path, idioma, exageracion, registro):
    """Handler GENERADOR: emite (pasos, audio, listo) en cada yield.

    El audio queda en None hasta que la voz está lista. El 'portero' se respeta: si el
    texto falla (TransformacionError), cortamos ANTES del TTS; si falla la VOZ, el texto
    ya quedó mostrado arriba.
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

    # Cada yield es (pasos, audio, listo). El 3er valor —el "✅ ¡Listo!" que va DEBAJO
    # del audio— queda vacío hasta el final, cuando ya cargó todo.
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

    # 2) VOZ (GPU). El texto ya se mostró; si la voz falla, igual queda arriba.
    try:
        ruta = _sintetizar_gpu(paisa, exageracion)
    except SintesisError as e:
        lineas.append(f"⚠️ **La síntesis de voz falló**: {e}")
        yield render(), None, ""
        return

    # Todo cargó: header de síntesis en el panel + el audio + "✅ ¡Listo!" DEBAJO.
    yield render("💬🤖 **Síntesis TTS:**"), ruta, "✅ **¡Listo!**"


with gr.Blocks(title="A lo Paisa") as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("Grabá o subí un audio y te lo devuelvo **a lo paisa**.")
            audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Voz de entrada")
            idioma = gr.Dropdown(
                choices=list(pipeline.DIAL_A_CODIGO), value="español", label="Idioma de entrada"
            )
            registro = gr.Dropdown(choices=["montañero", "urbano"], value="montañero", label="Registro")
            exageracion = gr.Slider(1, 3, value=2, step=1, label="Exageración (1 suave - 2 cotidiano - 3 recargado)")
            boton = gr.Button("¡Hágale pues!", variant="primary")
        with gr.Column():
            pasos = gr.Markdown(value=PLACEHOLDER)
            audio_out = gr.Audio(label="Voz de salida", type="filepath")
            listo = gr.Markdown()  # "✅ ¡Listo!" DEBAJO del audio, solo cuando todo cargó

    boton.click(procesar, [audio_in, idioma, exageracion, registro], [pasos, audio_out, listo])


if __name__ == "__main__":
    # launch() SIN host/puerto fijos: en local Gradio usa 127.0.0.1 y busca un puerto
    # libre (si 7860 está ocupado, sube a 7861…), así no choca con otra instancia. En el
    # Space (Docker) el host/puerto los fija el Dockerfile con GRADIO_SERVER_NAME /
    # GRADIO_SERVER_PORT (0.0.0.0:7860), que Gradio respeta automáticamente.
    demo.launch()
