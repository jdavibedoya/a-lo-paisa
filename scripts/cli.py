"""ENTRYPOINT de línea de comandos del pipeline completo (voz -> voz).

Capa DELGADA: parsea argumentos, consigue el audio (archivo o micrófono), llama a la
orquestación reusable (a_lo_paisa.pipeline.audio_a_paisa), PRESENTA el texto y recién
entonces sintetiza la voz. Toda la lógica del pipeline vive en el paquete; acá solo
está lo propio de un CLI (argparse, micrófono, prints, códigos de salida).

El "portero" se ve explícito abajo: audio_a_paisa() hace el TEXTO y puede fallar con
TransformacionError ANTES de devolver; solo si devuelve bien llamamos al TTS, así que
NUNCA se sintetiza un error. Un fallo de VOZ (SintesisError) es posterior y distinto:
para entonces el texto paisa ya se mostró, así que solo avisamos.

Uso:
    uv run python scripts/cli.py --audio voz.wav --idioma español
    uv run python scripts/cli.py --idioma inglés          # graba del micrófono
"""

import argparse
import sys
import tempfile
import wave

from a_lo_paisa import config
from a_lo_paisa import pipeline
from a_lo_paisa.llm import TransformacionError
from a_lo_paisa.synthesize import SintesisError, sintetizar

# Frecuencia de grabación del micrófono. 16 kHz mono es el estándar de whisper.
SR_GRABACION = 16000


def grabar_microfono() -> str:
    """Graba del micrófono hasta que el usuario presione Enter; guarda un .wav temporal.

    Importamos sounddevice/numpy ACÁ DENTRO (no arriba) a propósito: así, si se usa
    --audio (un archivo), el script no exige tener micrófono/PortAudio ni la
    dependencia de grabación. Solo quien graba paga ese import.

    Capturamos con sounddevice y escribimos el WAV con `wave` (stdlib), para no
    sumar otra dependencia solo para guardar el archivo.
    """
    import queue

    import numpy as np
    import sounddevice as sd

    cola: queue.Queue = queue.Queue()

    def callback(indata, frames, tiempo, status):
        # El callback corre en un hilo aparte de sounddevice; solo acumula bloques.
        if status:
            print(f"  (aviso de audio: {status})", file=sys.stderr)
        cola.put(indata.copy())

    print("🎙️  Grabando... hablá y presioná Enter para terminar.")
    with sd.InputStream(samplerate=SR_GRABACION, channels=1, dtype="int16", callback=callback):
        input()  # bloquea el hilo principal mientras el callback acumula audio

    bloques = []
    while not cola.empty():
        bloques.append(cola.get())
    if not bloques:
        print("No se capturó audio del micrófono.", file=sys.stderr)
        sys.exit(1)
    audio = np.concatenate(bloques, axis=0)

    # Guardamos a un .wav temporal (PCM 16-bit mono).
    tmp = tempfile.NamedTemporaryFile(prefix="a_lo_paisa_", suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes por muestra
        wf.setframerate(SR_GRABACION)
        wf.writeframes(audio.tobytes())
    print(f"  audio guardado en: {tmp.name}")
    return tmp.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline completo: STT -> normalización -> paisa -> TTS.",
    )
    parser.add_argument("--audio", help="Ruta a un .wav. Si falta, GRABA del micrófono.")
    parser.add_argument(
        "--idioma",
        # Las opciones del dial salen de la MISMA fuente que usa el pipeline.
        choices=list(pipeline.DIAL_A_CODIGO),
        default="español",
        help="Dial de idioma de ENTRADA (default: español).",
    )
    parser.add_argument("--exageracion", type=int, default=2, help="Dial paisa 1-3: gobierna texto Y voz (default: 2).")
    parser.add_argument("--registro", default="montañero", help="urbano | montañero (default: montañero).")
    parser.add_argument(
        "--salida",
        default=str(config.OUTPUT_DIR / "salida_pipeline.wav"),
        help="Ruta del .wav de salida (default: outputs/salida_pipeline.wav).",
    )
    args = parser.parse_args()

    # 1) Conseguir el audio: archivo dado o grabación del micrófono.
    ruta_audio = args.audio or grabar_microfono()

    # 2) TEXTO (STT -> normalización -> paisa). El portero: si falla acá, cortamos
    #    ANTES de tocar el TTS.
    try:
        res = pipeline.audio_a_paisa(ruta_audio, args.idioma, args.exageracion, args.registro)
    except TransformacionError as e:
        print(f"\n⚠️  No se pudo procesar el mensaje: {e}")
        sys.exit(1)

    # 3) Texto OK: lo mostramos APENAS lo tenemos, antes de la voz (el TTS es lento; así
    #    el usuario ve su texto paisa de inmediato, y lo conserva aunque el TTS falle).
    if res.se_normalizo:
        print(f"Normalizado: {res.texto_es}")
    print(f"\n🗣️  Paisa: {res.paisa}")

    # 4) VOZ: el último eslabón. El MISMO dial de exageración gobierna también el TTS.
    try:
        ruta = sintetizar(res.paisa, args.salida, args.exageracion)
    except SintesisError as e:
        # Falló la VOZ, no el texto: el paisa ya está impreso arriba. Avisamos y salimos.
        print(f"\n⚠️  La síntesis de voz falló (tu texto paisa está arriba): {e}")
        sys.exit(1)

    print(f"\nAudio guardado en: {ruta}")


if __name__ == "__main__":
    main()
