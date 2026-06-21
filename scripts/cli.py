"""Entrypoint de línea de comandos del pipeline completo (voz -> voz).

Capa delgada: parsea argumentos, consigue el audio (archivo o micrófono), llama a la
orquestación reusable (pipeline.audio_a_paisa), muestra el texto y recién entonces
sintetiza. Toda la lógica del pipeline vive en el paquete; acá solo lo propio del CLI
(argparse, micrófono, prints). El "portero" se ve abajo: si el texto falla, no se llega
al TTS; un fallo de voz es posterior y para entonces el texto ya se mostró.

Uso:
    uv run python scripts/cli.py --audio voz.wav --idioma español
    uv run python scripts/cli.py --idioma inglés          # graba del micrófono
"""

import argparse
import sys
import tempfile
import wave

from a_lo_paisa import config, pipeline
from a_lo_paisa.llm import TransformacionError
from a_lo_paisa.synthesize import SintesisError, sintetizar

SR_GRABACION = 16000  # 16 kHz mono, el estándar de whisper


def grabar_microfono() -> str:
    """Graba del micrófono hasta Enter y guarda un .wav temporal.

    Importa sounddevice/numpy acá dentro a propósito: si se usa --audio, el script no
    exige tener micrófono ni esas dependencias.
    """
    import queue

    import numpy as np
    import sounddevice as sd

    cola: queue.Queue = queue.Queue()

    def callback(indata, frames, tiempo, status):
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

    tmp = tempfile.NamedTemporaryFile(prefix="a_lo_paisa_", suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SR_GRABACION)
        wf.writeframes(audio.tobytes())
    print(f"  audio guardado en: {tmp.name}")
    return tmp.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline completo: STT -> normalización -> paisa -> TTS.")
    parser.add_argument("--audio", help="Ruta a un .wav. Si falta, GRABA del micrófono.")
    parser.add_argument(
        "--idioma",
        choices=list(pipeline.DIAL_A_CODIGO),  # misma fuente que usa el pipeline
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

    ruta_audio = args.audio or grabar_microfono()

    # TEXTO (portero: si falla acá, cortamos antes de tocar el TTS).
    try:
        res = pipeline.audio_a_paisa(ruta_audio, args.idioma, args.exageracion, args.registro)
    except TransformacionError as e:
        print(f"\n⚠️  No se pudo procesar el mensaje: {e}")
        sys.exit(1)

    # Mostramos el texto antes de la voz (el TTS es lento; el usuario lo conserva aunque falle).
    if res.se_normalizo:
        print(f"Normalizado: {res.texto_es}")
    print(f"\n🗣️  Paisa: {res.paisa}")

    # VOZ (último eslabón; el mismo dial de exageración la gobierna).
    try:
        ruta = sintetizar(res.paisa, args.salida, args.exageracion)
    except SintesisError as e:
        print(f"\n⚠️  La síntesis de voz falló (tu texto paisa está arriba): {e}")
        sys.exit(1)

    print(f"\nAudio guardado en: {ruta}")


if __name__ == "__main__":
    main()
