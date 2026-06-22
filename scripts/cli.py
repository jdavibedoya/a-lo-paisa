"""Entrypoint de línea de comandos del pipeline completo (archivo de voz -> voz paisa).

Capa delgada: parsea argumentos, llama a la orquestación reusable (pipeline.audio_a_paisa),
muestra el texto y recién entonces sintetiza. Toda la lógica del pipeline vive en el
paquete; acá solo lo propio del CLI (argparse, prints). El "portero" se ve abajo: si el
texto falla, no se llega al TTS; un fallo de voz es posterior y para entonces el texto ya
se mostró. (Para grabar en vivo del micrófono está la app web; el CLI procesa un archivo.)

Uso:
    uv run python scripts/cli.py --audio voz.wav --idioma español
"""

import argparse
import sys

from a_lo_paisa import config, pipeline
from a_lo_paisa.llm import TransformacionError
from a_lo_paisa.synthesize import SintesisError, sintetizar


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline completo: STT -> normalización -> paisa -> TTS.")
    parser.add_argument("--audio", required=True, help="Ruta al .wav de entrada.")
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

    # TEXTO (portero: si falla acá, cortamos antes de tocar el TTS).
    try:
        res = pipeline.audio_a_paisa(args.audio, args.idioma, args.exageracion, args.registro)
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
