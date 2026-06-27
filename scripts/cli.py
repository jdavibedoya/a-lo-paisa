"""Entrypoint de línea de comandos del pipeline.

Parsea argumentos, consume el pipeline y muestra cada paso.
Uso: uv run python scripts/cli.py --audio voz_de_entrada.wav --idioma español
"""

import argparse
import sys
from importlib.metadata import metadata

from a_lo_paisa import config, pipeline
from a_lo_paisa.provider import TransformacionError
from a_lo_paisa.synthesize import SintesisError, warm_up


def main() -> None:
    parser = argparse.ArgumentParser(description=metadata("a-lo-paisa")["Summary"])
    parser.add_argument("--audio", required=True, help="Ruta del audio de entrada.")
    parser.add_argument(
        "--idioma",
        choices=list(pipeline.DIAL_A_CODIGO),
        default="español",
        help="Dial de idioma de entrada (default: español).",
    )
    parser.add_argument("--exageracion", type=int, default=2,
                        help="1 suave | 2 cotidiano | 3 recargado: controla transformación y voz (default: 2).")
    parser.add_argument("--registro", default="montañero", help="urbano | montañero (default: montañero).")
    parser.add_argument(
        "--salida",
        default=str(config.PROJECT_ROOT / "outputs" / "pipeline_cli_test.wav"),
        help="Ruta del .wav de salida (default: outputs/pipeline_cli_test.wav).",
    )
    args = parser.parse_args()
    warm_up()  # precarga el TTS mientras corren STT/LLM.

    # Consumimos el pipeline, mostrando cada paso al correr.
    # Portero: si el texto falla, no se llega al TTS.
    try:
        for paso in pipeline.pasos_pipeline(args.audio, args.idioma, args.exageracion, args.registro, args.salida):
            if paso.etapa == "traducción":
                print(f"Traducción: {paso.texto}")
            elif paso.etapa == "reescritura":
                print(f"\nTransformación: {paso.texto}")
            elif paso.etapa == "tts":
                print(f"\nAudio guardado en: {paso.texto}")
    except TransformacionError as e:
        print(f"\n⚠️  No se pudo procesar el mensaje: {e}")
        sys.exit(1)
    except SintesisError as e:
        print(f"\n⚠️  La síntesis de voz falló: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
