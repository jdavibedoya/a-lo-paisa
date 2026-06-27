"""Pre-descarga los modelos (TTS + STT) — se corre en el build del Docker.

Hornea los pesos en la imagen para evitar redescargas en runtime:
acelera el cold-start (despertar del Space).
"""

from a_lo_paisa import synthesize, transcribe


def main() -> None:
    transcribe._descargar_pesos(transcribe.STT_MODEL_SIZE)  # STT: baja los pesos sin instanciar el modelo.
    synthesize._preparar_ckpt_latam()  # TTS: ensambla el Language Pack LatAm.
    print("✅ Modelos horneados en la imagen (STT + TTS).")


if __name__ == "__main__":
    main()
