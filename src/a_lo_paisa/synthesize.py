"""TTS (Text-To-Speech) con voz clonada usando Chatterbox Multilingual.

Chatterbox (Resemble AI) hace clonación de voz zero-shot: se le pasa audio de referencia
(data/voice_reference.wav) y el modelo imita ese timbre.
El audio generado incluye PerTh, una marca de agua invisible que lo identifica como IA.
"""

import threading
from pathlib import Path

import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # clase MULTILINGÜE (la inglesa está en .tts)

from a_lo_paisa import config

# Usamos Chatterbox V3 + el Language Pack es-MX (LatAm), con un finetune del T3 (pronunciación/acento).
LATAM_REPO = "ResembleAI/Chatterbox-Multilingual-es-mx-latam"  # T3 finetuneado, s3gen
LATAM_T3_FILE = "t3_es_mx_latam.safetensors"
LATAM_S3GEN_FILE = "s3gen_v3.pt"  # decoder V3 del pack
BASE_REPO = "ResembleAI/chatterbox"  # ve, conds, grapheme

# Dial de exageración (mismo del texto)
# Controla la expresividad de la voz. La doc de Chatterbox recomienda bajar cfg_weight al subir exaggeration.
EXAGERACION_A_PARAMS = {
    1: {"exaggeration": 0.5,  "cfg_weight": 0.5},
    2: {"exaggeration": 0.75, "cfg_weight": 0.25},
    3: {"exaggeration": 1.0,  "cfg_weight": 0.0},
}

VOICE_REFERENCE = config.PROJECT_ROOT / "data" / "voice_reference.wav"  # Audio de referencia con la voz a clonar


class SintesisError(Exception):
    """Falla al sintetizar la voz.

    Cuando salta, el usuario ya tiene su texto paisa.
    """


def _device_tts() -> str:
    """Device para Chatterbox. En Apple Silicon fuerza CPU a propósito.

    torchaudio tiene un bug al resamplear sobre MPS; el pipeline no es tiempo real, CPU basta.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        print("ℹ️ CPU para el TTS (Chatterbox no funciona con MPS).")
    return "cpu"


def _preparar_ckpt_latam() -> Path:
    """Arma un directorio con los assets del Language Pack LatAm y lo devuelve.

    from_local() espera los pesos en una carpeta, pero el pack está repartido:
    ve/conds/grapheme en BASE_REPO, el T3 finetuneado (específico de LatAm) y el
    s3gen_v3 están en LATAM_REPO. Bajamos cada archivo y los enlazamos por symlink.
    """
    ruta_pack_ensamblado = config.PROJECT_ROOT / ".cache" / "chatterbox_es_latam"
    ruta_pack_ensamblado.mkdir(parents=True, exist_ok=True)
    ruta_s3gen = ruta_pack_ensamblado / "s3gen.pt"

    if not ruta_s3gen.exists():
        from huggingface_hub import hf_hub_download
        import librosa

        token = config.HF_TOKEN  # None si no está; los repos son públicos igualmente

        def _enlazar(repo: str, archivo: str) -> None:
            origen = hf_hub_download(repo_id=repo, filename=archivo, repo_type="model", token=token)
            destino = ruta_pack_ensamblado / archivo
            if destino.exists() or destino.is_symlink():
                destino.unlink()
            destino.symlink_to(origen)

        _enlazar(BASE_REPO, "ve.pt")
        _enlazar(BASE_REPO, "grapheme_mtl_merged_expanded_v1.json")
        _enlazar(BASE_REPO, "conds.pt")
        _enlazar(LATAM_REPO, LATAM_T3_FILE)

        # Al decoder s3gen_v3.pt le faltan 2 buffers del tokenizer (mel filterbank + ventana STFT)
        # que from_local() exige; son DSP deterministas, se calculan y se guarda el combinado.
        v3_s3gen = hf_hub_download(repo_id=LATAM_REPO, filename=LATAM_S3GEN_FILE, repo_type="model", token=token)
        sd_v3 = torch.load(v3_s3gen, map_location="cpu", weights_only=True)
        sd_v3["tokenizer._mel_filters"] = torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=128)).float()
        sd_v3["tokenizer.window"] = torch.hann_window(400)
        torch.save(sd_v3, ruta_s3gen)

        # Podamos el source s3gen_v3 (~1 GB): ya quedó en el combinado.
        fuente = Path(v3_s3gen)
        blob = fuente.resolve()
        fuente.unlink()
        blob.unlink(missing_ok=True)

    return ruta_pack_ensamblado


_tts_lock = threading.Lock()
_tts_cache: dict[str, ChatterboxMultilingualTTS] = {}


def warm_up() -> None:
    """Precarga el TTS en un hilo daemon (no bloquea), para que esté listo al primer uso.

    El lock en _get_tts evita que este hilo y el primer request inicien dos cargas a la vez.
    """
    threading.Thread(target=lambda: _get_tts(_device_tts()), daemon=True).start()


def _get_tts(device: str) -> ChatterboxMultilingualTTS:
    """Carga el modelo Chatterbox una sola vez (cacheado) y lo reusa.

    Es lazy, pero warm_up() lo precarga al arrancar.
    Usa un lock con doble chequeo (en vez de @lru_cache) para evitar la condición de carrera
    entre el warm-up y un request concurrente.
    """
    if device in _tts_cache:  # si ya está cargado, evita el costo de pedir el lock
        return _tts_cache[device]
    with _tts_lock:
        if device in _tts_cache:  # lo cargó otro hilo mientras esperábamos el lock
            return _tts_cache[device]
        print(f"Cargando Chatterbox V3 + Language Pack es-MX (LatAm) en '{device}'...")
        ckpt_dir = _preparar_ckpt_latam()
        modelo = ChatterboxMultilingualTTS.from_local(ckpt_dir, device, t3_model=LATAM_T3_FILE)
        _tts_cache[device] = modelo
        return modelo


def sintetizar(texto: str, ruta_salida: str, exageracion: int = 2) -> str:
    """Sintetiza `texto` con la voz clonada, guarda el .wav y devuelve la ruta.

    `exageracion` es el mismo dial del texto; mapea a los floats de Chatterbox (exaggeration/cfg_weight).
    Al fallar levanta SintesisError.
    """
    if not VOICE_REFERENCE.exists():
        raise SintesisError(f"No existe el audio de referencia '{VOICE_REFERENCE}'.")
    params = EXAGERACION_A_PARAMS.get(exageracion, EXAGERACION_A_PARAMS[2])
    try:
        model = _get_tts(_device_tts())
        audio = model.generate(texto, language_id="es", audio_prompt_path=str(VOICE_REFERENCE), **params)
        salida = Path(ruta_salida)
        salida.parent.mkdir(parents=True, exist_ok=True)
        ta.save(str(salida), audio, model.sr)  # model.sr = sample rate propio del modelo
        print(f"Audio escrito en: {salida}")
        return str(salida)
    except Exception as e:
        raise SintesisError(f"No se pudo sintetizar la voz: {e}") from e


# Prueba independiente:  uv run python -m a_lo_paisa.synthesize
if __name__ == "__main__":
    texto_prueba = "¡Hágale pues mijo, vení ligerito, esto es una berraquera!"
    destino = config.PROJECT_ROOT / "outputs" / "tts_test.wav"
    print(f"Sintetizando texto de prueba en: {destino}")
    sintetizar(texto_prueba, str(destino))
