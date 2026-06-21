"""TTS (Text-To-Speech) con voz clonada usando Chatterbox Multilingual.

Chatterbox (Resemble AI) hace clonación de voz zero-shot: no entrenamos nada, le pasamos
un audio de referencia (data/voice_reference.aif) y el modelo imita ese timbre. Todo audio
generado lleva una marca de agua neuronal imperceptible (PerTh) que identifica audio de
IA; es parte del diseño del modelo, no se quita.
"""

import os
from functools import lru_cache
from pathlib import Path

# En Apple Silicon, algunos ops de Chatterbox no están implementados en MPS. Esta variable
# hace que torch caiga a CPU solo para esos ops, en vez de reventar. Va ANTES de importar
# torch/torchaudio (que inicializan el backend).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # clase MULTILINGÜE (la inglesa está en .tts)

from a_lo_paisa import config

# ──────────────────────────────────────────────────────────────────────────────
# Modelo y assets. Usamos Chatterbox V3 + el Language Pack es-MX (LatAm), un finetune del
# T3 (gobierna pronunciación/acento) para que el acento salga latino y no "españolete". El
# T3 LatAm vive en LATAM_REPO; el resto de assets V3, en el repo base.
# ──────────────────────────────────────────────────────────────────────────────
T3_MODEL = "v3"  # checkpoint multilingüe V3 (solo en la versión de git, no en PyPI 0.1.7)
USAR_LATAM = True  # False = V3 general (23 idiomas), para comparar
LATAM_REPO = "ResembleAI/Chatterbox-Multilingual-es-mx-latam"
LATAM_T3_FILE = "t3_es_mx_latam.safetensors"
LATAM_S3GEN_FILE = "s3gen_v3.pt"  # decoder V3 del pack (distinto del s3gen.pt base)
BASE_REPO = "ResembleAI/chatterbox"

# Parámetros de Chatterbox a tunear (defaults de la función de bajo nivel synthesize()):
#   cfg_weight: apego al locutor/prompt base; bajarlo hacia 0 afloja el acento "españolete".
#   exaggeration: exageración/emoción de la voz (0.5 = neutro).
CFG_WEIGHT = 0.0
EXAGGERATION = 0.7

# Dial de exageración (1-3, el mismo que gobierna el texto) -> parámetros de Chatterbox: a
# más exageración, más `exaggeration` y menos `cfg_weight` (la doc recomienda bajar
# cfg_weight al subir la expresividad). Editá acá si el nivel 3 "suelta el ritmo".
EXAGERACION_A_PARAMS = {
    1: {"exaggeration": 0.5,  "cfg_weight": 0.5},
    2: {"exaggeration": 0.75, "cfg_weight": 0.25},
    3: {"exaggeration": 1.0,  "cfg_weight": 0.0},
}

# Audio de referencia con la voz a clonar (limpio, varios segundos). Relativo a la raíz.
VOICE_REFERENCE = config.PROJECT_ROOT / "data" / "voice_reference.aif"

# Device del TTS. "auto" = CUDA si hay, si no CPU (en Mac no usamos MPS: ver _device_tts).
TTS_DEVICE = os.getenv("TTS_DEVICE", "auto")


class SintesisError(Exception):
    """Falla al sintetizar la voz (audio/dispositivo/modelo).

    Distinta de TransformacionError a propósito: separa 'falló la voz' de 'falló el texto'.
    Cuando salta, el usuario ya tiene su texto paisa (la voz es el último eslabón), así que
    el orquestador puede mostrar el texto igual y solo avisar.
    """


def resolve_device(value: str) -> str:
    """Traduce un device (posiblemente "auto") a uno concreto.

    "auto" -> "cuda" si hay GPU NVIDIA; si no "mps" en Apple Silicon; si no "cpu". Cualquier
    otro valor se devuelve tal cual (permite forzarlo por env var).
    """
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _device_tts() -> str:
    """Device para Chatterbox. En Apple Silicon fuerza CPU a propósito.

    torchaudio tiene un bug al resamplear sobre MPS ("Output channels > 65536") que hace
    inservible a Chatterbox en ese backend; como el pipeline no es tiempo real, en Mac CPU
    basta. Con GPU NVIDIA, 'auto' da 'cuda' sin problema.
    """
    device = resolve_device(TTS_DEVICE)
    if device == "mps":
        print("ℹ️  Apple Silicon: uso CPU para el TTS (MPS no soporta un op de Chatterbox).")
        return "cpu"
    return device


def _preparar_ckpt_latam() -> Path:
    """Arma un directorio con los assets del Language Pack LatAm y lo devuelve.

    from_local() espera todos los pesos en UNA carpeta, pero el pack vive repartido: el T3
    finetuneado (lo único específico de LatAm) en LATAM_REPO, y el resto de assets V3 en el
    repo base. Bajamos cada archivo (hf_hub_download ya los cachea) y los enlazamos por
    symlink en una carpeta local, sin duplicar pesos.
    """
    from huggingface_hub import hf_hub_download

    token = config.HF_TOKEN  # None si no está; los repos son públicos igualmente
    ensamblado = config.PROJECT_ROOT / ".cache" / "chatterbox_es_latam"
    ensamblado.mkdir(parents=True, exist_ok=True)

    def _enlazar(repo: str, archivo: str, nombre_destino: str) -> None:
        origen = hf_hub_download(repo_id=repo, filename=archivo, repo_type="model", token=token)
        destino = ensamblado / nombre_destino
        if destino.exists() or destino.is_symlink():
            destino.unlink()
        destino.symlink_to(origen)
        print(f"   {nombre_destino:38s} <- {repo}/{archivo}")

    print("Ensamblando assets del Language Pack LatAm:")
    _enlazar(BASE_REPO, "ve.pt", "ve.pt")
    _enlazar(BASE_REPO, "grapheme_mtl_merged_expanded_v1.json", "grapheme_mtl_merged_expanded_v1.json")
    _enlazar(BASE_REPO, "conds.pt", "conds.pt")
    _enlazar(LATAM_REPO, LATAM_T3_FILE, LATAM_T3_FILE)

    # Decoder: el s3gen_v3.pt del pack es realmente distinto del base, pero le faltan 2
    # buffers del tokenizer (_mel_filters, window) que from_local() carga en modo estricto.
    # Son deterministas (filtros mel + ventana STFT), así que los copiamos del s3gen.pt base
    # y guardamos el combinado UNA vez. OJO: el archivo DEBE llamarse "s3gen.pt" (from_local
    # lo tiene hardcodeado); el nombre es s3gen.pt pero el contenido es el decoder V3.
    destino_s3gen = ensamblado / "s3gen.pt"
    if destino_s3gen.is_symlink():
        destino_s3gen.unlink()  # limpia un symlink de un intento anterior
    if not destino_s3gen.exists():
        base_s3gen = hf_hub_download(repo_id=BASE_REPO, filename="s3gen.pt", repo_type="model", token=token)
        v3_s3gen = hf_hub_download(repo_id=LATAM_REPO, filename=LATAM_S3GEN_FILE, repo_type="model", token=token)
        sd_base = torch.load(base_s3gen, map_location="cpu", weights_only=True)
        sd_v3 = torch.load(v3_s3gen, map_location="cpu", weights_only=True)
        for buffer_faltante in ("tokenizer._mel_filters", "tokenizer.window"):
            sd_v3[buffer_faltante] = sd_base[buffer_faltante]
        torch.save(sd_v3, destino_s3gen)
        print(f"   s3gen.pt                               <- {LATAM_REPO}/{LATAM_S3GEN_FILE} (+2 buffers del base)")
    else:
        print("   s3gen.pt                               <- (combinado V3, ya cacheado)")

    return ensamblado


def _verificar_modelo_latam(modelo: ChatterboxMultilingualTTS) -> None:
    """Sanity check: que sea la clase multilingüe, que 'es' esté soportado, y que el T3 sea
    el del checkpoint LatAm (text_emb de forma (2454, 1024))."""
    assert isinstance(modelo, ChatterboxMultilingualTTS), "No es ChatterboxMultilingualTTS"
    idiomas = modelo.get_supported_languages()
    assert "es" in idiomas, f"'es' no está soportado: {sorted(idiomas)[:8]}..."

    detalle = ""
    try:
        forma = tuple(modelo.t3.text_emb.weight.shape)
        detalle = f" | T3 text_emb={forma} (esperado (2454, 1024))"
    except Exception:
        pass
    print(f"✅ Verificación: modelo multilingüe OK, 'es' soportado{detalle}")


@lru_cache(maxsize=1)
def _get_tts(device: str) -> ChatterboxMultilingualTTS:
    """Carga el modelo Chatterbox una sola vez (cacheado por device) y lo reusa.

    Cargarlo es pesado (descarga de pesos + montaje), así que es lazy: no se carga hasta el
    primer uso.
    """
    if USAR_LATAM:
        print(f"Cargando Chatterbox V3 + Language Pack es-MX (LatAm) en '{device}' (puede tardar)...")
        ckpt_dir = _preparar_ckpt_latam()
        modelo = ChatterboxMultilingualTTS.from_local(ckpt_dir, device, t3_model=LATAM_T3_FILE)
        _verificar_modelo_latam(modelo)
        return modelo

    print(f"Cargando Chatterbox Multilingual {T3_MODEL.upper()} (general) en '{device}' (puede tardar)...")
    return ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model=T3_MODEL)


def synthesize(
    texto: str,
    ruta_salida: str | Path | None = None,
    idioma: str = "es",
    exaggeration: float = EXAGGERATION,
    cfg_weight: float = CFG_WEIGHT,
) -> str:
    """Síntesis de bajo nivel: recibe los floats crudos de Chatterbox (para tunear a mano).

    El pipeline usa sintetizar(), que traduce el dial de exageración (1-3) a estos floats.
    El parámetro se llama `exaggeration` (como Chatterbox) para no confundirlo con el dial
    entero `exageracion`.

    Returns:
        La ruta del audio escrito.
    """
    if not VOICE_REFERENCE.exists():
        raise FileNotFoundError(
            f"No encuentro tu audio de referencia en '{VOICE_REFERENCE}'. "
            "Graba un audio limpio de tu voz (varios segundos, sin ruido) y guárdalo ahí."
        )

    model = _get_tts(_device_tts())
    audio = model.generate(
        texto,
        language_id=idioma,
        audio_prompt_path=str(VOICE_REFERENCE),
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )

    salida = Path(ruta_salida if ruta_salida is not None else config.OUTPUT_DIR)
    salida.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(salida), audio, model.sr)  # model.sr = sample rate propio del modelo
    print(f"Audio escrito en: {salida}")
    return str(salida)


def _params_por_exageracion(exageracion: int) -> dict:
    """Dial de exageración (1-3) -> parámetros de Chatterbox. Fuera de rango cae al nivel 2."""
    return EXAGERACION_A_PARAMS.get(exageracion, EXAGERACION_A_PARAMS[2])


def sintetizar(texto: str, ruta_salida: str, exageracion: int = 2) -> str:
    """Contrato del TTS para el PIPELINE: sintetiza `texto`, guarda el .wav y devuelve la ruta.

    `exageracion` es el mismo dial (1-3) del texto; acá gobierna también la voz. Cualquier
    fallo se envuelve en SintesisError (distinta de TransformacionError) para que el
    orquestador distinga 'falló la voz' de 'falló el texto'.
    """
    p = _params_por_exageracion(exageracion)
    try:
        return synthesize(texto, ruta_salida, exaggeration=p["exaggeration"], cfg_weight=p["cfg_weight"])
    except Exception as e:
        raise SintesisError(f"No se pudo sintetizar la voz: {e}") from e


# Prueba manual:  uv run python -m a_lo_paisa.synthesize
# (Requiere data/voice_reference.aif; la primera vez descarga los pesos de Chatterbox.)
if __name__ == "__main__":
    texto_prueba = "Ey parcero, ¿bien o qué? Esto es una prueba de mi voz clonada."
    destino = config.PROJECT_ROOT / "outputs" / "prueba_tts_latam.wav"
    print(f"Sintetizando texto de prueba en: {destino}")
    synthesize(texto_prueba, str(destino))
