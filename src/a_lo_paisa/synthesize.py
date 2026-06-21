"""Etapa de TTS (Text-To-Speech) con voz clonada usando Chatterbox Multilingual.

Por qué Chatterbox Multilingual (de Resemble AI) y no XTTS:
- Licencia MIT (permisiva), soporte explícito de español, mantenimiento activo,
  y un control de "exageración"/emoción que queremos exponer como slider.

Conceptos clave (porque estás aprendiendo):

* Clonación de voz "zero-shot":
  No entrenamos nada. Le pasamos un audio de referencia tuyo (audio_prompt_path)
  y el modelo imita tu timbre al sintetizar el texto. Por eso necesitas grabar un
  audio limpio de tu voz en data/voice_reference.aif.

* Marca de agua PerTh (¡importante saberlo!):
  TODO audio generado por Chatterbox lleva incrustada una marca de agua neuronal
  IMPERCEPTIBLE de Resemble AI (Perth Watermarker). No la oirás, sobrevive a la
  compresión MP3, y sirve para poder identificar audio generado por IA. No es un
  bug ni algo que debas quitar: es parte del diseño responsable del modelo.

* exageración (exaggeration):
  Controla cuánta exageración/emoción mete el modelo. 0.5 es el default neutro.
  Lo exponemos como argumento porque más adelante lo conectarás a un slider de
  "exageración paisa" en la UI.

La API (clase, método, nombres de parámetros) fue verificada contra el README
oficial de resemble-ai/chatterbox antes de escribir esto, no de memoria.
"""

import os
from functools import lru_cache
from pathlib import Path

# IMPORTANTE en Apple Silicon: algunos ops de Chatterbox aún no están implementados
# en el backend MPS (GPU de Apple). Esta variable hace que torch caiga a CPU SOLO
# para esos ops puntuales, en vez de reventar con "not implemented for MPS". Hay
# que ponerla ANTES de importar torch/torchaudio (que inicializan el backend).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# torch para preguntar por el dispositivo en resolve_device(); torchaudio ("ta") para
# guardar el WAV. Ambos DESPUÉS del setdefault de MPS de arriba (inicializan el backend).
import torch
import torchaudio as ta

# La clase MULTILINGÜE vive en el submódulo mtl_tts (la inglesa está en .tts).
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

from a_lo_paisa import config

# ──────────────────────────────────────────────────────────────────────────────
# Parámetros fáciles de TUNEAR (para pelear con el acento "españolete").
# ──────────────────────────────────────────────────────────────────────────────
# t3_model: "v3" usa el checkpoint multilingüe V3 (general, 23 idiomas). Solo existe
# en la versión de Chatterbox instalada desde GitHub (no en la 0.1.7 de PyPI).
T3_MODEL = "v3"

# Language Pack de español LATINOAMERICANO (es-MX). Es un finetune del T3 (el modelo
# que gobierna pronunciación/acento) optimizado para LatAm, en un repo aparte de HF.
# Activarlo cambia el acento de "españolete" (es-es) a latino. Ponlo en False para
# volver al V3 general y comparar.
USAR_LATAM = True
LATAM_REPO = "ResembleAI/Chatterbox-Multilingual-es-mx-latam"
LATAM_T3_FILE = "t3_es_mx_latam.safetensors"
# Decoder de audio V3. OJO: el pack V3 usa "s3gen_v3.pt", que es un archivo DISTINTO
# del "s3gen.pt" base (verificado: difieren en tamaño). La receta oficial del Space
# empareja el T3 LatAm con ESTE decoder, no con el base.
LATAM_S3GEN_FILE = "s3gen_v3.pt"
# Repo base con los assets V3 compartidos que el pack NO trae (voice encoder,
# tokenizer, voz builtin). Igual que hace el Space oficial.
BASE_REPO = "ResembleAI/chatterbox"

# cfg_weight: peso del "classifier-free guidance". Bajarlo hacia 0 afloja el apego
# del modelo al prompt/locutor base y suele reducir el acento españolete. Ya lo
# tienes en 0; aquí queda visible para seguir tuneando.
CFG_WEIGHT = 0.0

# exaggeration: exageración/emoción de la voz (0.5 = neutro). Default de la función
# de bajo nivel synthesize() (para tunear a mano desde el __main__).
EXAGGERATION = 0.7

# MAPEO del DIAL de exageración (1-3, el MISMO que gobierna el texto) a los parámetros
# de Chatterbox, para que ahora gobierne también la VOZ. Regla: a más exageración, más
# `exaggeration` y MENOS `cfg_weight` (la doc recomienda bajar cfg_weight al subir la
# expresividad, para que el ritmo no se acelere).
# EDITABLE Y VISIBLE: si el nivel 3 (cfg_weight 0.0) "suelta el ritmo", subí ese
# cfg_weight acá; es el parámetro a tunear.
EXAGERACION_A_PARAMS = {
    1: {"exaggeration": 0.5,  "cfg_weight": 0.5},
    2: {"exaggeration": 0.75, "cfg_weight": 0.25},
    3: {"exaggeration": 1.0,  "cfg_weight": 0.0},
}


class SintesisError(Exception):
    """Falla al SINTETIZAR la voz (audio/dispositivo/modelo TTS).

    Es DISTINTA de la TransformacionError del texto a propósito: separa 'falló la voz'
    de 'falló el texto'. Cuando salta, el usuario ya tiene su texto paisa (la síntesis
    es el último eslabón), así que el orquestador puede mostrar el texto igual y solo
    avisar que la voz falló.
    """

# Ruta al audio de referencia con TU voz. El modelo clona el timbre de aquí.
# Debe ser un audio limpio (sin ruido/música de fondo), idealmente de varios
# segundos. Lo resolvemos respecto a la raíz del proyecto (definida en config)
# para que funcione sin importar desde dónde ejecutes.
VOICE_REFERENCE = config.PROJECT_ROOT / "data" / "voice_reference.aif"

# TTS_DEVICE: perilla de entorno. "auto" = GPU si hay, si no CPU. En local (Mac) se
# resuelve a "cpu" por el bug de MPS (ver _device_tts); en el Space con GPU, a "cuda".
TTS_DEVICE = os.getenv("TTS_DEVICE", "auto")


def resolve_device(value: str) -> str:
    """Traduce un valor de dispositivo (posiblemente "auto") a uno concreto.

    Reglas:
      - "auto" -> "cuda" si hay GPU NVIDIA; si no, "mps" en Apple Silicon; si no, "cpu".
      - cualquier otro valor ("cpu", "cuda", "mps") -> se devuelve TAL CUAL, lo que
        permite FORZAR un dispositivo desde la env var.

    (Vivía en config.py con un `import torch` PEREZOSO, porque config se importaba en
    etapas sin torch. Acá ya no hace falta: synthesize importa torch arriba.)
    """
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    # MPS = Metal Performance Shaders, la GPU de Apple Silicon (Mac M1/M2/M3...).
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _device_tts() -> str:
    """Elige el dispositivo para Chatterbox según dónde corramos.

    - CUDA (p. ej. el GPU de Hugging Face Spaces, destino de PRODUCCIÓN): usamos la
      GPU. Ahí no hay ningún problema.
    - Apple Silicon: NO usamos MPS a propósito. torchaudio tiene un bug al
      resamplear el audio de referencia sobre MPS ("Output channels > 65536") que
      hace inservible a Chatterbox en ese backend, así que vamos directo a CPU (sin
      intentar MPS en vano). Como el pipeline NO es tiempo real, en local CPU basta.
    - Sin GPU: CPU.

    En resumen: en tu Mac corre en CPU; en el Space con GPU, 'auto' dará 'cuda'.
    """
    device = resolve_device(TTS_DEVICE)
    if device == "mps":
        print(
            "ℹ️  Apple Silicon: uso CPU para el TTS (MPS no soporta un op de Chatterbox). En producción con GPU CUDA se usará la GPU."
        )
        return "cpu"
    return device


def _preparar_ckpt_latam() -> Path:
    """Arma un directorio con los assets del Language Pack LatAm y los devuelve.

    Chatterbox.from_local() espera TODOS los pesos en UNA carpeta. Pero el pack vive
    repartido: el T3 finetuneado (lo único específico de LatAm) está en LATAM_REPO,
    y el resto de assets V3 (voice encoder, decoder s3gen, tokenizer, voz builtin)
    están en el repo base. Replicamos lo que hace el Space oficial: bajamos cada
    archivo de su repo y los ENLAZAMOS (symlink) en una carpeta local, sin duplicar
    los pesos (cada hf_hub_download ya los cachea; solo creamos enlaces).
    """
    from huggingface_hub import hf_hub_download

    token = config.HF_TOKEN  # None si no está; los repos son públicos igualmente

    # Carpeta de ensamblado (bajo .cache/, que está en .gitignore).
    ensamblado = config.PROJECT_ROOT / ".cache" / "chatterbox_es_latam"
    ensamblado.mkdir(parents=True, exist_ok=True)

    def _enlazar(repo: str, archivo: str, nombre_destino: str) -> None:
        # Descarga (o usa caché) el archivo y crea un symlink en `ensamblado`.
        origen = hf_hub_download(repo_id=repo, filename=archivo, repo_type="model", token=token)
        destino = ensamblado / nombre_destino
        if destino.exists() or destino.is_symlink():
            destino.unlink()
        destino.symlink_to(origen)
        # Imprimimos el origen real para poder VERIFICAR qué pesos se usan.
        print(f"   {nombre_destino:38s} <- {repo}/{archivo}")

    print("Ensamblando assets del Language Pack LatAm:")
    # Assets que el pack NO trae, desde el repo base.
    _enlazar(BASE_REPO, "ve.pt", "ve.pt")
    _enlazar(BASE_REPO, "grapheme_mtl_merged_expanded_v1.json", "grapheme_mtl_merged_expanded_v1.json")
    _enlazar(BASE_REPO, "conds.pt", "conds.pt")
    # El T3 específico de LatAm (lo que gobierna el acento), desde el repo del pack.
    _enlazar(LATAM_REPO, LATAM_T3_FILE, LATAM_T3_FILE)

    # Decoder V3: usamos s3gen_v3.pt del pack (decoder REALMENTE distinto del base:
    # 328/2488 pesos difieren). Pero le faltan 2 buffers del tokenizer
    # (_mel_filters, window) que from_local() carga en modo estricto. Esos buffers
    # son deterministas (filtros mel + ventana STFT, fijos de la arquitectura), así
    # que los copiamos del s3gen.pt base. Construimos el archivo combinado UNA vez y
    # lo cacheamos en el dir de ensamblado.
    # OJO con el nombre: el archivo DEBE llamarse "s3gen.pt" porque from_local() lo
    # tiene hardcodeado (mtl_tts.py: torch.load(ckpt_dir / "s3gen.pt")). El NOMBRE es
    # s3gen.pt, pero el CONTENIDO que guardamos abajo son los pesos del decoder V3
    # (sd_v3), no los del base. No confundir el nombre del archivo con el decoder.
    destino_s3gen = ensamblado / "s3gen.pt"
    if destino_s3gen.is_symlink():
        destino_s3gen.unlink()  # limpia un symlink de un intento anterior
    if not destino_s3gen.exists():
        import torch

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
    """Confirma que cargamos el modelo MULTILINGÜE correcto con la versión LatAm.

    Chequeos:
      - es la clase multilingüe (no la inglesa monolingüe);
      - 'es' está entre los idiomas soportados;
      - la forma del embedding de texto del T3 = (2454, 1024), que es la del vocab
        multilingüe del checkpoint LatAm según su README (sanity check de que se
        cargó ese T3 y no otro).
    """
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
    """Carga el modelo Chatterbox una sola vez y lo reutiliza.

    Cargar Chatterbox es MUY pesado (descarga de pesos la primera vez + montaje en
    memoria/GPU), así que cachearlo con @lru_cache evita repetir ese arranque entre
    varias síntesis de una misma corrida. Es lazy: no se carga hasta el primer uso.

    Recibe el dispositivo ya elegido por _device_tts() ("cuda" o "cpu"); lo pasamos
    como argumento para que sea la clave del @lru_cache.
    """
    if USAR_LATAM:
        # Language Pack LatAm: ensamblamos los assets y cargamos con from_local,
        # apuntando t3_model al finetune LatAm. Es la vía que usa el Space oficial
        # (swap del T3 sobre los assets V3 base). El acento sale latino, no es-es.
        print(f"Cargando Chatterbox V3 + Language Pack es-MX (LatAm) en '{device}' (puede tardar)...")
        ckpt_dir = _preparar_ckpt_latam()
        modelo = ChatterboxMultilingualTTS.from_local(ckpt_dir, device, t3_model=LATAM_T3_FILE)
        _verificar_modelo_latam(modelo)
        return modelo

    # V3 general (23 idiomas). Requiere la versión de Chatterbox instalada desde
    # GitHub (master); la 0.1.7 de PyPI NO acepta t3_model.
    print(f"Cargando Chatterbox Multilingual {T3_MODEL.upper()} (general) en '{device}' (puede tardar)...")
    return ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model=T3_MODEL)


def precargar_tts() -> None:
    """Carga el modelo TTS YA (al arrancar), en vez de en el primer uso.

    En ZeroGPU es lo recomendado: colocar el modelo en 'cuda' a nivel de módulo, FUERA
    de las funciones @spaces.GPU, donde la transferencia a CUDA está optimizada (cargarlo
    perezosamente DENTRO de @spaces.GPU está desaconsejado por ineficiente). Reusa la
    caché de _get_tts, así que no recarga si ya estaba. En local solo adelanta la carga.
    """
    _get_tts(_device_tts())


def synthesize(
    texto: str,
    ruta_salida: str | Path | None = None,
    idioma: str = "es",
    exaggeration: float = EXAGGERATION,
    cfg_weight: float = CFG_WEIGHT,
) -> str:
    """Síntesis de BAJO NIVEL: recibe los floats CRUDOS de Chatterbox.

    Es la función para tunear a mano (el __main__ la usa). El pipeline usa sintetizar(),
    que traduce el dial de exageración (1-3) a estos floats. Nombramos el parámetro
    `exaggeration` (como Chatterbox) para no confundirlo con el dial entero `exageracion`.

    Args:
        texto: el texto a convertir en voz (en esta etapa ya vendría "en paisa").
        ruta_salida: ruta del archivo WAV de salida.
        idioma: código de idioma para el modelo. "es" = español (confirmado como
            soportado en el README). Internamente se pasa como `language_id`.
        exaggeration: exageración/emoción de la voz (0.5 = neutro).
        cfg_weight: apego al locutor/prompt base (bajarlo afloja el acento/ritmo).

    Returns:
        La ruta del audio escrito (como str).
    """
    # Verificación amable: si falta el audio de referencia, explicamos qué hacer
    # en vez de soltar un error críptico desde dentro del modelo.
    if not VOICE_REFERENCE.exists():
        raise FileNotFoundError(
            f"""No encuentro tu audio de referencia en '{VOICE_REFERENCE}'. 
            Graba un audio limpio de tu voz (varios segundos, sin ruido) y guárdalo ahí."""
        )

    model = _get_tts(_device_tts())

    # generate() devuelve un tensor de audio. Parámetros verificados:
    #   - language_id: el idioma (aquí "es").
    #   - audio_prompt_path: el audio de referencia para clonar tu voz.
    #   - exaggeration: la exageración/emoción.
    #   - cfg_weight: apego al locutor/prompt base (bajarlo afloja el acento).
    audio = model.generate(
        texto,
        language_id=idioma,
        audio_prompt_path=str(VOICE_REFERENCE),
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )

    if ruta_salida is None:
        ruta_salida = config.OUTPUT_DIR

    # Aseguramos que la carpeta de salida exista antes de escribir.
    salida = Path(ruta_salida)
    salida.parent.mkdir(parents=True, exist_ok=True)

    # model.sr es la frecuencia de muestreo (sample rate) propia del modelo; hay
    # que pasarla a ta.save para que el audio suene a la velocidad correcta.
    ta.save(str(salida), audio, model.sr)
    print(f"Audio escrito en: {salida}")
    return str(salida)


def _params_por_exageracion(exageracion: int) -> dict:
    """Traduce el dial de exageración (1-3) a los parámetros de Chatterbox.

    Lee del dict editable EXAGERACION_A_PARAMS. Si llega un valor fuera de 1-3, cae al
    nivel 2 (cotidiano) en vez de reventar.
    """
    return EXAGERACION_A_PARAMS.get(exageracion, EXAGERACION_A_PARAMS[2])


def sintetizar(texto: str, ruta_salida: str, exageracion: int = 2) -> str:
    """Contrato del TTS para el PIPELINE: sintetiza `texto`, guarda el .wav y devuelve la ruta.

    `exageracion` es el MISMO dial (1-3) que gobierna el texto; acá gobierna también la
    VOZ (lo mapeamos a exaggeration/cfg_weight). El modelo se carga UNA sola vez (vía
    _get_tts cacheado), igual que el STT: es lo más pesado del pipeline.

    Cualquier fallo de síntesis se envuelve en SintesisError —DISTINTA de la
    TransformacionError del texto— para que el orquestador distinga 'falló la voz' de
    'falló el texto' y avise sin perder el texto paisa ya generado.

    Returns:
        La ruta del .wav escrito (== ruta_salida).
    """
    p = _params_por_exageracion(exageracion)
    try:
        # Reusa synthesize() (bajo nivel): carga del modelo cacheada, guardado y return.
        return synthesize(
            texto,
            ruta_salida,
            exaggeration=p["exaggeration"],
            cfg_weight=p["cfg_weight"],
        )
    except Exception as e:
        # Falló la VOZ (audio/dispositivo/modelo): error claro y separado del texto.
        # `from e` conserva la traza original para depurar.
        raise SintesisError(f"No se pudo sintetizar la voz: {e}") from e


# ──────────────────────────────────────────────────────────────────────────────
# Prueba manual desde la terminal.
#
# Uso:
#   uv run python -m a_lo_paisa.synthesize
# Requiere que exista data/voice_reference.aif con tu voz.
# La PRIMERA ejecución descargará los pesos de Chatterbox (varios GB).
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    texto_prueba = "Ey parcero, ¿bien o qué? Esto es una prueba de mi voz clonada."
    #texto_prueba = "Ey parsero, ¿bien o qué? Esta es una prueba de mi vos clonada." # versión latina con truco
    destino = config.PROJECT_ROOT / "outputs" / "prueba_tts_latam.wav"
    print(f"Sintetizando texto de prueba en: {destino}")
    synthesize(texto_prueba, str(destino))
