"""Transformación a paisa: la capa de GENERACIÓN del pipeline STT -> (normalización) -> paisa.

Acá vive solo lo de GENERAR con el LLM: el prompt de transformación, los few-shot, la
llamada al modelo con reintentos, y los dos pasos del pipeline que usan el LLM
(transformar_a_paisa y normalizar_a_espanol). La RECUPERACIÓN (glosario, embeddings,
búsqueda semántica) vive en retrieve.py; el cliente de Gemini y la excepción común,
en llm.py.

Es un WORKFLOW determinista A PROPÓSITO (pasos fijos orquestados por código), NO un
agente autónomo: el LLM no decide qué herramientas usar ni planifica. Igual quedó
estructurado por si algún día una necesidad REAL de autonomía lo justifica —
`lookup_paisa` (en retrieve.py) ya es una tool limpia y tipada—, pero hoy no hace falta
y agregarlo sería complejidad sin retorno.
"""

import sys
import textwrap
import time

from google import genai
from google.genai import types, errors

from a_lo_paisa.llm import MODEL, TransformacionError, get_client
from a_lo_paisa.retrieve import formatear_contexto, lookup_paisa

# ──────────────────────────────────────────────────────────────────────────────
# Defaults de la llamada de GENERACIÓN. El IDENTIFICADOR del modelo (MODEL) vive en
# llm.py y se importa arriba.
# ──────────────────────────────────────────────────────────────────────────────
# Temperaturas SEPARADAS por tarea (cada función pasa la suya a llamar_modelo):
#   - normalización = traducción FIEL a español neutro -> casi determinista, que NO
#     invente ni adorne.
#   - transformación = reescritura paisa, con algo de chispa pero CONTROLADA (por
#     debajo de 1.0, el default de Gemini, que tiende a inventar/exagerar).
TEMP_NORMALIZACION = 0.2
TEMP_TRANSFORMACION = 0.7

MAX_REINTENTOS = 4
BACKOFF_SEGUNDOS = 8.0


# ──────────────────────────────────────────────────────────────────────────────
# Few-shot etiquetados. Cada uno es (exageración, registro, entrada neutra, salida
# paisa) y cubre las 4 esquinas del espectro: niveles {1,3} × registros {urbano,
# montañero}, para que el modelo vea el CONTRASTE entre niveles y registros.
# ──────────────────────────────────────────────────────────────────────────────
FEW_SHOT = [
    (1, "urbano",     "Hola, ¿cómo estás? ¿Todo en orden?",
                      "Hola, ¿bien o qué? ¿Todo bien parce?"),
    (1, "montañero",  "Voy a tomar un café y a descansar un rato.",
                      "Me voy a tomar un tinto y a descansar un ratico."),
    (3, "urbano",     "Amigo, no tienes dinero, ¿estás aburrido?",
                      "Parcero, vos andás achilao, sin lucas, estás llevao, ¿sí o qué?"),
    (3, "montañero",  "Ven rápido a ver esto, es increíble.",
                      "¡Hágale pues mijo, vení ligerito a ver esta berraquera!"),
]


def _formato_entrada(texto: str, exageracion: int, registro: str | None) -> str:
    """Etiqueta una entrada con el nivel y registro pedidos.

    El modelo recibe cada texto como '[exageración N · registro R]\\n<texto>'. Esta
    etiqueta es la que le dice qué nivel/registro aplicar, y es la que hace que los
    few-shot (que mezclan niveles y registros) tengan sentido: el modelo ve la
    etiqueta de cada ejemplo y aprende a qué corresponde cada estilo.
    """
    reg = registro or "general"
    return f"[exageración {exageracion} · registro {reg}]\n{texto}"


def construir_prompt_transformacion(glosario_contexto: str | None = None) -> str:
    """Arma el system prompt de PRODUCCIÓN.

    No fija un nivel/registro concreto: eso viaja ETIQUETADO en cada turno del
    usuario ([exageración N · registro R]). El prompt describe la escala y los
    registros de forma general; los few-shot muestran cada combinación. El bloque del
    glosario solo se incluye cuando hay contexto recuperado, y se enmarca como
    vocabulario DISPONIBLE (no obligatorio), porque en pruebas el modelo se "encerraba"
    en los términos recuperados y sonaba rígido.
    """
    # Bloque del glosario (solo cuando hay contexto recuperado). Lo armamos con
    # dedent (indentado bajo la función, profesional) y sustituimos el contexto
    # DESPUÉS de dedent, NO por f-string: interpolar texto multilínea a margen 0
    # ANTES de dedent rompería el cálculo del prefijo común (lo dejaría en 0).
    bloque_glosario = ""
    if glosario_contexto:
        bloque_glosario = textwrap.dedent(
            """
            GLOSARIO PAISA DISPONIBLE:
            Es vocabulario de REFERENCIA, no una lista obligatoria ni exhaustiva.
            Usá los términos que encajen natural con el sentido de la frase.
            Podés usar términos de exageración menor, pero evitá los de mayor.
            Ignorá los que no peguen, y sentite libre de usar OTRAS palabras o giros paisas que no estén acá.
            La lista NO limita tu vocabulario; solo te ofrece opciones:
            {GLOSARIO_CONTEXTO}
            """
        ).replace("{GLOSARIO_CONTEXTO}", glosario_contexto)

    # Plantilla del prompt con dedent. El bloque del glosario se inserta con replace
    # DESPUÉS de dedent (misma razón de arriba). Por eso es str normal, no f-string,
    # y el centinela {BLOQUE_GLOSARIO} se sustituye al final.
    plantilla = textwrap.dedent(
        """\
        Sos un experto en el español paisa de Antioquia, Colombia (el de Medellín y sus pueblos).
        Tu tarea es REESCRIBIR el texto del usuario en habla paisa natural.
        NO traduzcas palabra por palabra: reescribí la frase entera para que suene a como hablaría un paisa, conservando el mismo significado.

        Cada texto viene ETIQUETADO así: [exageración N · registro R].
        Aplicá ese nivel y ese registro.

        REGLA INNEGOCIABLE — solo paisa de Antioquia:
        Usá únicamente jerga y entonación de Antioquia.
        NO uses marcadores de otras regiones de Colombia (NADA de 'nojoda', 'erda', 'ajá', etc.) ni de otros países.
        El voseo debe ser el antioqueño (vos vení, vos sabés), no el caleño ni el rioplatense.
        Si dudás de si algo es paisa, no lo uses.

        NIVEL DE EXAGERACIÓN (1 a 3):
        - Nivel 1 (SUTIL): español casi estándar, entendible para cualquier hispanohablante.
        Solo UN toque paisa (ocasionalmente un 'pues', un diminutivo, voseo).
        NADA de interjecciones fuertes ('¡Ave María!'), NADA de parlache marcado.
        Ante la duda en nivel 1, contenete.
        - Nivel 2 (COTIDIANO): claramente paisa pero de uso diario.
        - Nivel 3 (RECARGADO): bien paisa, con interjecciones, parlache y sabor montañero.

        REGISTRO:
        - urbano: parlache de ciudad/Medellín (ejemplos: nea, visaje, lucas, parche, paila).
        - montañero: rural/tradicional de pueblo (ejemplos: mijo, ome, berriondo, avispao).

        RASGOS DE ESTILO (modulalos según el nivel):
        - Voseo: pronominal y VERBAL ('vos' en vez de 'tú', vení, mirá, contá, vos sabés).
        - Diminutivos afectivos (ahorita, momentico, tintico): usalos con MESURA. REGLA ESTRICTA: máximo UNO en cada frase para que el resultado no sea empalagoso.
        - La partícula 'pues' y muletillas paisas (vea pues, ome, ¿sí o qué?), intercaladas con naturalidad y CON SENTIDO, nunca amontonadas ni en cada frase.
        - Podés usar ustedeo donde suene natural (un regaño, un consejo serio, o hablando con cariño familiar).
        - NO uses comas vocativas en expresiones como 'hágale pues mijo', 'todo bien parcero', 'eh Ave María ome': cortan la forma en que un paisa las pronuncia de corrido.

        FIDELIDAD AL CONTENIDO:
        - Conservá el significado y la intención original.
        NO agregués hechos, datos ni detalles que no estaban (si el texto no dice dónde ni con quién, no lo inventés).
        - Mantené el mismo tipo de mensaje: una pregunta sigue siendo pregunta, una orden sigue siendo orden.
        - Que suene natural y con un toque jocoso cuando el nivel lo permita, NUNCA forzado ni caricaturesco.

        SOBRE LOS EJEMPLOS:
        TODOS los ejemplos que veas son SOLO muestras del registro y del nivel; NO los copiés ni reutilices sus frases o muletillas.
        Cada texto es distinto: reescribilo según su propio contenido.
        Son referencia de estilo, no molde a calcar.

        {BLOQUE_GLOSARIO}

        Devolvé ÚNICAMENTE el texto reescrito: sin la etiqueta, sin comillas, sin explicaciones, sin notas ni encabezados."""
    )
    return plantilla.replace("{BLOQUE_GLOSARIO}", bloque_glosario)


def llamar_modelo(
    client: genai.Client,
    system_prompt: str,
    texto_usuario: str,
    *,
    model: str,
    few_shot: list = (),
    temperature: float,
    max_reintentos: int = MAX_REINTENTOS,
    backoff: float = BACKOFF_SEGUNDOS,
) -> str:
    """ÚNICA función que habla con el modelo de generación. La reusan los dos pasos.

    `few_shot` es OPCIONAL a propósito: transformar_a_paisa la llama CON los ejemplos
    paisa (FEW_SHOT), pero normalizar_a_espanol la llama SIN ejemplos (lista vacía),
    porque la normalización a español neutro NO debe ver jerga paisa —se la pasáramos,
    la contaminaría—. Si few_shot viene vacío, la conversación es solo el turno del
    usuario.

    Reintenta ante errores TRANSITORIOS (429 y 5xx "high demand") con backoff
    exponencial. Si la falla es NO recuperable (reintentos agotados, respuesta vacía
    o bloqueada, u otro error de API), LEVANTA TransformacionError con un mensaje apto
    para el usuario —nunca devuelve un string "[ERROR...]"—, para que el portero del
    orquestador la atrape y no la confunda con el texto generado.
    """
    # Few-shot como turnos alternados user->model (puede ir vacío). El modelo ve el
    # patrón entrada->salida antes del texto real. Cada ejemplo se ETIQUETA con su
    # nivel/registro.
    contents = []
    for exa_ej, reg_ej, entrada_ej, salida_paisa in few_shot:
        entrada_etiquetada = _formato_entrada(entrada_ej, exa_ej, reg_ej)
        contents.append(types.Content(role="user", parts=[types.Part(text=entrada_etiquetada)]))
        contents.append(types.Content(role="model", parts=[types.Part(text=salida_paisa)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=texto_usuario)]))

    cfg = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
    )

    for intento in range(1, max_reintentos + 1):
        # Espera creciente entre reintentos: backoff, backoff*2, backoff*4...
        espera = backoff * (2 ** (intento - 1))
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=cfg)

        except errors.ServerError as e:
            # 5xx (incluye el 503 "high demand"/congestión): transitorio, reintentamos.
            if intento < max_reintentos:
                print(f"    ⏳ Servidor congestionado (5xx). Reintento {intento}/{max_reintentos - 1} en {espera:.0f}s...")
                time.sleep(espera)
                continue
            raise TransformacionError(
                "El servicio de IA está congestionado y no respondió tras varios intentos. Probá de nuevo en un momento."
            ) from e

        except errors.ClientError as e:
            # ClientError cubre 4xx. El 429 = límite de tasa; también es transitorio.
            code = getattr(e, "code", None)
            if code == 429 and intento < max_reintentos:
                print(f"    ⏳ Límite de tasa (429). Reintento {intento}/{max_reintentos - 1} en {espera:.0f}s...")
                time.sleep(espera)
                continue
            if code == 429:
                raise TransformacionError(
                    "Se alcanzó el límite de uso de la IA por ahora. Esperá un momento y volvé a intentar."
                ) from e
            # Otros 4xx (400, 403, 404...) NO son transitorios: no reintentamos.
            raise TransformacionError(
                f"Hubo un problema al contactar la IA (código {code}). Revisá la configuración e intentá de nuevo."
            ) from e

        except errors.APIError as e:
            # Cualquier otro error de la API: no recuperable.
            raise TransformacionError("Hubo un problema al contactar la IA. Intentá de nuevo.") from e

        # Llegamos acá SOLO con respuesta (sin excepción): validamos que traiga texto.
        # resp.text vacío suele ser bloqueo por filtros de seguridad; no reintentamos.
        texto = (resp.text or "").strip()
        if not texto:
            raise TransformacionError(
                "La IA devolvió una respuesta vacía (posible bloqueo por filtros de contenido)."
            )
        return texto

    # Salvaguarda (no debería alcanzarse): el bucle terminó sin return ni raise.
    raise TransformacionError("No se pudo obtener respuesta de la IA tras varios intentos.")


def transformar_a_paisa(texto: str, exageracion: int = 2, registro: str | None = "montañero") -> str:
    """WORKFLOW completo: lookup_paisa -> formatear_contexto -> prompt -> generar.

    Args:
        texto: frase neutra a reescribir.
        exageracion: 1 (sutil) a 3 (recargado). Default 2.
        registro: "urbano", "montañero" o None (= general). Default "montañero".

    Returns:
        El texto reescrito en paisa.

    Raises:
        TransformacionError: si el LLM falla de forma no recuperable (lo levanta
            llamar_modelo). NO devuelve marcadores "[ERROR...]".
    """
    # 1) Recuperar contexto semánticamente relevante del glosario (capa retrieve).
    entradas = lookup_paisa(texto)
    contexto = formatear_contexto(entradas)

    # 2) Construir el prompt. El nivel/registro NO van acá: viajan etiquetados en el
    #    turno del usuario.
    system_prompt = construir_prompt_transformacion(glosario_contexto=contexto)

    # 3) Generar CON los few-shot paisa.
    entrada_etiquetada = _formato_entrada(texto, exageracion, registro)
    return llamar_modelo(
        get_client(), system_prompt, entrada_etiquetada,
        model=MODEL, few_shot=FEW_SHOT, temperature=TEMP_TRANSFORMACION,
    )


def normalizar_a_espanol(texto: str) -> str:
    """Traduce/normaliza `texto` (de CUALQUIER idioma) a ESPAÑOL NEUTRO vía Gemini.

    Es el puente STT -> agente para los diales 'inglés' y 'otro': deja el texto en
    español estándar (NO paisa) para que transformar_a_paisa parta siempre del mismo
    punto. CUÁNDO se llama lo decide el dial en el orquestador (no esta función).

    Reusa llamar_modelo SIN few-shot (lista vacía por defecto): la normalización no
    usa ejemplos paisa —se los pasáramos, la confundirían con jerga—. Por eso NO le
    pasamos FEW_SHOT.

    Raises:
        TransformacionError: si el LLM falla de forma no recuperable (lo levanta
            llamar_modelo). Es la MISMA excepción que usa transformar_a_paisa, así el
            portero del orquestador atrapa ambos pasos con un único except.
    """

    plantilla = textwrap.dedent(
        """\
        Recibes un texto transcrito de un audio. 
        Puede estar en cualquier idioma. 
        Devuelve EXACTAMENTE el mismo mensaje traducido a ESPAÑOL NEUTRO: fiel al sentido, natural y completo, conservando el significado, el tono y el tipo de mensaje (una pregunta sigue siendo pregunta, una orden sigue siendo orden). 
        NO agregues ni quites información. Devuelve ÚNICAMENTE el texto en español, sin comillas ni explicaciones."""
    )

    return llamar_modelo(get_client(), plantilla, texto, model=MODEL, temperature=TEMP_NORMALIZACION)


# ──────────────────────────────────────────────────────────────────────────────
# Prueba manual:  uv run python -m a_lo_paisa.paisa_transform
#
# Caso clave: una frase con SINÓNIMOS que NO están en el glosario ('compañero' por
# amigo, 'billete' por dinero). Si el RAG semántico funciona, debería traer igual
# las entradas 'amigo' y 'dinero' por SIGNIFICADO, cosa que la recuperación léxica
# no lograría con 'compañero'.
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        frases = [
            "mi compañero anda sin billete",
            "quedé agotado después de tanta rumba el fin de semana",
        ]
        for texto in frases:
            print("=" * 72)
            print(f"ENTRADA: {texto}")
            recuperadas = lookup_paisa(texto)
            print("RAG (semántico) trajo:", ", ".join(str(e.get("neutro")) for e in recuperadas))
            for exa in (1, 3):
                print(f"  [exageración {exa}] {transformar_a_paisa(texto, exageracion=exa)}")
    except (FileNotFoundError, RuntimeError) as e:
        # Errores de índice ausente/desfasado: mensaje accionable, sin traceback.
        print(f"\n❌ {e}")
        sys.exit(1)
    except TransformacionError as e:
        # Falla no recuperable del LLM, ya sea de generación o del embed de
        # lookup_paisa (que retrieve.py también pliega a esta excepción): mensaje
        # amable, sin traceback.
        print(f"\n⚠️  {e}")
        sys.exit(1)
