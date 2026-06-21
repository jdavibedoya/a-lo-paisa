"""Transformación a paisa: la capa de GENERACIÓN (LLM) del pipeline.

Acá vive lo de generar con el LLM: el prompt de transformación, los few-shot, la llamada
con reintentos, y los dos pasos que usan el LLM (transformar_a_paisa y normalizar_a_espanol).
La recuperación (RAG) vive en retrieve.py; el cliente y la excepción común, en llm.py.
"""

import sys
import textwrap
import time

from google import genai
from google.genai import types, errors

from a_lo_paisa.llm import MODEL, TransformacionError, get_client
from a_lo_paisa.retrieve import formatear_contexto, lookup_paisa

# Temperaturas separadas por tarea: normalización casi determinista (traducción fiel, que
# no invente); transformación con algo de chispa pero controlada (debajo de 1.0).
TEMP_NORMALIZACION = 0.2
TEMP_TRANSFORMACION = 0.7

MAX_REINTENTOS = 4
BACKOFF_SEGUNDOS = 8.0

# Few-shot (exageración, registro, entrada neutra, salida paisa), cubriendo las 4 esquinas
# del espectro: niveles {1,3} × registros {urbano, montañero}, para que el modelo vea el
# contraste entre niveles y registros.
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
    """Etiqueta una entrada como '[exageración N · registro R]\\n<texto>'.

    Esa etiqueta le dice al modelo qué nivel/registro aplicar, y hace que los few-shot (que
    mezclan niveles y registros) tengan sentido.
    """
    reg = registro or "general"
    return f"[exageración {exageracion} · registro {reg}]\n{texto}"


def construir_prompt_transformacion(glosario_contexto: str | None = None) -> str:
    """Arma el system prompt de producción.

    No fija nivel/registro: eso viaja etiquetado en cada turno del usuario. El bloque del
    glosario solo se incluye si hay contexto recuperado, y se enmarca como vocabulario
    DISPONIBLE (no obligatorio): en pruebas, el modelo se "encerraba" en los términos
    recuperados y sonaba rígido.
    """
    # Bloque del glosario (solo si hay contexto). Sustituimos el contexto DESPUÉS de dedent
    # (no por f-string): interpolar texto multilínea a margen 0 antes de dedent rompería el
    # cálculo del prefijo común.
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

    # El bloque se inserta con replace después de dedent (misma razón), por eso el centinela
    # {BLOQUE_GLOSARIO} y no un f-string.
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
    """Única función que habla con el modelo de generación; la reusan los dos pasos.

    `few_shot` es opcional a propósito: transformar_a_paisa la llama CON los ejemplos paisa,
    normalizar_a_espanol SIN ellos (la normalización no debe ver jerga paisa, la
    contaminaría). Reintenta ante errores transitorios (429 y 5xx) con backoff exponencial;
    ante una falla no recuperable levanta TransformacionError con un mensaje apto para el
    usuario (nunca devuelve un string "[ERROR...]").
    """
    # Few-shot como turnos alternados user->model (puede ir vacío), cada uno etiquetado.
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

        # Respuesta sin excepción: validamos que traiga texto (vacío suele ser bloqueo por filtros).
        texto = (resp.text or "").strip()
        if not texto:
            raise TransformacionError(
                "La IA devolvió una respuesta vacía (posible bloqueo por filtros de contenido)."
            )
        return texto

    # Salvaguarda (no debería alcanzarse): el bucle terminó sin return ni raise.
    raise TransformacionError("No se pudo obtener respuesta de la IA tras varios intentos.")


def transformar_a_paisa(texto: str, exageracion: int = 2, registro: str | None = "montañero") -> str:
    """Recupera contexto del glosario (RAG), arma el prompt y genera el texto paisa.

    Args:
        texto: frase neutra a reescribir.
        exageracion: 1 (sutil) a 3 (recargado). Default 2.
        registro: "urbano", "montañero" o None (= general). Default "montañero".

    Raises:
        TransformacionError: si el LLM falla de forma no recuperable.
    """
    entradas = lookup_paisa(texto)  # contexto semánticamente relevante (RAG)
    contexto = formatear_contexto(entradas)
    # El nivel/registro no van en el system prompt: viajan etiquetados en el turno del usuario.
    system_prompt = construir_prompt_transformacion(glosario_contexto=contexto)
    entrada_etiquetada = _formato_entrada(texto, exageracion, registro)
    return llamar_modelo(
        get_client(), system_prompt, entrada_etiquetada,
        model=MODEL, few_shot=FEW_SHOT, temperature=TEMP_TRANSFORMACION,
    )


def normalizar_a_espanol(texto: str) -> str:
    """Normaliza `texto` (de cualquier idioma) a español neutro (no paisa) vía Gemini.

    Es el puente STT -> reescritura para los diales 'inglés'/'otro': deja todo en el mismo
    punto de partida. Cuándo se llama lo decide el dial. Sin few-shot a propósito (la
    normalización no debe ver jerga paisa).

    Raises:
        TransformacionError: si el LLM falla de forma no recuperable.
    """
    plantilla = textwrap.dedent(
        """\
        Recibes un texto transcrito de un audio. 
        Puede estar en cualquier idioma. 
        Devuelve EXACTAMENTE el mismo mensaje traducido a ESPAÑOL NEUTRO: fiel al sentido, natural y completo, conservando el significado, el tono y el tipo de mensaje (una pregunta sigue siendo pregunta, una orden sigue siendo orden). 
        NO agregues ni quites información. Devuelve ÚNICAMENTE el texto en español, sin comillas ni explicaciones."""
    )

    return llamar_modelo(get_client(), plantilla, texto, model=MODEL, temperature=TEMP_NORMALIZACION)


# Prueba manual:  uv run python -m a_lo_paisa.paisa_transform
# Frases con SINÓNIMOS fuera del glosario ('compañero' por amigo, 'billete' por dinero):
# si el RAG semántico funciona, trae igual 'amigo'/'dinero' por significado.
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
        print(f"\n❌ {e}")  # índice ausente/desfasado: mensaje accionable, sin traceback
        sys.exit(1)
    except TransformacionError as e:
        print(f"\n⚠️  {e}")  # falla no recuperable del LLM (o del embed): mensaje amable
        sys.exit(1)
