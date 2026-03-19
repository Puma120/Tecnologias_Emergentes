from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import json
import os
import re
from datetime import datetime

load_dotenv()

# Archivo donde el agente guarda lo que aprende de tus gustos.
HISTORIAL_FILE = "logs/recetas_historial.json"

# Se usa para asociar feedback (ej. "le doy 4/5") a la ultima receta recomendada.
ultima_receta = None

# Estadisticas simples de uso de herramientas durante la sesion. 
tool_usage_stats = {"total": 0, "por_tool": {}}


def registrar_uso_tool(nombre_tool: str):
    tool_usage_stats["total"] += 1
    tool_usage_stats["por_tool"][nombre_tool] = tool_usage_stats["por_tool"].get(nombre_tool, 0) + 1
    herramientas_usadas = ", ".join(sorted(tool_usage_stats["por_tool"].keys()))
    print(f"[TOOL] Accedio a: {nombre_tool}")
    print(f"[TOOL] Total de accesos: {tool_usage_stats['total']}")
    print(f"[TOOL] Herramientas usadas hasta ahora: {herramientas_usadas}")


def cargar_historial() -> dict:
    os.makedirs("logs", exist_ok=True)
    if os.path.exists(HISTORIAL_FILE):
        try:
            with open(HISTORIAL_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass
    return {"historial": [], "favoritas": [], "evitar": []}


def guardar_historial(data: dict):
    os.makedirs("logs", exist_ok=True)
    with open(HISTORIAL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _guardar_feedback_directo(receta: str, calificacion: int, comentario: str):
    data = cargar_historial()
    data["historial"].append(
        {
            "timestamp": datetime.now().isoformat(),
            "receta": receta,
            "calificacion": calificacion,
            "comentario": comentario,
        }
    )

    if calificacion >= 4 and receta not in data["favoritas"]:
        data["favoritas"].append(receta)
    elif calificacion <= 2 and receta not in data["evitar"]:
        data["evitar"].append(receta)

    guardar_historial(data)


def detectar_calificacion(texto: str):
    # Ejemplos que detecta: "le doy 4", "4/5", "califico 5".
    match = re.search(r"\b([1-5])\s*(?:/\s*5)?\b", texto)
    palabras_feedback = ["doy", "daria", "daría", "calific", "merece", "le doy"]
    if match and any(w in texto.lower() for w in palabras_feedback):
        return int(match.group(1))
    return None


def convertir_a_texto_plano(contenido) -> str:
    """Convierte respuestas estructuradas (str/list/dict) en texto plano."""
    if isinstance(contenido, str):
        return contenido

    if isinstance(contenido, list):
        partes = []
        for item in contenido:
            if isinstance(item, str):
                partes.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    partes.append(str(item.get("text", "")))
                elif "text" in item:
                    partes.append(str(item.get("text", "")))
                else:
                    partes.append(str(item))
            else:
                partes.append(str(item))
        return "\n".join(p for p in partes if p).strip()

    if isinstance(contenido, dict):
        if "text" in contenido:
            return str(contenido.get("text", ""))
        return str(contenido)

    return str(contenido)


def extraer_receta_de_respuesta(respuesta: str):
    respuesta = convertir_a_texto_plano(respuesta)

    # Intenta encontrar nombre de receta en frases comunes.
    match = re.search(
        r"(?:receta de|recomiendo(?:\s+la\s+receta\s+de)?|preparar)\s+([A-ZÁÉÍÓÚÑ][^\n.?!,]{3,40})",
        respuesta,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Fallback: si la primera linea parece titulo de receta.
    primera_linea = respuesta.strip().split("\n")[0]
    if len(primera_linea) < 60 and not primera_linea.endswith((".", "?")):
        return primera_linea
    return None


@tool
def consultar_gustos() -> str:
    """Consulta historial del usuario para personalizar recomendaciones."""
    registrar_uso_tool("consultar_gustos")
    data = cargar_historial()

    if not data["historial"]:
        return "Sin historial. Es la primera recomendacion."

    resumen = f"Recetas probadas: {len(data['historial'])}\n"
    resumen += "Ultimas recetas:\n"
    for r in data["historial"][-5:]:
        resumen += f"- {r['receta']} (calificacion: {r['calificacion']}/5): {r.get('comentario', '')}\n"

    if data["favoritas"]:
        resumen += f"Le ha gustado: {', '.join(data['favoritas'])}\n"
    if data["evitar"]:
        resumen += f"No le ha gustado: {', '.join(data['evitar'])}\n"

    return resumen


@tool
def registrar_feedback(receta: str, calificacion: int, comentario: str) -> str:
    """Guarda feedback de una receta. Calificacion valida: 1 a 5."""
    registrar_uso_tool("registrar_feedback")
    if not 1 <= calificacion <= 5:
        return "La calificacion debe ser entre 1 y 5."

    _guardar_feedback_directo(receta, calificacion, comentario)
    return f"Feedback guardado: '{receta}' con {calificacion}/5."


@tool
def recetas_ya_probadas() -> str:
    """Retorna recetas ya probadas para evitar recomendaciones repetidas."""
    registrar_uso_tool("recetas_ya_probadas")
    data = cargar_historial()
    if not data["historial"]:
        return "Ninguna receta probada aun."

    nombres = [r["receta"] for r in data["historial"]]
    return f"Recetas ya probadas (no repetir): {', '.join(nombres)}"


@tool
def buscar_wikipedia(consulta: str) -> str:
    """Busca informacion en Wikipedia para responder dudas de cultura general."""
    registrar_uso_tool("buscar_wikipedia")
    wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
    )
    return wiki.run(consulta)

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.7)
# model = ChatOllama(model="qwen2.5:3b", temperature=0.7)
tools = [consultar_gustos, registrar_feedback, recetas_ya_probadas, buscar_wikipedia]

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="""
Eres ChefBot, un asistente experto en recetas de cocina.
Escribe tu respuesta de forma clara y directa, sin rodeos. Usa las herramientas disponibles para personalizar tus recomendaciones según los gustos del usuario y evitar sugerir recetas que ya ha probado.
No escribas con markdown ni uses formato especial, responde solo con texto plano.

Al recibir cualquier mensaje:
1. Usa consultar_gustos para ver historial y preferencias.
2. Usa recetas_ya_probadas para no repetir recetas.
3. Recomienda UNA receta con nombre, ingredientes y pasos.
4. Si hay feedback, usa registrar_feedback para guardarlo.
5. Si preguntan datos enciclopedicos sobre ingredientes o de cultura general, usa buscar_wikipedia.

Se breve y directo.
""",
)


def chat(pregunta: str):
    global ultima_receta

    # Guarda feedback aunque el modelo no invoque tools correctamente.
    calificacion = detectar_calificacion(pregunta)
    if calificacion is not None and ultima_receta:
        _guardar_feedback_directo(ultima_receta, calificacion, pregunta)
        print(f"[Feedback guardado: '{ultima_receta}' - {calificacion}/5]")

    result = agent.invoke({"messages": [HumanMessage(content=pregunta)]})

    # Reporte por turno: cuantas tools uso el agente y cuales fueron.
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    tools_turno = [m.name for m in tool_messages if getattr(m, "name", None)]
    if tools_turno:
        print(f"[TOOLS-TURNO] Accesos: {len(tools_turno)} | Cuales: {', '.join(tools_turno)}")
    else:
        print("[TOOLS-TURNO] Accesos: 0 | Cuales: ninguna")

    respuesta = convertir_a_texto_plano(result["messages"][-1].content)
    print(f"\nChefBot: {respuesta}\n")

    receta_extraida = extraer_receta_de_respuesta(respuesta)
    if receta_extraida:
        ultima_receta = receta_extraida

    return respuesta


if __name__ == "__main__":
    print("=== ChefBot - Recomendador de Recetas ===")
    print("Como usar:")
    print("- Pide receta: 'quiero una receta con pollo'")
    print("- Da feedback: 'le doy 4/5, estuvo rica'")
    print("- Salida limpia: escribe 'salir', 's', 'exit' o 'quit'\n")

    try:
        while True:
            pregunta = input("Tu: ").strip()

            # Opcion para salir sin interrumpir el proceso.
            if pregunta.lower() in ("salir", "s", "exit", "quit"):
                print("Hasta luego!")
                break

            if pregunta:
                chat(pregunta)
    except KeyboardInterrupt:
        # Ctrl+C ahora termina de forma controlada.
        print("\nSesion finalizada por teclado.")
