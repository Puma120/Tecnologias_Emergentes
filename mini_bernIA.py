from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()

@tool 
def calculadora(operacion: str) -> str:
    """
    Realiza operaciones matemáticas básicas.
    Args: operacion: Expresion como string (ejemplo: "2 + 2", "5 * 3", "10 / 2")
    Returns: Resultado de la operación como string.
    """
    try:
        #TODO: Nunca usar eval en producción
        resultado = eval(operacion)
        return f"El resultado de {operacion} es {resultado} ヾ(≧▽≦*)o"
    except Exception as e:
        return f"Error al calcular: {str(e)}"
    
@tool
def obtener_temperatura(ciudad: str) -> str:
    """
    Obtiene la temperatura actual de una ciudad.
    Args: ciudad: Nombre de la ciudad (ejemplo: "Madrid", "New York")
    Returns: Temperatura actual de la ciudad como string.
    """
    temperaturas = {
        "puebla": "22°C",
        "ciudad de méxico": "25°C",
        "guadalajara": "24°C",
        "monterrey": "28°C",
        "cancún": "30°C"
    }

    ciudad_lower = ciudad.lower()
    if ciudad_lower in temperaturas:
        return f"La temperatura actual en {ciudad} es {temperaturas[ciudad_lower]} (❁´◡`❁)"
    else:        return f"No tengo información de temperatura para {ciudad} (￣﹏￣；)"
    
@tool
def buscar_definicion(termino: str) -> str:
    """
    Busca la definicion de un termino de programacion.
    Args: termino: Termino a buscar (ejemplo: "funcion", "variable", "clase")
    Returns: Definición del termino como string.
    """
    definiciones = {
        "funcion": "Una función es un bloque de código que realiza una tarea específica y puede ser reutilizado en diferentes partes de un programa.",
        "variable": "Una variable es un espacio en la memoria que se utiliza para almacenar datos que pueden cambiar durante la ejecución de un programa.",
        "clase": "Una clase es una plantilla para crear objetos en programación orientada a objetos. Define atributos y métodos que los objetos creados a partir de la clase pueden tener.",
        "Bernia": "Bernia es un modelo de lenguaje desarrollado por Ollama, diseñado para generar texto de alta calidad y realizar tareas de procesamiento de lenguaje natural con una amplia gama de aplicaciones.",
        "Antia": "Antia es un modelo de lenguaje especializada en tareas de análisis de sentimientos y comprensión de emociones en texto, desarrollado por Ollama para mejorar la interacción humano-máquina."
    }

    termino_lower = termino.lower()
    if termino_lower in definiciones:
        return f"La definición de {termino} es: {definiciones[termino_lower]} (✿◠‿◠)"
    else:
        return f"No tengo información sobre la definición de {termino} (￣﹏￣；)"


model = ChatOllama(model="qwen2.5:3b", temperature=0.1)
tools = [calculadora, obtener_temperatura, buscar_definicion]
model_with_tools = model.bind_tools(tools)

def ejecutar_tool(tool_call):
    tool_map = {
        "calculadora": calculadora,
        "obtener_temperatura": obtener_temperatura,
        "buscar_definicion": buscar_definicion
    }

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    if tool_name in tool_map:
        resultado = tool_map[tool_name].invoke(tool_args)
        return resultado
    return f"Tool {tool_name} no encontrada. (⊙_⊙)？"

def bernIA(pregunta: str):
    print(f"\nPREGUNTA: {pregunta}")

    messages = [HumanMessage(content=pregunta)]

    response = model_with_tools.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        print(f"- El modelo ha decidido usar: {len(response.tool_calls)} tools\n")
        for tool_call in response.tool_calls:
            print(f" {tool_call['name']}({tool_call['args']})")

            resultado = ejecutar_tool(tool_call)
            print(f"  Resultado: {resultado}\n")

            tool_message = ToolMessage(
                content=resultado,
                tool_call_id=tool_call["id"]
            )
            messages.append(tool_message)

        print("Generando respuesta... (⌐■_■)")
        final_response = model_with_tools.invoke(messages)
        print(f"RESPUESTA: {final_response.content}\n")

        return final_response.content
    else:
        print(f"RESPUESTA (●'◡'●): {response.content} \n")
        return response.content
    
if __name__ == "__main__":
    # bernIA("¿Cuánto es 4 * 4?")
    # bernIA("Cuanto es 4 multiplicado por 4?")
    # bernIA("Cuanto es 4 veces 4?")
    bernIA("¿Cuál es la temperatura actual en Puebla?")
    bernIA("¿Qué es BernIA?")
    bernIA("¿Qué es AntIA?")
    bernIA("Cual es la temperatura en Cancún divida entre 2?")