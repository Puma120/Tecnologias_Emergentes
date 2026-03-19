from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from langchain_community.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

@tool
def buscar_web(consulta: str) -> str:
    '''busca información en la web con DuckDuckGo'''
    search_tool = DuckDuckGoSearchResults()
    resultado = search_tool.invoke(consulta)
    return f"Resultados de búsqueda para '{consulta}': {resultado}" 


@tool
def calculadora(operacion: str) -> str:
    '''realiza operaciones matemáticas'''
    try:
        resultado = eval(operacion, {"__builtins__": {}})
        return f"El resultado de {operacion} es {resultado}"
    except Exception as e:
        return f"Error al calcular {operacion}: {e}"

wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=12000
    )
)

#tool tavily
tavily = TavilySearchResults(max_results=3)

agent = create_agent(
    model=ChatOllama(model="qwen2.5:3b"),
    tools=[buscar_web, calculadora, wikipedia, tavily],
    system_prompt="""
Eres un asistente de investigacion.
Usa:
- buscar_web para consultas generales.
- calculadora para operaciones matematicas.
- wikipedia para consultas enciclopédicas.
- tavily para busquedas en la web con fuentes.
Responde de manera clara y concisa, citando tus fuentes cuando uses herramientas.
"""
)

human_msg = HumanMessage(content="¿cual fue el primer presidente de mexico y dame el resultado de 2.4334 por 3.1416?")

result = agent.invoke({"messages": [human_msg]})

print(result["messages"][-1].content)

print()
for i, msg in enumerate(result["messages"]):
    print(msg.pretty_print())
