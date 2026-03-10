from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
'''
SystemMessage: Instrucciones o contexto para el modelo (role: system).
HumanMessage: Input del usuario (role: usuario).
AIMessage: Respuesta del modelo (role: assistant).
ToolMessage: Resultado de ejecutar una herramienta (role: tool).
'''

#chat models
gemini = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
ollama = ChatOllama(model="qwen2.5:3b")

message = [
    SystemMessage(content="Eres un experto en programacion en python. Muy serio, por lo tanto responderas de manera breve y concisa, sin rodeos ni explicaciones innecesarias."),
    HumanMessage(content="Explicame el concepto de arboles binarios de busqueda en python con un ejemplo sencillo.")
]

# response_gemini = gemini.invoke(message)
# print("Respuesta Gemini:")
# print(response_gemini)

# print("\n---------------------------------------------------------------------------------------\n")

# response_ollama = ollama.invoke(message)
# print("\nRespuesta Ollama:")
# print(response_ollama)
# print(response_ollama.additional_kwargs)
# print(response_ollama.response_metadata)

# msj = HumanMessage(content="Hola mundo")
# print(msj.content) #Contenido
# print(msj.type) #Rol del mensaje
# print(msj.additional_kwargs) #Cualquier información adicional que se haya incluido en el mensaje
# print(msj.response_metadata) # Solo en AImessage (tokens usados)
# print(msj.id) #Identificador único del mensaje

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en {tema}. Responde en {idioma}."),
    ("human", "{pregunta}")
])

formatted = prompt.invoke({
    "tema": "inteligencia artificial",
    "idioma": "español",
    "pregunta": "¿Qué es un transformer?"
})

print("Prompt formateado:")
print(formatted)

parser = StrOutputParser()

# Cadena con Gemini
chain_gemini = prompt | gemini
response_gemini = chain_gemini.invoke({
    "tema": "inteligencia artificial",
    "idioma": "español",
    "pregunta": "¿Qué es un transformer?"
})

print("Respuesta Gemini:")
print(response_gemini.content)
print("\nMetadata Gemini:")
print("usage_metadata:", response_gemini.usage_metadata)
print("response_metadata:", response_gemini.response_metadata)

print("\n------------------------------------------------------------\n")

# Cadena con Ollama
chain_ollama = prompt | ollama
response_ollama = chain_ollama.invoke({
    "tema": "inteligencia artificial",
    "idioma": "español",
    "pregunta": "¿Qué es un transformer?"
})

print("Respuesta Ollama:")
print(response_ollama.content)
print("\nMetadata Ollama:")
print("usage_metadata:", response_ollama.usage_metadata)
print("response_metadata:", response_ollama.response_metadata)