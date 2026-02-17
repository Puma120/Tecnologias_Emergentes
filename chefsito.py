from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

#Definimos los esquemas
class Receta(BaseModel):
    nombre: str = Field(description="Nombre de la receta")
    ingredientes: list[str] = Field(description="Lista de ingredientes")
    tiempo_minutos: int = Field(description="Tiempo de preparación en minutos")

# Creamos el parser
json_parser = JsonOutputParser(pydantic_object=Receta)

# Creamos el prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Eres un chef profesional." 
        "Responde unicamente con un JSON."
        "{format_instructions}"
    ),
    (
        "human",
        "Dame una receta sencilla para preparar una hamburguesa gourmet."
    )
])

# Obtenemos las instrucciones de formato del parser
format_instructions = json_parser.get_format_instructions()

# Creamos el modelo
model = ChatOllama(model="qwen2.5:3b", temperature=0)

# Creamos el pipeline
chain = prompt.partial(
    format_instructions=format_instructions
) | model | json_parser

# Ejecutamos el pipeline
response = chain.invoke({})

print("Receta obtenida:")
print(response)
print(type(response))
print()
print(response.response_metadata)
