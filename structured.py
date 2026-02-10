from google import genai
from dotenv import load_dotenv
import ollama
import time
import json
import os
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

load_dotenv()

class Pelicula(BaseModel):
    titulo: str = Field(..., description="Título de la película")
    año: int = Field(..., ge=1900, le=2026, description="Año de estreno")
    director: str = Field(..., description="Nombre del director")
    genero: List[str] = Field(..., min_length=1, description="Lista de géneros")
    calificacion: float = Field(..., ge=0, le=10, description="Calificación de 0 a 10")
    duracion_minutos: int = Field(..., ge=1, description="Duración en minutos")
    sinopsis: str = Field(..., min_length=10, description="Breve sinopsis")
    actores_principales: List[str] = Field(..., min_length=1, description="Lista de actores principales")
    presupuesto_millones_usd: Optional[float] = Field(None, description="Presupuesto en millones USD")
    ganadora_oscar: bool = Field(..., description="Si ganó algún Oscar")

prompt = """Genera información sobre una película famosa en formato JSON.
El JSON debe tener EXACTAMENTE esta estructura (sin texto adicional, solo el JSON):
{
    "titulo": "string",
    "año": number (1900-2026),
    "director": "string",
    "genero": ["string", ...],
    "calificacion": number (0-10),
    "duracion_minutos": number,
    "sinopsis": "string (mínimo 10 caracteres)",
    "actores_principales": ["string", ...],
    "presupuesto_millones_usd": number o null,
    "ganadora_oscar": boolean
}

Responde SOLO con el JSON válido, sin explicaciones ni markdown."""

def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    return json.loads(text)

def validate_response(text: str, model_name: str) -> dict:
    result = {
        "valid": False,
        "data": None,
        "error": None
    }
    try:
        data = extract_json(text)
        pelicula = Pelicula(**data)
        result["valid"] = True
        result["data"] = pelicula.model_dump()
    except json.JSONDecodeError as e:
        result["error"] = f"JSON inválido: {str(e)}"
    except ValidationError as e:
        result["error"] = f"Validación fallida: {e.errors()}"
    except Exception as e:
        result["error"] = f"Error: {str(e)}"
    
    return result

print("GEMINI (gemini-3-flash-preview)")
client = genai.Client()
start_gemini = time.time()
response_gemini = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt,
)
end_gemini = time.time()

gemini_validation = validate_response(response_gemini.text, "Gemini")
print(response_gemini.text)
print(f"\nVálido: {gemini_validation['valid']}")
if gemini_validation['error']:
    print(f"Error: {gemini_validation['error']}")
print(f"Tiempo: {end_gemini - start_gemini:.2f}s")

print("OLLAMA (qwen2.5:3b)")
start_ollama = time.time()
response_ollama = ollama.chat(
    model="qwen2.5:3b",
    messages=[{"role": "user", "content": prompt}]
)
end_ollama = time.time()

ollama_text = response_ollama["message"]["content"]
ollama_validation = validate_response(ollama_text, "Ollama")
print(ollama_text)
print(f"\nVálido: {ollama_validation['valid']}")
if ollama_validation['error']:
    print(f"Error: {ollama_validation['error']}")
print(f"Tiempo: {end_ollama - start_ollama:.2f}s")

print("RESUMEN VALIDACION")
print(f"Gemini: {'PASS' if gemini_validation['valid'] else 'FAIL'}")
print(f"Ollama: {'PASS' if ollama_validation['valid'] else 'FAIL'}")


log_entry = {
    "timestamp": datetime.now().isoformat(),
    "test_type": "structured_output",
    "schema": "Pelicula",
    "prompt": prompt,
    "gemini": {
        "model": "gemini-3-flash-preview",
        "raw_response": response_gemini.text,
        "validation_passed": gemini_validation["valid"],
        "validation_error": gemini_validation["error"],
        "parsed_data": gemini_validation["data"],
        "time_seconds": round(end_gemini - start_gemini, 2)
    },
    "ollama": {
        "model": "qwen2.5:3b",
        "raw_response": ollama_text,
        "validation_passed": ollama_validation["valid"],
        "validation_error": ollama_validation["error"],
        "parsed_data": ollama_validation["data"],
        "time_seconds": round(end_ollama - start_ollama, 2)
    }
}

log_file = "logs/pydantic.json"
os.makedirs("logs", exist_ok=True)

if os.path.exists(log_file):
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except (json.JSONDecodeError, ValueError):
        logs = []
else:
    logs = []

logs.append(log_entry)

with open(log_file, "w", encoding="utf-8") as f:
    json.dump(logs, f, indent=2, ensure_ascii=False)

print(f"\nLog guardado en {log_file}")
