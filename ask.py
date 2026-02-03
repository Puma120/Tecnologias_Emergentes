from google import genai
from dotenv import load_dotenv
import ollama
import time
import json
import os
from datetime import datetime

load_dotenv()

# Precios Gemini por millón de tokens (USD)
GEMINI_INPUT_PRICE = 0.50 / 1_000_000
GEMINI_OUTPUT_PRICE = 3.00 / 1_000_000

prompt = "Cuentame un chiste sobre programadores."

client = genai.Client()
start_gemini = time.time()
response_gemini = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt,
)
end_gemini = time.time()
print(response_gemini.text)
print(f"Tokens: {response_gemini.usage_metadata.total_token_count}")
print(f"\nTiempo: {end_gemini - start_gemini:.2f} segundos")

print("OLLAMA (qwen2.5:3b)")
start_ollama = time.time()
response_ollama = ollama.chat(
    model="qwen2.5:3b",
    messages=[{"role": "user", "content": prompt}]
)
end_ollama = time.time()
print(response_ollama["message"]["content"])
print(f"Tokens: {response_ollama['prompt_eval_count'] + response_ollama['eval_count']}")
print(f"\nTiempo: {end_ollama - start_ollama:.2f} segundos")

print("RESUMEN")
print(f"Gemini:  {end_gemini - start_gemini:.2f}s")
print(f"Ollama:  {end_ollama - start_ollama:.2f}s")

# Calcular costos Gemini
gemini_input_tokens = response_gemini.usage_metadata.prompt_token_count
gemini_output_tokens = response_gemini.usage_metadata.candidates_token_count
gemini_cost = (gemini_input_tokens * GEMINI_INPUT_PRICE) + (gemini_output_tokens * GEMINI_OUTPUT_PRICE)

# Tokens Ollama
ollama_input_tokens = response_ollama['prompt_eval_count']
ollama_output_tokens = response_ollama['eval_count']

# Log entry
log_entry = {
    "timestamp": datetime.now().isoformat(),
    "prompt": prompt,
    "gemini": {
        "model": "gemini-3-flash-preview",
        "response": response_gemini.text,
        "input_tokens": gemini_input_tokens,
        "output_tokens": gemini_output_tokens,
        "total_tokens": response_gemini.usage_metadata.total_token_count,
        "time_seconds": round(end_gemini - start_gemini, 2),
        "cost_usd": round(gemini_cost, 6)
    },
    "ollama": {
        "model": "qwen2.5:3b",
        "response": response_ollama["message"]["content"],
        "input_tokens": ollama_input_tokens,
        "output_tokens": ollama_output_tokens,
        "total_tokens": ollama_input_tokens + ollama_output_tokens,
        "time_seconds": round(end_ollama - start_ollama, 2),
        "cost_usd": 0.0
    }
}

# Append al archivo JSON
log_file = "logs/terminal.json"
os.makedirs("logs", exist_ok=True)

if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8") as f:
        logs = json.load(f)
else:
    logs = []

logs.append(log_entry)

with open(log_file, "w", encoding="utf-8") as f:
    json.dump(logs, f, indent=2, ensure_ascii=False)

print(f"\nLog guardado en {log_file}")

