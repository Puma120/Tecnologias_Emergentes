import json
import matplotlib.pyplot as plt
import numpy as np

# Cargar logs
with open("logs/terminal.json", "r", encoding="utf-8") as f:
    logs = json.load(f)

# Extraer datos
indices = list(range(1, len(logs) + 1))
gemini_times = [log["gemini"]["time_seconds"] for log in logs]
ollama_times = [log["ollama"]["time_seconds"] for log in logs]
gemini_tokens = [log["gemini"]["total_tokens"] for log in logs]
ollama_tokens = [log["ollama"]["total_tokens"] for log in logs]
gemini_costs = [log["gemini"]["cost_usd"] for log in logs]
ollama_costs = [log["ollama"]["cost_usd"] for log in logs]
gemini_input = sum(log["gemini"]["input_tokens"] for log in logs)
gemini_output = sum(log["gemini"]["output_tokens"] for log in logs)
ollama_input = sum(log["ollama"]["input_tokens"] for log in logs)
ollama_output = sum(log["ollama"]["output_tokens"] for log in logs)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Comparación Gemini vs Ollama", fontsize=14, fontweight="bold")

# 1. Tiempo de respuesta
ax1 = axes[0, 0]
x = np.arange(len(indices))
width = 0.35
ax1.bar(x - width/2, gemini_times, width, label="Gemini", color="#4285F4")
ax1.bar(x + width/2, ollama_times, width, label="Ollama", color="#FF6B6B")
ax1.set_xlabel("Consulta #")
ax1.set_ylabel("Segundos")
ax1.set_title("Tiempo de Respuesta")
ax1.set_xticks(x)
ax1.set_xticklabels(indices)
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

# 2. Tokens totales
ax2 = axes[0, 1]
ax2.bar(x - width/2, gemini_tokens, width, label="Gemini", color="#4285F4")
ax2.bar(x + width/2, ollama_tokens, width, label="Ollama", color="#FF6B6B")
ax2.set_xlabel("Consulta #")
ax2.set_ylabel("Tokens")
ax2.set_title("Tokens Totales por Consulta")
ax2.set_xticks(x)
ax2.set_xticklabels(indices)
ax2.legend()
ax2.grid(axis="y", alpha=0.3)

# 3. Costo USD por prompt
ax3 = axes[1, 0]
ax3.bar(x - width/2, gemini_costs, width, label="Gemini", color="#4285F4")
ax3.bar(x + width/2, ollama_costs, width, label="Ollama (gratis)", color="#FF6B6B")
ax3.set_xlabel("Consulta #")
ax3.set_ylabel("USD")
ax3.set_title("Costo por Consulta (USD)")
ax3.set_xticks(x)
ax3.set_xticklabels(indices)
ax3.legend()
ax3.grid(axis="y", alpha=0.3)
# Mostrar costo acumulado
total_gemini_cost = sum(gemini_costs)
ax3.annotate(f"Total Gemini: ${total_gemini_cost:.4f}", xy=(0.95, 0.95), 
             xycoords="axes fraction", ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

# 4. Distribución tokens input vs output
ax4 = axes[1, 1]
categories = ["Gemini\nInput", "Gemini\nOutput", "Ollama\nInput", "Ollama\nOutput"]
values = [gemini_input, gemini_output, ollama_input, ollama_output]
colors = ["#4285F4", "#34A853", "#FF6B6B", "#FFB347"]
bars = ax4.bar(categories, values, color=colors)
ax4.set_ylabel("Tokens")
ax4.set_title("Distribución de Tokens (Acumulado)")
ax4.grid(axis="y", alpha=0.3)
# Agregar valores encima de las barras
for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             str(val), ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("logs/comparacion.png", dpi=150)
plt.show()

print("Gráfica guardada en logs/comparacion.png")
