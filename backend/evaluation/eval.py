import requests
import json
from datasets import load_dataset


# ─────────────────────────────────────────────
# Cargar dataset FinanceBench
# ─────────────────────────────────────────────
dataset = load_dataset("PatronusAI/financebench", split="train")

MAX_QUESTIONS = 100

preguntas_test = []
for i, sample in enumerate(dataset):
    if i >= MAX_QUESTIONS:
        break

    preguntas_test.append({
        "pregunta": sample["question"],
        "respuesta": sample.get("answer", "N/A"),
        "company_name": sample.get("company", None) or sample.get("company_name", None),
        "ticker": sample.get("ticker", None),
    })

# ─────────────────────────────────────────────
# Evaluación
# ─────────────────────────────────────────────

endpoint = "https://rayford-superinclusive-eden.ngrok-free.dev/api/chat"

with open("resultados_financebench.txt", "w", encoding="utf-8") as f:
    f.write("===== FINANCE REASONING BENCHMARK =====\n")

    for idx, pregunta in enumerate(preguntas_test):
        f.write(f"\nPREGUNTA {idx+1}:\n{pregunta['pregunta']}\n")
        f.write(f"RESPUESTA ESPERADA:\n{pregunta['respuesta']}\n")

        payload = {
            "prompt": pregunta["pregunta"],
            "session_id": f"test-session-{idx+1}",
            "mode": "benchmark",
            "company_name": pregunta["company_name"],
            "ticker": pregunta["ticker"],
        }

        try:
            response = requests.post(endpoint, json=payload, timeout=120)

            f.write(f"Status code: {response.status_code}\n")
            f.write("RAW RESPONSE TEXT:\n")
            f.write(response.text[:5000] + "\n\n")

            if "application/json" in response.headers.get("Content-Type", ""):
                full_json = response.json()

                f.write("RESPUESTA OBTENIDA:\n")
                f.write(json.dumps(full_json, indent=4, ensure_ascii=False))
                f.write("\n\n")
            else:
                f.write("FALLO: la respuesta no es JSON válido.\n\n")

        except Exception as e:
            f.write(f"FALLO: {e}\n\n")