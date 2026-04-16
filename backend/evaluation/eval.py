import requests
import json
from datasets import load_dataset


# ─────────────────────────────────────────────
# Cargar dataset Finance-Reasoning
# ─────────────────────────────────────────────
dataset = load_dataset("PatronusAI/financebench", split="train")

MAX_QUESTIONS = 100

preguntas_test = []
for i, sample in enumerate(dataset):
    if i >= MAX_QUESTIONS:
        break

    preguntas_test.append({
        "pregunta": sample["question"],
        "respuesta": sample.get("answer", "N/A")
    })

# ─────────────────────────────────────────────
# Evaluación
# ─────────────────────────────────────────────

    with open(f"resultados_financebench.txt", "a", encoding="utf-8") as f:

        endpoint = "https://rayford-superinclusive-eden.ngrok-free.dev" + "/api/chat"

        f.write("\n\n===== FINANCE REASONING BENCHMARK =====\n")

        for idx, pregunta in enumerate(preguntas_test):
            f.write(f"\nPREGUNTA {idx+1}:\n{pregunta['pregunta']}\n")
            f.write(f"RESPUESTA ESPERADA:\n{pregunta['respuesta']}\n")

            payload = {
                "prompt": pregunta["pregunta"],
                "session_id": "test-session-1",
                "user_profile": {
                    "risk_level": "moderate",
                    "investment_horizon": "long",
                    "capital_amount": 10000,
                    "preferences": ["technology", "diversification"]
                }
            }

            try:
                response = requests.post(endpoint, json=payload)
                f.write(f"Status code: {response.status_code}\n")

                full_json = response.json()

                f.write("RESPUESTA OBTENIDA:\n")
                f.write(json.dumps(full_json, indent=4, ensure_ascii=False))
                f.write("\n\n")

            except Exception as e:
                f.write(f"FALLO: {e}\n\n")