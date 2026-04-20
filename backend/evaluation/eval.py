"""Local evaluator for advisor-style prompts against the FastAPI backend.

Usage:
    1. Start the backend in another terminal:
       python -m backend.api.app
    2. Run this evaluator:
       python -m backend.evaluation.eval
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8045"
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_OUTPUT_PATH = Path("backend/evaluation/resultados_eval_8045.json")


DEFAULT_USER_PROFILE = {
    "risk_level": "moderate",
    "investment_horizon": "long",
    "capital_amount": 10000,
    "preferences": ["technology", "diversification"],
}


def make_question(prompt: str, company_name: str, ticker: str) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "company_name": company_name,
        "ticker": ticker,
        "session_id": "test",
        "user_profile": DEFAULT_USER_PROFILE,
    }


# Keep advisor questions anchored to a single primary company per request,
# because the current market pipeline retrieves evidence for one main ticker.
PORTFOLIO_QUESTIONS: list[dict[str, Any]] = [
    make_question(
        "Como deberia distribuir 10.000 euros si quiero incluir Apple en una cartera moderada y a largo plazo?",
        "Apple",
        "AAPL",
    ),
    make_question(
        "Que peso maximo tendria sentido dar a NVIDIA para aprovechar crecimiento sin disparar demasiado la volatilidad?",
        "NVIDIA",
        "NVDA",
    ),
    make_question(
        "Como combinarias Alphabet (Google) con renta fija y ETFs globales para mantener una cartera equilibrada?",
        "Alphabet",
        "GOOGL",
    ),
    make_question(
        "Que papel deberia tener Microsoft como posicion principal dentro de una cartera moderada a largo plazo?",
        "Microsoft",
        "MSFT",
    ),
    make_question(
        "Que porcentaje de la cartera destinarias a Amazon frente a fondos indexados si busco crecimiento a largo plazo?",
        "Amazon",
        "AMZN",
    ),
    make_question(
        "Si ya tengo Apple con un peso relevante, como diversificarias el resto de la cartera para no concentrar demasiado riesgo en tecnologia?",
        "Apple",
        "AAPL",
    ),
    make_question(
        "Incluirias Meta como posicion secundaria en una cartera moderada o la dejarias fuera por su perfil de riesgo?",
        "Meta",
        "META",
    ),
    make_question(
        "Tiene sentido incluir Tesla solo como posicion pequena dentro de una cartera moderada o mejor evitarla?",
        "Tesla",
        "TSLA",
    ),
    make_question(
        "Tendria sentido usar NVIDIA como posicion tactica y completar el resto de la exposicion tecnologica con ETFs amplios?",
        "NVIDIA",
        "NVDA",
    ),
    make_question(
        "Como protegerias la cartera si una posicion importante en Microsoft sufre una correccion fuerte?",
        "Microsoft",
        "MSFT",
    ),
    make_question(
        "Cada cuanto rebalancearias una cartera en la que Alphabet tiene un peso relevante?",
        "Alphabet",
        "GOOGL",
    ),
    make_question(
        "Que proporcion mantendrias en liquidez si quiero entrar de forma gradual en NVIDIA sin asumir demasiado riesgo de timing?",
        "NVIDIA",
        "NVDA",
    ),
    make_question(
        "Incluirias oro o algun activo refugio para compensar una exposicion relevante a Amazon dentro de la cartera?",
        "Amazon",
        "AMZN",
    ),
    make_question(
        "Como dividirias la inversion entre Estados Unidos, Europa y Asia si quiero mantener Apple pero reducir sesgo geografico?",
        "Apple",
        "AAPL",
    ),
]


STOCK_PICKING_QUESTIONS: list[dict[str, Any]] = [
    make_question(
        "Apple encaja como posicion nucleo para un inversor moderado a largo plazo o la tratarias con mas cautela? Explica fortalezas y riesgos.",
        "Apple",
        "AAPL",
    ),
    make_question(
        "Que tres riesgos clave vigilarias antes de abrir una posicion en NVIDIA para este perfil?",
        "NVIDIA",
        "NVDA",
    ),
    make_question(
        "Que senales fundamentales y de noticias recientes deberia vigilar en Alphabet antes de decidir si comprarla?",
        "Alphabet",
        "GOOGL",
    ),
    make_question(
        "Microsoft te parece una accion adecuada para construir una posicion core a largo plazo con perfil moderado? Por que?",
        "Microsoft",
        "MSFT",
    ),
    make_question(
        "Si quisiera entrar en Amazon, lo harias de golpe o de forma escalonada dadas sus caracteristicas y el perfil del inversor?",
        "Amazon",
        "AMZN",
    ),
    make_question(
        "Meta tiene suficiente calidad para justificar una posicion pequena en cartera o los riesgos pesan demasiado?",
        "Meta",
        "META",
    ),
    make_question(
        "Tesla puede tener cabida como apuesta tactica pequena o el nivel de riesgo no compensa para este perfil?",
        "Tesla",
        "TSLA",
    ),
    make_question(
        "Que tendria que pasar para que dejases de ver atractiva una tesis de inversion en Apple?",
        "Apple",
        "AAPL",
    ),
    make_question(
        "Que catalizadores y que senales de alerta mirarias en Microsoft antes de aumentar posicion?",
        "Microsoft",
        "MSFT",
    ),
    make_question(
        "Que noticias o cambios fundamentales te harian evitar Amazon por ahora en lugar de incorporarla ya a la cartera?",
        "Amazon",
        "AMZN",
    ),
]


QUESTIONS: list[dict[str, Any]] = PORTFOLIO_QUESTIONS + STOCK_PICKING_QUESTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate advisor prompts against the local API.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL for the backend API. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Path to save the JSON results. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-request timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}",
    )
    return parser.parse_args()


def derive_investment_goals(user_profile: dict[str, Any]) -> list[str]:
    goal_map = {
        "technology": "growth",
        "diversification": "preservation",
        "income": "income",
        "growth": "growth",
        "preservation": "preservation",
        "speculation": "speculation",
    }

    goals: list[str] = []
    for preference in user_profile.get("preferences", []):
        mapped_goal = goal_map.get(str(preference).lower())
        if mapped_goal and mapped_goal not in goals:
            goals.append(mapped_goal)
    return goals


def build_payload(example: dict[str, Any], index: int) -> dict[str, Any]:
    payload = json.loads(json.dumps(example))
    payload["mode"] = "advisor"

    base_session_id = payload.get("session_id", "eval")
    payload["session_id"] = f"{base_session_id}-q{index:02d}"

    user_profile = payload.get("user_profile") or {}
    if user_profile and "investment_goals" not in user_profile:
        user_profile["investment_goals"] = derive_investment_goals(user_profile)
    payload["user_profile"] = user_profile

    return payload


def wait_for_backend(base_url: str, timeout_seconds: int, max_attempts: int = 5) -> None:
    health_url = f"{base_url.rstrip('/')}/api/health"
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(health_url, timeout=10)
            response.raise_for_status()
            print(f"[health] Backend listo en {health_url}")
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[health] Intento {attempt}/{max_attempts} fallido: {exc}")
            if attempt < max_attempts:
                time.sleep(min(timeout_seconds, 3))

    raise RuntimeError(
        "No se pudo conectar con el backend. Ejecuta `python -m backend.api.app` "
        f"y comprueba que responde en {health_url}."
    ) from last_error


def call_api(base_url: str, payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/chat"
    started_at = time.perf_counter()
    response = requests.post(url, json=payload, timeout=timeout_seconds)
    latency = time.perf_counter() - started_at

    record: dict[str, Any] = {
        "status_code": response.status_code,
        "latency_seconds": round(latency, 3),
        "content_type": response.headers.get("Content-Type", ""),
    }

    try:
        record["response_json"] = response.json()
    except ValueError:
        record["response_text"] = response.text

    return record


def run_evaluation(base_url: str, output_path: Path, timeout_seconds: int) -> dict[str, Any]:
    wait_for_backend(base_url, timeout_seconds)

    results: list[dict[str, Any]] = []
    success_count = 0

    total_questions = len(QUESTIONS)
    for index, example in enumerate(QUESTIONS, start=1):
        payload = build_payload(example, index)
        prompt = payload["prompt"]
        print(f"[{index:02d}/{total_questions:02d}] Enviando pregunta: {prompt}")

        result_entry: dict[str, Any] = {
            "index": index,
            "request": payload,
        }

        try:
            api_result = call_api(base_url, payload, timeout_seconds)
            result_entry.update(api_result)
            if api_result["status_code"] == 200:
                success_count += 1
                print(
                    f"[{index:02d}/{total_questions:02d}] OK "
                    f"({api_result['latency_seconds']} s)"
                )
            else:
                print(
                    f"[{index:02d}/{total_questions:02d}] ERROR HTTP "
                    f"{api_result['status_code']}"
                )
        except Exception as exc:  # noqa: BLE001
            result_entry["error"] = str(exc)
            print(f"[{index:02d}/{total_questions:02d}] FALLO: {exc}")

        results.append(result_entry)

    summary = {
        "base_url": base_url,
        "total_questions": total_questions,
        "successful_requests": success_count,
        "failed_requests": total_questions - success_count,
    }

    output = {
        "summary": summary,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output = run_evaluation(
        base_url=args.base_url,
        output_path=output_path,
        timeout_seconds=args.timeout,
    )

    summary = output["summary"]
    print()
    print("=== Resumen ===")
    print(f"Base URL: {summary['base_url']}")
    print(f"Preguntas totales: {summary['total_questions']}")
    print(f"Solicitudes correctas: {summary['successful_requests']}")
    print(f"Solicitudes fallidas: {summary['failed_requests']}")
    print(f"Resultados guardados en: {output_path}")


if __name__ == "__main__":
    main()
