# --- Implementación de las herramientas ---

import os
import requests
import json
import re

import numexpr as ne
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()

"""
otras opciones para agentes:
- from llama_index
    from llama_index.llms import Ollama
    from llama_index.agent import ReActAgent
    from llama_index.tools import FunctionTool
- https://www.tavily.com/
- https://www.langchain.com/langgraph

"""

@tool
def calculator(expression: str) -> str:
    """Evalúa una expresión matemática simple. Útil para realizar cálculos aritméticos."""

    # dejar solo caracteres matemáticos
    expression = re.sub(r"[^0-9+\-*/().%\s]", "", expression)

    try:
        result = ne.evaluate(expression)
        return str(result)
    except Exception as e:
        return f"Error calculando: {e}"
        # Mejor hacer raise ...

@tool
def internet_search(query: str) -> str:
    """Busca información en internet usando Tavily API. SIEMPRE usa esta herramienta para buscar información sobre personas, lugares, tecnología o cualquier dato factual. Input: la consulta de búsqueda."""
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            return "Error: TAVILY_API_KEY no está configurada en las variables de entorno."
        
        client = TavilyClient(api_key=tavily_api_key)
        response = client.search(query=query, max_results=1)
        
        if response and "results" in response and len(response["results"]) > 0:
            first_result = response["results"][0]
            title = first_result.get("title", "")
            content = first_result.get("content", "")
            url = first_result.get("url", "")
            return f"{title}\n{content}\nFuente: {url}"
        else:
            return "No se encontraron resultados relevantes en la búsqueda."
    except Exception as e:
        return f"Error en la búsqueda: {e}"

@tool
def company_fundamentals(ticker: str) -> str:
    """
    Obtiene datos financieros básicos de una empresa pública (ingresos, beneficios, activos)
    usando datos oficiales de SEC EDGAR.

    Input: ticker bursátil (ej. AAPL)
    """
    try:
        ticker = ticker.upper()

        # Map ticker → CIK
        cik_url = "https://www.sec.gov/files/company_tickers.json"
        headers = {"User-Agent": "LLM-Agent/1.0 (contact@example.com)"}
        cik_data = requests.get(cik_url, headers=headers).json()

        cik = None
        for item in cik_data.values():
            if item["ticker"] == ticker:
                cik = str(item["cik_str"]).zfill(10)
                break

        if not cik:
            return f"No se pudo encontrar el CIK para {ticker}"

        facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        facts = requests.get(facts_url, headers=headers).json()

        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        def latest_value(field):
            data = us_gaap.get(field, {}).get("units", {}).get("USD", [])
            return data[-1]["val"] if data else "No disponible"

        result = {
            "empresa": facts.get("entityName", ticker),
            "revenue": latest_value("Revenues"),
            "net_income": latest_value("NetIncomeLoss"),
            "total_assets": latest_value("Assets"),
            "total_liabilities": latest_value("Liabilities")
        }

        return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        return f"Error obteniendo fundamentales de {ticker}: {e}"


@tool
def company_events(ticker: str) -> str:
    """
    Obtiene eventos recientes y noticias materiales de una empresa
    usando filings 8-K de SEC EDGAR.
    Input: ticker bursátil (ej. AAPL)
    """
    try:
        ticker = ticker.upper()
        headers = {"User-Agent": "LLM-Agent/1.0 (contact@example.com)"}

        # 1. Ticker → CIK
        cik_url = "https://www.sec.gov/files/company_tickers.json"
        cik_data = requests.get(cik_url, headers=headers).json()

        cik = None
        for item in cik_data.values():
            if item["ticker"] == ticker:
                cik = str(item["cik_str"]).zfill(10)
                break

        if not cik:
            return f"No se pudo encontrar el CIK para {ticker}"

        # 2. Obtener filings recientes
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        submissions = requests.get(submissions_url, headers=headers).json()

        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        descriptions = recent.get("primaryDocDescription", [])

        events = []
        for form, date, desc in zip(forms, dates, descriptions):
            if form == "8-K":
                events.append({
                    "date": date,
                    "type": "8-K",
                    "description": desc or "Evento material reportado"
                })

        if not events:
            return f"No se encontraron eventos recientes (8-K) para {ticker}"

        return json.dumps({
            "company": submissions.get("name", ticker),
            "events": events[:5]  # últimos 5 eventos
        }, indent=2, ensure_ascii=False)

    except Exception as e:
        return f"Error obteniendo eventos de {ticker}: {e}"

@tool
def stock_price(ticker: str) -> str:
    """
    Obtiene el precio actual de una acción usando Alpha Vantage (free tier).
    """
    try:
        import os, requests, json

        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key:
            return "Error: ALPHAVANTAGE_API_KEY no configurada."

        ticker = ticker.upper()
        url = (
            "https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}"
        )

        r = requests.get(url, timeout=10)
        data = r.json()

        quote = data.get("Global Quote", {})
        if not quote:
            return f"No se encontró información de mercado para {ticker}"

        result = {
            "ticker": ticker,
            "price": quote.get("05. price"),
            "change": quote.get("09. change"),
            "change_percent": quote.get("10. change percent")
        }

        return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        return f"Error obteniendo precio de {ticker}: {e}"


