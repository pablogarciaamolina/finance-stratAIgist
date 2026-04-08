"""Financial tools — SEC EDGAR and Alpha Vantage integrations."""

import os
import json
import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()


@tool
def company_fundamentals(ticker: str) -> str:
    """
    Obtiene datos financieros básicos de una empresa pública (ingresos, beneficios, activos)
    usando datos oficiales de SEC EDGAR.
    Input: ticker bursátil (ej. AAPL)
    """
    try:
        ticker = ticker.upper()
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
            "total_liabilities": latest_value("Liabilities"),
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

        cik_url = "https://www.sec.gov/files/company_tickers.json"
        cik_data = requests.get(cik_url, headers=headers).json()

        cik = None
        for item in cik_data.values():
            if item["ticker"] == ticker:
                cik = str(item["cik_str"]).zfill(10)
                break

        if not cik:
            return f"No se pudo encontrar el CIK para {ticker}"

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
                    "description": desc or "Evento material reportado",
                })

        if not events:
            return f"No se encontraron eventos recientes (8-K) para {ticker}"

        return json.dumps(
            {"company": submissions.get("name", ticker), "events": events[:5]},
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:
        return f"Error obteniendo eventos de {ticker}: {e}"


@tool
def stock_price(ticker: str) -> str:
    """Obtiene el precio actual de una acción usando Alpha Vantage (free tier)."""
    try:
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
            "change_percent": quote.get("10. change percent"),
        }
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error obteniendo precio de {ticker}: {e}"
