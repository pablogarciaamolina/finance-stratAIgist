"""
Financial tools for the multi-agent system.

Provides LangChain tools for:
- current stock price
- latest company fundamentals
- recent 8-K events
- historical financial statement data by fiscal year

Sources:
- Alpha Vantage
- SEC EDGAR company tickers
- SEC companyfacts
- SEC submissions
"""

import json
import os
from typing import Any, Dict, List, Optional

import requests
from langchain.tools import tool

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

SEC_HEADERS = {
    "User-Agent": "FinanceStratAIgist/0.1 contact@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}


_TICKER_TO_CIK_CACHE: Optional[Dict[str, str]] = None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _safe_get_json(url: str, headers: Optional[dict] = None, timeout: int = 15) -> dict:
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _load_ticker_to_cik_map() -> Dict[str, str]:
    global _TICKER_TO_CIK_CACHE

    if _TICKER_TO_CIK_CACHE is not None:
        return _TICKER_TO_CIK_CACHE

    data = _safe_get_json(SEC_TICKERS_URL, headers=SEC_HEADERS, timeout=20)
    mapping = {}

    for _, item in data.items():
        ticker = item.get("ticker")
        cik = item.get("cik_str")
        if ticker and cik is not None:
            mapping[str(ticker).upper()] = str(cik).zfill(10)

    _TICKER_TO_CIK_CACHE = mapping
    return mapping


def _get_cik_for_ticker(ticker: str) -> Optional[str]:
    if not ticker:
        return None

    mapping = _load_ticker_to_cik_map()
    return mapping.get(str(ticker).upper())


def _get_companyfacts(ticker: str) -> Optional[dict]:
    cik = _get_cik_for_ticker(ticker)
    if not cik:
        return None

    url = SEC_COMPANYFACTS_URL.format(cik=cik)
    try:
        return _safe_get_json(url, headers=SEC_HEADERS, timeout=20)
    except Exception:
        return None


def _get_submissions(ticker: str) -> Optional[dict]:
    cik = _get_cik_for_ticker(ticker)
    if not cik:
        return None

    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    try:
        return _safe_get_json(url, headers=SEC_HEADERS, timeout=20)
    except Exception:
        return None


def _get_latest_entries_for_concept(
    companyfacts: dict,
    concept_names: List[str],
    unit: str = "USD",
) -> List[dict]:
    if not companyfacts:
        return []

    facts = companyfacts.get("facts", {}).get("us-gaap", {})
    for concept in concept_names:
        concept_block = facts.get(concept, {})
        units = concept_block.get("units", {})
        entries = units.get(unit, [])
        if entries:
            return entries

    return []


def _pick_latest_annual_entry(entries: List[dict]) -> Optional[dict]:
    if not entries:
        return None

    annual_forms = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
    filtered = [e for e in entries if e.get("form") in annual_forms]

    if not filtered:
        filtered = entries

    filtered = sorted(
        filtered,
        key=lambda x: (
            x.get("fy", 0) or 0,
            x.get("end", "") or "",
            x.get("filed", "") or "",
        ),
        reverse=True,
    )

    return filtered[0] if filtered else None


def _pick_entry_for_year(entries: List[dict], year: int) -> Optional[dict]:
    if not entries:
        return None

    annual_forms = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}

    filtered = [
        e for e in entries
        if e.get("fy") == year and e.get("form") in annual_forms
    ]

    if not filtered:
        filtered = [e for e in entries if e.get("fy") == year]

    filtered = sorted(
        filtered,
        key=lambda x: (
            x.get("end", "") or "",
            x.get("filed", "") or "",
        ),
        reverse=True,
    )

    return filtered[0] if filtered else None


def _extract_value_for_year(
    companyfacts: dict,
    concept_names: List[str],
    year: int,
    unit: str = "USD",
) -> Optional[dict]:
    entries = _get_latest_entries_for_concept(companyfacts, concept_names, unit=unit)
    entry = _pick_entry_for_year(entries, year)
    if not entry:
        return None
    return {
        "value": entry.get("val"),
        "fy": entry.get("fy"),
        "form": entry.get("form"),
        "filed": entry.get("filed"),
        "frame": entry.get("frame"),
    }


def _extract_latest_value(
    companyfacts: dict,
    concept_names: List[str],
    unit: str = "USD",
) -> Optional[Any]:
    entries = _get_latest_entries_for_concept(companyfacts, concept_names, unit=unit)
    entry = _pick_latest_annual_entry(entries)
    if not entry:
        return None
    return entry.get("val")


# ----------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------

@tool
def stock_price(ticker: str) -> str:
    """
    Get the current stock price for a ticker via Alpha Vantage.
    """
    if not ALPHAVANTAGE_API_KEY:
        return "Error: ALPHAVANTAGE_API_KEY no está configurada."

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}"
    )

    try:
        data = _safe_get_json(url, timeout=15)
        quote = data.get("Global Quote", {})

        if not quote:
            return f"Error: no se encontró precio para el ticker {ticker}."

        result = {
            "ticker": ticker.upper(),
            "price": quote.get("05. price"),
            "change": quote.get("09. change"),
            "change_percent": quote.get("10. change percent"),
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error ejecutando stock_price: {str(e)}"


@tool
def company_fundamentals(ticker: str) -> str:
    """
    Get latest company fundamentals from SEC companyfacts.
    """
    try:
        facts = _get_companyfacts(ticker)
        if not facts:
            return f"Error: no se pudieron recuperar fundamentales para {ticker}."

        result = {
            "empresa": facts.get("entityName"),
            "revenue": _extract_latest_value(facts, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]),
            "net_income": _extract_latest_value(facts, ["NetIncomeLoss"]),
            "total_assets": _extract_latest_value(facts, ["Assets"]),
            "total_liabilities": _extract_latest_value(facts, ["Liabilities"]),
        }

        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error ejecutando company_fundamentals: {str(e)}"


@tool
def company_events(ticker: str) -> str:
    """
    Get recent 8-K events for a company from SEC submissions.
    """
    try:
        submissions = _get_submissions(ticker)
        if not submissions:
            return f"Error: no se pudieron recuperar eventos para {ticker}."

        company_name = submissions.get("name")
        recent = submissions.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accession_numbers = recent.get("accessionNumber", [])

        events = []
        for form, date, accession in zip(forms, filing_dates, accession_numbers):
            if form == "8-K":
                events.append({
                    "date": date,
                    "type": form,
                    "description": form,
                    "accession_number": accession,
                })

        result = {
            "company": company_name,
            "events": events[:5],
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error ejecutando company_events: {str(e)}"


@tool
def company_financial_history(ticker: str, year: int) -> str:
    """
    Get historical financial statement data for a given fiscal year from SEC companyfacts.

    Especially useful for benchmark-style questions about:
    - capital expenditures
    - operating cash flow
    - revenue
    - net income
    - assets / liabilities
    """
    try:
        facts = _get_companyfacts(ticker)
        if not facts:
            return f"Error: no se pudieron recuperar datos históricos para {ticker}."

        result = {
            "company": facts.get("entityName"),
            "ticker": ticker.upper(),
            "fiscal_year": year,
            "capital_expenditures": _extract_value_for_year(
                facts,
                [
                    "PaymentsToAcquirePropertyPlantAndEquipment",
                    "CapitalExpendituresIncurredButNotYetPaid",
                ],
                year,
            ),
            "operating_cash_flow": _extract_value_for_year(
                facts,
                ["NetCashProvidedByUsedInOperatingActivities"],
                year,
            ),
            "revenue": _extract_value_for_year(
                facts,
                ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
                year,
            ),
            "net_income": _extract_value_for_year(
                facts,
                ["NetIncomeLoss"],
                year,
            ),
            "total_assets": _extract_value_for_year(
                facts,
                ["Assets"],
                year,
            ),
            "total_liabilities": _extract_value_for_year(
                facts,
                ["Liabilities"],
                year,
            ),
        }

        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error ejecutando company_financial_history: {str(e)}"