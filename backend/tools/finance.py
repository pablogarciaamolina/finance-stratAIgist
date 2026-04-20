"""
Financial tools for the multi-agent system.

Provides LangChain tools for:
- current stock price
- latest company fundamentals
- recent company events
- historical financial statement data by fiscal year

Sources:
- Alpha Vantage
- SEC EDGAR company tickers
- SEC companyfacts
- SEC submissions
"""

from __future__ import annotations

import json
import os
import time
from datetime import date
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
ALPHAVANTAGE_API_KEYS = [
    key
    for key in [
        os.getenv("ALPHAVANTAGE_API_KEY"),
        os.getenv("ALPHAVANTAGE_API_KEY2"),
        os.getenv("ALPHAVANTAGE_API_KEY3"),
    ]
    if key
]
SEC_CONTACT_EMAIL = os.getenv("SEC_CONTACT_EMAIL", "contact@example.com")

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

SEC_HEADERS = {
    "User-Agent": f"FinanceStratAIgist/0.1 ({SEC_CONTACT_EMAIL})",
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

_TICKER_TO_CIK_CACHE: Optional[Dict[str, str]] = None
_ALPHAVANTAGE_ROTATION_LOCK = Lock()
_ALPHAVANTAGE_KEY_EXHAUSTED_ON: Dict[str, str] = {}
_ALPHAVANTAGE_CURSOR = 0


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _format_exception_message(exc: Exception) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__


def _today_key() -> str:
    return date.today().isoformat()


def _mask_api_key(api_key: str) -> str:
    if not api_key:
        return "unknown"
    if len(api_key) <= 4:
        return "***"
    return f"...{api_key[-4:]}"


def _is_alpha_vantage_rate_limited_detail(detail: Optional[str]) -> bool:
    lower = (detail or "").lower()
    return any(
        token in lower
        for token in [
            "requests per day",
            "rate limit",
            "call frequency",
            "premium",
        ]
    )


def _get_alpha_vantage_candidate_keys() -> List[str]:
    if not ALPHAVANTAGE_API_KEYS:
        return []

    today = _today_key()

    with _ALPHAVANTAGE_ROTATION_LOCK:
        available_keys = [
            api_key
            for api_key in ALPHAVANTAGE_API_KEYS
            if _ALPHAVANTAGE_KEY_EXHAUSTED_ON.get(api_key) != today
        ]

        if not available_keys:
            return []

        ordered_all = list(ALPHAVANTAGE_API_KEYS)
        start = _ALPHAVANTAGE_CURSOR % len(ordered_all)
        rotated = ordered_all[start:] + ordered_all[:start]

        return [api_key for api_key in rotated if api_key in available_keys]


def _mark_alpha_vantage_key_rate_limited(api_key: str) -> None:
    if not api_key:
        return

    with _ALPHAVANTAGE_ROTATION_LOCK:
        _ALPHAVANTAGE_KEY_EXHAUSTED_ON[api_key] = _today_key()


def _mark_alpha_vantage_key_success(api_key: str) -> None:
    global _ALPHAVANTAGE_CURSOR

    if not api_key or api_key not in ALPHAVANTAGE_API_KEYS:
        return

    with _ALPHAVANTAGE_ROTATION_LOCK:
        _ALPHAVANTAGE_KEY_EXHAUSTED_ON.pop(api_key, None)
        current_index = ALPHAVANTAGE_API_KEYS.index(api_key)
        _ALPHAVANTAGE_CURSOR = (current_index + 1) % len(ALPHAVANTAGE_API_KEYS)


def _safe_get_json(
    url: str,
    *,
    headers: Optional[dict] = None,
    params: Optional[dict] = None,
    timeout: int = 15,
    retries: int = 2,
    backoff_seconds: float = 0.6,
) -> dict:
    last_error: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=timeout)

            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(backoff_seconds * (attempt + 1))
                continue

            response.raise_for_status()

            try:
                return response.json()
            except ValueError as exc:
                preview = response.text[:200].replace("\n", " ").strip()
                raise ValueError(f"respuesta no JSON: {preview}") from exc
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(backoff_seconds * (attempt + 1))

    if last_error is not None:
        raise last_error
    raise RuntimeError("Error desconocido recuperando JSON.")


def _normalize_ticker_records(data: Any) -> List[dict]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        if all(isinstance(item, dict) for item in data.values()):
            return [item for item in data.values() if isinstance(item, dict)]
        if isinstance(data.get("data"), list):
            return [item for item in data["data"] if isinstance(item, dict)]

    return []


def _load_ticker_to_cik_map() -> Dict[str, str]:
    global _TICKER_TO_CIK_CACHE

    if _TICKER_TO_CIK_CACHE is not None:
        return _TICKER_TO_CIK_CACHE

    data = _safe_get_json(SEC_TICKERS_URL, headers=SEC_HEADERS, timeout=20, retries=2)
    mapping: Dict[str, str] = {}

    for item in _normalize_ticker_records(data):
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


def _get_companyfacts(ticker: str) -> Tuple[Optional[dict], Optional[str]]:
    try:
        cik = _get_cik_for_ticker(ticker)
    except Exception as exc:
        return None, f"No se pudo cargar el mapa ticker->CIK: {_format_exception_message(exc)}"

    if not cik:
        return None, f"No se encontro CIK para {ticker}."

    url = SEC_COMPANYFACTS_URL.format(cik=cik)
    try:
        return _safe_get_json(url, headers=SEC_HEADERS, timeout=20, retries=2), None
    except Exception as exc:
        return None, (
            f"No se pudo recuperar companyfacts de SEC para {ticker} "
            f"(CIK {cik}): {_format_exception_message(exc)}"
        )


def _get_submissions(ticker: str) -> Tuple[Optional[dict], Optional[str]]:
    try:
        cik = _get_cik_for_ticker(ticker)
    except Exception as exc:
        return None, f"No se pudo cargar el mapa ticker->CIK: {_format_exception_message(exc)}"

    if not cik:
        return None, f"No se encontro CIK para {ticker}."

    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    try:
        return _safe_get_json(url, headers=SEC_HEADERS, timeout=20, retries=2), None
    except Exception as exc:
        return None, (
            f"No se pudo recuperar submissions de SEC para {ticker} "
            f"(CIK {cik}): {_format_exception_message(exc)}"
        )


def _alpha_vantage_request(function: str, **params: Any) -> dict:
    if not ALPHAVANTAGE_API_KEYS:
        raise RuntimeError("ALPHAVANTAGE_API_KEY no esta configurada.")

    last_error: Optional[Exception] = None
    rate_limited_keys: List[str] = []
    candidate_keys = _get_alpha_vantage_candidate_keys()

    if not candidate_keys:
        return {
            "Note": (
                "Todas las API keys de Alpha Vantage configuradas ya estan marcadas como agotadas hoy. "
                "Espera al reinicio del limite diario o anade nuevas keys."
            )
        }

    for api_key in candidate_keys:
        full_params = {"function": function, "apikey": api_key}
        for key, value in params.items():
            if value is not None:
                full_params[key] = value

        try:
            payload = _safe_get_json(
                "https://www.alphavantage.co/query",
                params=full_params,
                timeout=15,
                retries=1,
            )
        except Exception as exc:
            last_error = exc
            continue

        detail = _alpha_vantage_error_detail(payload)
        if _is_alpha_vantage_rate_limited_detail(detail):
            _mark_alpha_vantage_key_rate_limited(api_key)
            rate_limited_keys.append(_mask_api_key(api_key))
            continue

        _mark_alpha_vantage_key_success(api_key)
        return payload

    if rate_limited_keys:
        unique_keys = ", ".join(dict.fromkeys(rate_limited_keys))
        return {
            "Note": (
                "Todas las API keys de Alpha Vantage disponibles parecen haber alcanzado el limite "
                f"diario o de frecuencia para esta sesion. Keys afectadas: {unique_keys}"
            )
        }

    if last_error is not None:
        raise last_error

    raise RuntimeError("No se pudo obtener respuesta de Alpha Vantage con ninguna API key.")


def _alpha_vantage_error_detail(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return "respuesta inesperada de Alpha Vantage"

    for key in ("Error Message", "Note", "Information"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _extract_price_from_global_quote(ticker: str, payload: Any) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None

    quote = payload.get("Global Quote", {})
    if not isinstance(quote, dict):
        return None

    price = quote.get("05. price")
    if not price:
        return None

    return {
        "ticker": ticker.upper(),
        "price": price,
        "change": quote.get("09. change"),
        "change_percent": quote.get("10. change percent"),
        "as_of": quote.get("07. latest trading day"),
        "source": "alphavantage_global_quote",
    }


def _extract_price_from_daily_series(ticker: str, payload: Any) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None

    series = payload.get("Time Series (Daily)", {})
    if not isinstance(series, dict) or not series:
        return None

    dates = sorted(series.keys(), reverse=True)
    latest_date = dates[0]
    latest = series.get(latest_date, {})
    price = latest.get("4. close")
    if not price:
        return None

    current_close = _safe_float(price)
    previous_close = None
    if len(dates) > 1:
        previous_close = _safe_float(series.get(dates[1], {}).get("4. close"))

    change = None
    change_percent = None
    if current_close is not None and previous_close not in (None, 0):
        delta = current_close - previous_close
        change = f"{delta:.2f}"
        change_percent = f"{(delta / previous_close) * 100:.2f}%"

    return {
        "ticker": ticker.upper(),
        "price": price,
        "change": change,
        "change_percent": change_percent,
        "as_of": latest_date,
        "source": "alphavantage_time_series_daily",
    }


def _recent_filings(
    forms: Sequence[Any],
    filing_dates: Sequence[Any],
    accession_numbers: Sequence[Any],
    preferred_forms: Sequence[str],
    limit: int = 5,
) -> List[dict]:
    preferred = set(preferred_forms)
    events: List[dict] = []

    for form, date, accession in zip(forms, filing_dates, accession_numbers):
        if form in preferred:
            events.append(
                {
                    "date": date,
                    "type": form,
                    "description": form,
                    "accession_number": accession,
                }
            )
        if len(events) >= limit:
            break

    return events


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


def _safe_sort_year(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_sort_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _pick_latest_annual_entry(entries: List[dict]) -> Optional[dict]:
    if not entries:
        return None

    annual_forms = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
    filtered = [entry for entry in entries if entry.get("form") in annual_forms]

    if not filtered:
        filtered = entries

    filtered = sorted(
        filtered,
        key=lambda entry: (
            _safe_sort_year(entry.get("fy")),
            _safe_sort_text(entry.get("end")),
            _safe_sort_text(entry.get("filed")),
        ),
        reverse=True,
    )

    return filtered[0] if filtered else None


def _pick_entry_for_year(entries: List[dict], year: int) -> Optional[dict]:
    if not entries:
        return None

    annual_forms = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
    filtered = [
        entry for entry in entries if entry.get("fy") == year and entry.get("form") in annual_forms
    ]

    if not filtered:
        filtered = [entry for entry in entries if entry.get("fy") == year]

    filtered = sorted(
        filtered,
        key=lambda entry: (
            _safe_sort_text(entry.get("end")),
            _safe_sort_text(entry.get("filed")),
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


def _extract_latest_fact(
    companyfacts: dict,
    concept_names: List[str],
    unit: str = "USD",
) -> Optional[dict]:
    entries = _get_latest_entries_for_concept(companyfacts, concept_names, unit=unit)
    entry = _pick_latest_annual_entry(entries)
    if not entry:
        return None

    return {
        "value": entry.get("val"),
        "fy": entry.get("fy"),
        "form": entry.get("form"),
        "filed": entry.get("filed"),
        "frame": entry.get("frame"),
    }


# ----------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------

@tool
def stock_price(ticker: str) -> str:
    """
    Get the current stock price for a ticker via Alpha Vantage.
    Falls back to TIME_SERIES_DAILY when GLOBAL_QUOTE is empty.
    """
    if not ALPHAVANTAGE_API_KEYS:
        return "Error: ALPHAVANTAGE_API_KEY no esta configurada."

    try:
        global_quote = _alpha_vantage_request("GLOBAL_QUOTE", symbol=ticker)
        result = _extract_price_from_global_quote(ticker, global_quote)
        global_quote_error = _alpha_vantage_error_detail(global_quote)

        if result:
            return json.dumps(result, ensure_ascii=False, indent=2)

        daily_series = _alpha_vantage_request(
            "TIME_SERIES_DAILY",
            symbol=ticker,
            outputsize="compact",
        )
        result = _extract_price_from_daily_series(ticker, daily_series)
        daily_series_error = _alpha_vantage_error_detail(daily_series)

        if result:
            return json.dumps(result, ensure_ascii=False, indent=2)

        error_messages = [message for message in [global_quote_error, daily_series_error] if message]
        if error_messages:
            return (
                f"Error: no se pudo recuperar precio para {ticker}. "
                f"Detalle: {' | '.join(error_messages)}"
            )

        return f"Error: no se encontro precio utilizable para el ticker {ticker}."
    except Exception as exc:
        return f"Error ejecutando stock_price: {_format_exception_message(exc)}"


@tool
def company_fundamentals(ticker: str) -> str:
    """
    Get latest company fundamentals from SEC companyfacts.
    Returns partial structured data when SEC responds but some metrics are missing.
    """
    try:
        facts, fetch_error = _get_companyfacts(ticker)
        if not facts:
            return f"Error: no se pudieron recuperar fundamentales para {ticker}. {fetch_error or ''}".strip()

        revenue_fact = _extract_latest_fact(
            facts,
            ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
        )
        net_income_fact = _extract_latest_fact(facts, ["NetIncomeLoss"])
        total_assets_fact = _extract_latest_fact(facts, ["Assets"])
        total_liabilities_fact = _extract_latest_fact(facts, ["Liabilities"])

        latest_report = next(
            (
                fact for fact in [
                    revenue_fact,
                    net_income_fact,
                    total_assets_fact,
                    total_liabilities_fact,
                ]
                if fact is not None
            ),
            None,
        )

        result = {
            "empresa": facts.get("entityName"),
            "ticker": ticker.upper(),
            "latest_report": {
                "fiscal_year": latest_report.get("fy") if latest_report else None,
                "form": latest_report.get("form") if latest_report else None,
                "filed": latest_report.get("filed") if latest_report else None,
            },
            "revenue": revenue_fact.get("value") if revenue_fact else None,
            "net_income": net_income_fact.get("value") if net_income_fact else None,
            "total_assets": total_assets_fact.get("value") if total_assets_fact else None,
            "total_liabilities": total_liabilities_fact.get("value") if total_liabilities_fact else None,
            "data_quality": (
                "complete"
                if all([revenue_fact, net_income_fact, total_assets_fact, total_liabilities_fact])
                else "partial"
            ),
        }

        if not any([revenue_fact, net_income_fact, total_assets_fact, total_liabilities_fact]):
            result["data_quality"] = "empty"
            result["warning"] = (
                "SEC companyfacts disponible, pero no se encontraron las metricas anuales objetivo."
            )

        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"Error ejecutando company_fundamentals: {_format_exception_message(exc)}"


@tool
def company_events(ticker: str) -> str:
    """
    Get recent company events from SEC submissions.
    Prefers 8-K / 6-K and falls back to recent material filings if needed.
    """
    try:
        submissions, fetch_error = _get_submissions(ticker)
        if not submissions:
            return f"Error: no se pudieron recuperar eventos para {ticker}. {fetch_error or ''}".strip()

        company_name = submissions.get("name")
        recent = submissions.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accession_numbers = recent.get("accessionNumber", [])

        events = _recent_filings(
            forms=forms,
            filing_dates=filing_dates,
            accession_numbers=accession_numbers,
            preferred_forms=["8-K", "6-K"],
            limit=5,
        )
        event_source = "8-K/6-K"

        if not events:
            events = _recent_filings(
                forms=forms,
                filing_dates=filing_dates,
                accession_numbers=accession_numbers,
                preferred_forms=["10-K", "10-Q", "20-F", "40-F"],
                limit=5,
            )
            event_source = "material_filings_fallback"

        result = {
            "company": company_name,
            "ticker": ticker.upper(),
            "event_source": event_source,
            "events": events[:5],
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"Error ejecutando company_events: {_format_exception_message(exc)}"


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
        facts, fetch_error = _get_companyfacts(ticker)
        if not facts:
            return (
                f"Error: no se pudieron recuperar datos historicos para {ticker}. "
                f"{fetch_error or ''}"
            ).strip()

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

        available_fields = [
            key
            for key in [
                "capital_expenditures",
                "operating_cash_flow",
                "revenue",
                "net_income",
                "total_assets",
                "total_liabilities",
            ]
            if result.get(key) is not None
        ]

        result["data_quality"] = "complete" if len(available_fields) >= 4 else "partial"
        result["available_fields"] = available_fields

        if not available_fields:
            result["data_quality"] = "empty"
            result["warning"] = (
                "SEC companyfacts disponible, pero no se encontraron metricas historicas "
                f"para el ejercicio fiscal {year}."
            )

        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"Error ejecutando company_financial_history: {_format_exception_message(exc)}"
