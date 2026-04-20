"""Summarize and compare evaluation JSON outputs.

Usage:
    python -m backend.evaluation.summarize_eval backend/evaluation/mi_eval.json
    python -m backend.evaluation.summarize_eval old.json --compare new.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SHORT_RESPONSE_THRESHOLD = 40
ENGLISH_MARKERS = (
    "okay, let's",
    "the recommendation is",
    "recommendation is based",
    "first, i need to",
    "let me start by",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize evaluation JSON outputs.")
    parser.add_argument(
        "path",
        help="Path to the evaluation JSON file to summarize.",
    )
    parser.add_argument(
        "--compare",
        help="Optional second evaluation JSON file to compare against the first.",
    )
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def safe_median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = round((len(ordered) - 1) * pct)
    return ordered[index]


def looks_english(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in ENGLISH_MARKERS)


def build_case_note(result: dict[str, Any]) -> str:
    request = result.get("request", {})
    response_json = result.get("response_json", {})
    metadata = response_json.get("metadata", {})
    response = str(response_json.get("response", "")).strip().replace("\n", " ")
    prompt = str(request.get("prompt", "")).strip()
    company = metadata.get("company_name") or request.get("company_name") or "N/A"
    return (
        f"#{result.get('index', '?')} | {company} | "
        f"prompt='{prompt[:80]}' | response='{response[:120]}'"
    )


def analyze_eval(data: dict[str, Any]) -> dict[str, Any]:
    summary = data.get("summary", {})
    results = data.get("results", [])

    latencies: list[float] = []
    status_codes = Counter()
    warnings_counter = Counter()
    recommendations = Counter()
    companies = defaultdict(lambda: Counter())
    question_types = defaultdict(lambda: Counter())

    grounded_true = 0
    grounded_false = 0
    think_leaks = 0
    english_responses = 0
    short_responses = 0
    direct_true = 0
    direct_false = 0
    structured_true = 0
    structured_false = 0

    sample_think: list[str] = []
    sample_short: list[str] = []
    sample_ungrounded: list[str] = []

    for result in results:
        request = result.get("request", {})
        response_json = result.get("response_json", {})
        metadata = response_json.get("metadata", {})
        response_text = str(response_json.get("response", ""))
        latency = result.get("latency_seconds")

        if isinstance(latency, (int, float)):
            latencies.append(float(latency))

        status_codes[result.get("status_code")] += 1

        warning = metadata.get("warning")
        if isinstance(warning, list):
            for item in warning:
                warnings_counter[str(item)] += 1
        elif warning:
            warnings_counter[str(warning)] += 1

        recommendation = metadata.get("recommendation")
        if recommendation:
            recommendations[str(recommendation)] += 1

        company = str(metadata.get("company_name") or request.get("company_name") or "N/A")
        question_type = str(metadata.get("question_type") or "unknown")
        companies[company]["count"] += 1
        question_types[question_type]["count"] += 1

        grounded = metadata.get("critic_grounded")
        if grounded is True:
            grounded_true += 1
            companies[company]["grounded_true"] += 1
            question_types[question_type]["grounded_true"] += 1
        elif grounded is False:
            grounded_false += 1
            companies[company]["grounded_false"] += 1
            question_types[question_type]["grounded_false"] += 1
            if len(sample_ungrounded) < 3:
                sample_ungrounded.append(build_case_note(result))

        answered_directly = metadata.get("recommendation_answered_directly")
        if answered_directly is True:
            direct_true += 1
            companies[company]["direct_true"] += 1
            question_types[question_type]["direct_true"] += 1
        elif answered_directly is False:
            direct_false += 1
            companies[company]["direct_false"] += 1
            question_types[question_type]["direct_false"] += 1

        structured = metadata.get("market_has_structured_evidence")
        if structured is True:
            structured_true += 1
            companies[company]["structured_true"] += 1
            question_types[question_type]["structured_true"] += 1
        elif structured is False:
            structured_false += 1
            companies[company]["structured_false"] += 1
            question_types[question_type]["structured_false"] += 1

        if "<think>" in response_text.lower():
            think_leaks += 1
            companies[company]["think_leaks"] += 1
            question_types[question_type]["think_leaks"] += 1
            if len(sample_think) < 3:
                sample_think.append(build_case_note(result))

        if looks_english(response_text):
            english_responses += 1
            companies[company]["english_responses"] += 1
            question_types[question_type]["english_responses"] += 1

        if len(response_text.strip()) <= SHORT_RESPONSE_THRESHOLD:
            short_responses += 1
            companies[company]["short_responses"] += 1
            question_types[question_type]["short_responses"] += 1
            if len(sample_short) < 3:
                sample_short.append(build_case_note(result))

    return {
        "path_summary": summary,
        "total_results": len(results),
        "status_codes": dict(status_codes),
        "latency": {
            "avg": round(safe_mean(latencies), 3),
            "median": round(safe_median(latencies), 3),
            "p90": round(percentile(latencies, 0.90), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "quality": {
            "grounded_true": grounded_true,
            "grounded_false": grounded_false,
            "think_leaks": think_leaks,
            "english_responses": english_responses,
            "short_responses": short_responses,
            "direct_true": direct_true,
            "direct_false": direct_false,
            "structured_true": structured_true,
            "structured_false": structured_false,
        },
        "recommendations": dict(recommendations),
        "warnings": dict(warnings_counter),
        "companies": {key: dict(value) for key, value in sorted(companies.items())},
        "question_types": {key: dict(value) for key, value in sorted(question_types.items())},
        "samples": {
            "think_leaks": sample_think,
            "short_responses": sample_short,
            "ungrounded": sample_ungrounded,
        },
    }


def format_pct(part: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{(part / total) * 100:.1f}%"


def print_summary(analysis: dict[str, Any], title: str) -> None:
    total = analysis["total_results"]
    quality = analysis["quality"]

    print(f"=== {title} ===")
    print(f"Total casos: {total}")
    print(f"HTTP 200: {analysis['status_codes'].get(200, 0)}")
    print(
        "Latencia: "
        f"avg={analysis['latency']['avg']}s | "
        f"median={analysis['latency']['median']}s | "
        f"p90={analysis['latency']['p90']}s | "
        f"max={analysis['latency']['max']}s"
    )
    print(
        "Calidad: "
        f"grounded={quality['grounded_true']} ({format_pct(quality['grounded_true'], total)}) | "
        f"think_leaks={quality['think_leaks']} | "
        f"english={quality['english_responses']} | "
        f"short={quality['short_responses']} | "
        f"direct={quality['direct_true']} ({format_pct(quality['direct_true'], total)}) | "
        f"structured={quality['structured_true']} ({format_pct(quality['structured_true'], total)})"
    )

    if analysis["recommendations"]:
        items = ", ".join(f"{k}={v}" for k, v in sorted(analysis["recommendations"].items()))
        print(f"Recomendaciones: {items}")

    if analysis["warnings"]:
        items = ", ".join(f"{k}={v}" for k, v in sorted(analysis["warnings"].items()))
        print(f"Warnings: {items}")

    print("Por empresa:")
    for company, stats in analysis["companies"].items():
        count = stats.get("count", 0)
        grounded = stats.get("grounded_true", 0)
        direct = stats.get("direct_true", 0)
        structured = stats.get("structured_true", 0)
        think_leaks = stats.get("think_leaks", 0)
        print(
            f"- {company}: n={count}, grounded={grounded}, direct={direct}, "
            f"structured={structured}, think_leaks={think_leaks}"
        )

    print("Por tipo de pregunta:")
    for qtype, stats in analysis["question_types"].items():
        count = stats.get("count", 0)
        grounded = stats.get("grounded_true", 0)
        direct = stats.get("direct_true", 0)
        short = stats.get("short_responses", 0)
        print(
            f"- {qtype}: n={count}, grounded={grounded}, direct={direct}, short={short}"
        )

    samples = analysis["samples"]
    if samples["think_leaks"]:
        print("Ejemplos con think leak:")
        for item in samples["think_leaks"]:
            print(f"- {item}")
    if samples["short_responses"]:
        print("Ejemplos de respuesta corta:")
        for item in samples["short_responses"]:
            print(f"- {item}")
    if samples["ungrounded"]:
        print("Ejemplos no grounded:")
        for item in samples["ungrounded"]:
            print(f"- {item}")

    print()


def compare_metric(name: str, old: int | float, new: int | float, invert_good: bool = False) -> str:
    delta = new - old
    sign = "+" if delta >= 0 else ""
    if invert_good:
        status = "mejora" if delta < 0 else "empeora" if delta > 0 else "igual"
    else:
        status = "mejora" if delta > 0 else "empeora" if delta < 0 else "igual"
    return f"- {name}: {old} -> {new} ({sign}{delta}, {status})"


def print_comparison(old_analysis: dict[str, Any], new_analysis: dict[str, Any]) -> None:
    old_q = old_analysis["quality"]
    new_q = new_analysis["quality"]
    print("=== Comparacion ===")
    print(compare_metric("grounded_true", old_q["grounded_true"], new_q["grounded_true"]))
    print(compare_metric("think_leaks", old_q["think_leaks"], new_q["think_leaks"], invert_good=True))
    print(compare_metric("english_responses", old_q["english_responses"], new_q["english_responses"], invert_good=True))
    print(compare_metric("short_responses", old_q["short_responses"], new_q["short_responses"], invert_good=True))
    print(compare_metric("direct_true", old_q["direct_true"], new_q["direct_true"]))
    print(compare_metric("structured_true", old_q["structured_true"], new_q["structured_true"]))
    print(compare_metric("latency_avg", old_analysis["latency"]["avg"], new_analysis["latency"]["avg"], invert_good=True))
    print()


def main() -> None:
    args = parse_args()

    primary_analysis = analyze_eval(load_json(args.path))
    print_summary(primary_analysis, f"Resumen: {args.path}")

    if args.compare:
        compare_analysis = analyze_eval(load_json(args.compare))
        print_summary(compare_analysis, f"Resumen: {args.compare}")
        print_comparison(primary_analysis, compare_analysis)


if __name__ == "__main__":
    main()
