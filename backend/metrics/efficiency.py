from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List

import numpy as np


def measure_latency(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start


def latency_stats(latencies: Iterable[float]) -> Dict[str, float]:
    latencies = [float(x) for x in latencies if x is not None]
    if not latencies:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}

    return {
        "mean": float(np.mean(latencies)),
        "p95": float(np.percentile(latencies, 95)),
        "max": float(np.max(latencies)),
    }


def compute_throughput(n_requests: int, total_time: float) -> float:
    return float(n_requests / total_time) if total_time and total_time > 0 else 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def aggregate_tokens(metrics_list: List[Dict[str, Any]]) -> Dict[str, int]:
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    for metric in metrics_list:
        tokens = metric.get("tokens", {})

        if isinstance(tokens, dict):
            input_tokens += int(tokens.get("input_tokens", 0) or 0)
            output_tokens += int(tokens.get("output_tokens", 0) or 0)
            total_tokens += int(tokens.get("total_tokens", 0) or 0)
        elif isinstance(tokens, (int, float)):
            total_tokens += int(tokens)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def compute_efficiency(metrics_list: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
    latencies = [
        _safe_float(metric.get("latency"))
        for metric in metrics_list
        if metric.get("latency") is not None
    ]
    latency_summary = latency_stats(latencies)
    token_summary = aggregate_tokens(metrics_list)

    return {
        "num_metric_events": len(metrics_list),
        "latency_mean": latency_summary["mean"],
        "latency_p95": latency_summary["p95"],
        "latency_max": latency_summary["max"],
        "throughput": compute_throughput(len(metrics_list), total_time),
        "tokens_input_total": token_summary["input_tokens"],
        "tokens_output_total": token_summary["output_tokens"],
        "tokens_total": token_summary["total_tokens"],
        "total_time": float(total_time) if total_time is not None else 0.0,
    }
