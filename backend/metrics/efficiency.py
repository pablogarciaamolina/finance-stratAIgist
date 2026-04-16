import time
import numpy as np


def measure_latency(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    
    return result, end - start


def latency_stats(latencies):
    return {
        "mean": np.mean(latencies),
        "p95": np.percentile(latencies, 95),
        "max": np.max(latencies),
    }

def compute_throughput(n_requests, total_time):
    return n_requests / total_time if total_time > 0 else 0


def aggregate_tokens(metrics_list):
    return sum(m.get("tokens", 0) for m in metrics_list)

def compute_efficiency(metrics_list, total_time):
    latencies = [m["latency"] for m in metrics_list if "latency" in m]
    
    return {
        "latency_mean": np.mean(latencies) if latencies else 0,
        "latency_p95": np.percentile(latencies, 95) if latencies else 0,
        "throughput": compute_throughput(len(metrics_list), total_time),
        "tokens_total": aggregate_tokens(metrics_list),
    }