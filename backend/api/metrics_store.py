from threading import Lock

class MetricsStore:
    """
    Thread-safe store for inference metrics.
    """
    def __init__(self):
        self._lock = Lock()
        self.inference_count = 0
        self.total_latency_ms = 0.0

    def record(self, latency_ms: float):
        """
        Record latency (in milliseconds) for one inference.
        """
        with self._lock:
            self.inference_count += 1
            self.total_latency_ms += latency_ms

    def get_metrics(self) -> dict:
        """
        Return current metrics: count and average latency.
        """
        with self._lock:
            if self.inference_count:
                avg = self.total_latency_ms / self.inference_count
            else:
                avg = 0.0
            return {
                "inference_count": self.inference_count,
                "average_latency_ms": avg
            }

# Singleton instance
metrics_store = MetricsStore()
