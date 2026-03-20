# src/mlops/metrics_exporter.py

from prometheus_client import Counter, Gauge, Histogram, start_http_server


class TrainingMetrics:
    """Expose training metrics for Prometheus scraping."""

    def __init__(self, port=8000):
        # Federation metrics
        self.round_counter = Counter(
            "fed_rounds_total",
            "Total federation rounds completed",
        )
        self.global_reward = Gauge(
            "fed_global_reward",
            "Global average reward across all clients",
        )
        self.client_reward = Gauge(
            "fed_client_reward",
            "Per-client reward",
            ["client_id"],
        )
        self.weight_divergence = Gauge(
            "fed_weight_divergence",
            "Average client weight drift from global model",
        )
        self.round_duration = Histogram(
            "fed_round_duration_seconds",
            "Duration of each federation round",
            buckets=[1, 5, 10, 30, 60, 120, 300],
        )
        self.model_version = Gauge(
            "fed_model_version",
            "Current global model version/round",
        )
        self.sla_rate = Gauge(
            "fed_sla_compliance_rate",
            "Global SLA compliance rate",
        )
        self.active_clients = Gauge(
            "fed_active_clients",
            "Number of active clients in current round",
        )

        # Start HTTP server for Prometheus to scrape
        start_http_server(port)
        print(f"Prometheus metrics server started on port {port}")

    def record_round(
        self,
        round_num: int,
        global_reward: float,
        client_rewards: list,
        duration: float,
        divergence: float = 0.0,
        sla_rate: float = 0.0,
    ):
        self.round_counter.inc()
        self.global_reward.set(global_reward)
        self.model_version.set(round_num)
        self.weight_divergence.set(divergence)
        self.round_duration.observe(duration)
        self.sla_rate.set(sla_rate)
        self.active_clients.set(len(client_rewards))

        for cid, reward in enumerate(client_rewards):
            self.client_reward.labels(client_id=str(cid)).set(reward)
