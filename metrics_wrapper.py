import time
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from functools import wraps

class MetricsWrapper:
    def __init__(self, metrics_port=8001):
        # Initialize Prometheus metrics
        self.request_counter = Counter(
            'http_server_requests_total', 
            'Total HTTP requests',
            ['status', 'method']
        )
        self.latency_histogram = Histogram(
            'http_server_request_duration_seconds',
            'Request duration in seconds',
            ['method'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25]
        )
        self.accuracy_gauge = Gauge(
            'model_prediction_accuracy',
            'Model prediction accuracy'
        )
        
        # Start Prometheus metrics server
        start_http_server(metrics_port)
    
    def wrap_method(self, method_name):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    self.request_counter.labels(status='200', method=method_name).inc()
                    
                    # Special handling for predict method
                    if method_name == 'predict':
                        # Example accuracy calculation - replace with your actual logic
                        self.accuracy_gauge.set(0.98)
                    
                    return result
                except Exception as e:
                    self.request_counter.labels(status='500', method=method_name).inc()
                    raise
                finally:
                    self.latency_histogram.labels(method=method_name).observe(
                        time.time() - start_time
                    )
            return wrapper
        return decorator


class WrappedAPI:
    """A wrapper class to add metrics to a LitAPI."""
    def __init__(self, api_instance):
        self.api_instance = api_instance
        self.metrics = MetricsWrapper()
        
        # Wrap methods with metrics
        self.decode_request = self.metrics.wrap_method('decode_request')(
            self.api_instance.decode_request
        )
        self.predict = self.metrics.wrap_method('predict')(
            self.api_instance.predict
        )
        self.encode_response = self.metrics.wrap_method('encode_response')(
            self.api_instance.encode_response
        )
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying API instance."""
        return getattr(self.api_instance, name)


def wrap_litapi(api_class):
    """Wraps a LitAPI class with metrics monitoring."""
    def wrapper(*args, **kwargs):
        api_instance = api_class(*args, **kwargs)
        return WrappedAPI(api_instance)
    return wrapper
