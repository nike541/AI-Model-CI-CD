# server.py
import litserve as ls
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# (STEP 1) - DEFINE THE API (compound AI system)
class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        # setup is called once at startup. Build a compound AI system (1+ models), connect DBs, load data, etc...
        self.model1 = lambda x: x**2
        self.model2 = lambda x: x**3
        
        # Initialize Prometheus metrics
        self.request_counter = Counter(
            'http_server_requests_total', 
            'Total HTTP requests',
            ['status']
        )
        self.latency_histogram = Histogram(
            'http_server_request_duration_seconds',
            'Request duration in seconds',
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25]
        )
        self.accuracy_gauge = Gauge(
            'model_prediction_accuracy',
            'Model prediction accuracy'
        )
        
        # Start Prometheus metrics server on a different port
        start_http_server(8001)
        
    def decode_request(self, request):
        # Convert the request payload to model input.
        try:
            result = request["input"]
            self.request_counter.labels(status='200').inc()
            return result
        except Exception as e:
            self.request_counter.labels(status='500').inc()
            raise

    def predict(self, x):
        # Track prediction latency
        start_time = time.time()
        try:
            # Easily build compound systems. Run inference and return the output.
            squared = self.model1(x)
            cubed = self.model2(x)
            output = squared + cubed
            
            # Calculate and update accuracy metric (example implementation)
            # In a real system, you would compare against ground truth
            expected = x**2 + x**3
            accuracy = 1.0 if abs(output - expected) < 0.001 else 0.95
            self.accuracy_gauge.set(accuracy)
            
            self.request_counter.labels(status='200').inc()
            return {"output": output}
        except Exception as e:
            self.request_counter.labels(status='500').inc()
            raise
        finally:
            self.latency_histogram.observe(time.time() - start_time)

    def encode_response(self, output):
        # Convert the model output to a response payload.
        try:
            result = {"output": output}
            self.request_counter.labels(status='200').inc()
            return result
        except Exception as e:
            self.request_counter.labels(status='500').inc()
            raise

# (STEP 2) - START THE SERVER
if __name__ == "__main__":
    # scale with advanced features (batching, GPUs, etc...)
    server = ls.LitServer(SimpleLitAPI(), accelerator="auto", max_batch_size=1)
    server.run(port=8000)