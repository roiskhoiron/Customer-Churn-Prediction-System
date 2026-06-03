from prometheus_client import start_http_server, Gauge
import psutil, time, threading

cpu_gauge = Gauge('system_cpu_percent', 'CPU usage percent')
mem_gauge = Gauge('system_memory_percent', 'Memory usage percent')

def collect_system_metrics():
    while True:
        cpu_gauge.set(psutil.cpu_percent(interval=1))
        mem_gauge.set(psutil.virtual_memory().percent)
        time.sleep(5)

if __name__ == "__main__":
    start_http_server(8001)
    thread = threading.Thread(target=collect_system_metrics, daemon=True)
    thread.start()
    while True:
        time.sleep(60)
