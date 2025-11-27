"""
Simple load tester for the prediction API using requests + ThreadPoolExecutor.
No compiled packages required — runs on plain Python and records latency/results to CSV.

Usage (PowerShell):
  python scripts\simple_load_test.py --image C:\path\to\image.png --requests 200 --concurrency 20 --host http://127.0.0.1:5000

Outputs:
  - CSV file under output/load_test_<timestamp>.csv with columns: request_id,start,elapsed_ms,status_code,success
  - A brief summary printed to stdout

Note: this is NOT a full-featured load test tool like Locust, but it's sufficient
for collecting latency/throughput data for your assignment when Locust isn't available.
"""
import argparse
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import csv
from datetime import datetime


def do_request(session, url, image_path, req_id):
    start = time.time()
    success = False
    status = None
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/png')}
            r = session.post(url, files=files, timeout=30)
            status = r.status_code
            success = r.ok
    except Exception as e:
        # treat as failed
        status = None
        success = False
    elapsed = (time.time() - start) * 1000.0
    return {'request_id': req_id, 'start': start, 'elapsed_ms': elapsed, 'status_code': status, 'success': success}


def run_load_test(host, image, total_requests, concurrency, output_dir):
    url = host.rstrip('/') + '/api/predict'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = output_dir / f'load_test_{ts}.csv'

    results = []
    session = requests.Session()

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(do_request, session, url, image, i): i for i in range(total_requests)}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
    end_time = time.time()

    # Save CSV
    with open(out_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['request_id', 'start', 'elapsed_ms', 'status_code', 'success'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Summary
    elapsed_total = end_time - start_time
    successes = sum(1 for r in results if r['success'])
    failures = len(results) - successes
    p50 = sorted([r['elapsed_ms'] for r in results])[len(results)//2] if results else 0
    p95 = sorted([r['elapsed_ms'] for r in results])[int(len(results)*0.95)-1] if results else 0
    avg = sum(r['elapsed_ms'] for r in results)/len(results) if results else 0

    summary = {
        'requests': len(results),
        'successes': successes,
        'failures': failures,
        'total_time_s': elapsed_total,
        'throughput_rps': len(results)/elapsed_total if elapsed_total>0 else None,
        'p50_ms': p50,
        'p95_ms': p95,
        'avg_ms': avg,
        'csv': str(out_file)
    }

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to an image file to POST')
    parser.add_argument('--requests', type=int, default=100, help='Total number of requests')
    parser.add_argument('--concurrency', type=int, default=10, help='Number of concurrent workers')
    parser.add_argument('--host', default='http://127.0.0.1:5000', help='Host URL for the API')
    parser.add_argument('--output', default='output', help='Directory to save CSV results')
    args = parser.parse_args()

    print('Running load test against', args.host)
    print('Image:', args.image)
    print('Total requests:', args.requests, 'Concurrency:', args.concurrency)
    summary = run_load_test(args.host, args.image, args.requests, args.concurrency, args.output)

    print('\nSummary:')
    for k, v in summary.items():
        print(f'  {k}: {v}')

    print('\nResults saved to', summary['csv'])
