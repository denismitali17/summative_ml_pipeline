import os
import sys
import csv
import statistics
from datetime import datetime


def analyze_csv(csv_path):
    rows = []
    with open(csv_path, newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            try:
                elapsed = float(r.get('elapsed_ms') or r.get('elapsed') or 0.0)
            except Exception:
                elapsed = 0.0
            try:
                start = float(r.get('start') or 0.0)
            except Exception:
                start = 0.0
            status = r.get('status_code') or r.get('status') or ''
            success = (r.get('success') or 'True').lower() in ('1','true','yes')
            rows.append({'elapsed': elapsed, 'start': start, 'status': status, 'success': success})

    if not rows:
        raise SystemExit('No rows found in CSV')

    elapsed_list = [r['elapsed'] for r in rows]
    starts = [r['start'] for r in rows]
    successes = sum(1 for r in rows if r['success'])
    failures = len(rows) - successes

    total_time_s = max(starts) - min(starts) if max(starts) > min(starts) else sum(elapsed_list)/1000.0
    throughput = len(rows) / total_time_s if total_time_s > 0 else float('nan')

    stats = {
        'requests': len(rows),
        'successes': successes,
        'failures': failures,
        'total_time_s': total_time_s,
        'throughput_rps': throughput,
        'p50_ms': statistics.median(elapsed_list),
        'p95_ms': percentile(elapsed_list, 95),
        'avg_ms': statistics.mean(elapsed_list),
    }

    out_dir = os.path.join(os.path.dirname(csv_path), '..', 'output')
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(csv_path))[0]
    summary_path = os.path.join(out_dir, f"{base}_summary.txt")
    with open(summary_path, 'w') as fh:
        fh.write('Load test summary\n')
        fh.write('Generated: ' + datetime.now().isoformat() + '\n\n')
        for k, v in stats.items():
            fh.write(f"{k}: {v}\n")

    # Try to plot if matplotlib available
    try:
        import matplotlib.pyplot as plt

        times = [s - min(starts) for s in starts]

        # time series (elapsed over time)
        plt.figure(figsize=(8, 4))
        plt.plot(times, elapsed_list, marker='.', linestyle='None', alpha=0.6)
        plt.xlabel('seconds since start')
        plt.ylabel('elapsed ms')
        plt.title('Request latency over time')
        timeseries_path = os.path.join(out_dir, f"{base}_timeseries.png")
        plt.savefig(timeseries_path, bbox_inches='tight')
        plt.close()

        # histogram
        plt.figure(figsize=(6, 4))
        plt.hist(elapsed_list, bins=30)
        plt.xlabel('elapsed ms')
        plt.ylabel('count')
        plt.title('Latency histogram')
        hist_path = os.path.join(out_dir, f"{base}_hist.png")
        plt.savefig(hist_path, bbox_inches='tight')
        plt.close()

        print('Plots saved:', timeseries_path, hist_path)
    except Exception as e:
        print('Matplotlib not available or plotting failed:', e)

    print('Summary saved to', summary_path)
    for k, v in stats.items():
        print(f'{k}: {v}')


def percentile(data, p):
    if not data:
        return None
    k = (len(data)-1) * (p/100.0)
    f = int(k)
    c = min(f+1, len(data)-1)
    if f == c:
        return sorted(data)[int(k)]
    d0 = sorted(data)[f] * (c-k)
    d1 = sorted(data)[c] * (k-f)
    return d0 + d1


def main():
    default = os.path.join('output', '')
    csv_glob = None
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # try to find latest CSV in output/
        out = os.path.abspath('output')
        files = [f for f in os.listdir(out) if f.startswith('load_test_') and f.endswith('.csv')]
        if not files:
            print('No CSV provided and none found in output/')
            return
        files.sort()
        csv_path = os.path.join(out, files[-1])

    if not os.path.exists(csv_path):
        print('CSV not found:', csv_path)
        return

    analyze_csv(csv_path)


if __name__ == '__main__':
    main()
