import json
import sys

def calculate_triage_utility(summary_path):
    try:
        with open(summary_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {summary_path}")
        return

    res = data['scenario_summary'].get('24', {})
    if not res:
        print("No data for scenario 24")
        return

    t_mean = res.get('triage_auprc', {}).get('mean')
    rf_mean = res.get('rf_auprc', {}).get('mean')
    lstm_mean = res.get('lstm_auprc', {}).get('mean')

    if any(x is None for x in [t_mean, rf_mean, lstm_mean]):
        print("Incomplete data in summary.")
        return

    print(f"Triage AUPRC: {t_mean:.4f}")
    print(f"RF AUPRC:     {rf_mean:.4f}")
    print(f"LSTM AUPRC:   {lstm_mean:.4f}")

    best_baseline = max(rf_mean, lstm_mean)
    utility = (t_mean - best_baseline) / best_baseline
    print(f"Relative Utility: {utility:.2%}")

if __name__ == '__main__':
    calculate_triage_utility('dev/active/artifacts/validation_package_full/summary.json')
