"""
benchmark_stream.py
-------------------
Streams the Wednesday dataset chunk-by-chunk. SVD is batched per chunk
(fast matrix ops), monoid is evaluated row-by-row (required for rule logic).
This simulates the live streaming pipeline while remaining practical to run.
"""

import pandas as pd
import numpy as np
import joblib
import json
import time
from sklearn.metrics import classification_report, confusion_matrix
from algebraic_engine import AlgebraicMonoidEngine

scaler     = joblib.load('subspace_models/geometric_scaler.joblib')
p_u_matrix = np.load('subspace_models/p_u_matrix.npy')

with open('subspace_models/subspace_config.json', 'r') as f:
    config = json.load(f)

EPSILON           = config['epsilon_threshold']
EXPECTED_FEATURES = scaler.feature_names_in_
logic_engine      = AlgebraicMonoidEngine()

print(f"[STREAM BENCHMARK] Epsilon: {EPSILON:.4f}")
print(f"[STREAM BENCHMARK] Streaming Wednesday dataset...\n")

true_labels      = []
svd_predictions  = []
mono_predictions = []
combined_preds   = []
raw_labels       = []

chunk_size  = 10_000
total_flows = 0
start_time  = time.time()

for chunk in pd.read_csv('Wednesday-workingHours.pcap_ISCX.csv', chunksize=chunk_size):
    chunk.columns = chunk.columns.str.strip()
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk.dropna(inplace=True)

    if chunk.empty:
        continue

    # --- SVD arm: batched per chunk (fast) ---
    x_chunk  = chunk.reindex(columns=EXPECTED_FEATURES, fill_value=0.0)
    x_scaled = scaler.transform(x_chunk)
    w_matrix = x_scaled - np.dot(x_scaled, p_u_matrix)
    residuals = np.linalg.norm(w_matrix, axis=1)
    svd_flags = residuals > EPSILON

    # --- Monoid arm: row-by-row (required for rule logic) ---
    for i, (_, row) in enumerate(chunk.iterrows()):
        row_dict   = row.to_dict()
        true_label = row_dict['Label']

        svd_flag    = bool(svd_flags[i])
        mono_result = logic_engine.evaluate_state_loop(row_dict)
        mono_flag   = mono_result['algebraic_anomaly']
        combined    = svd_flag or mono_flag

        raw_labels.append(true_label)
        true_labels.append(0 if true_label == 'BENIGN' else 1)
        svd_predictions.append(1 if svd_flag  else 0)
        mono_predictions.append(1 if mono_flag else 0)
        combined_preds.append(1 if combined   else 0)

        total_flows += 1

    if total_flows % 5_000 == 0:
        elapsed = time.time() - start_time
        rate    = total_flows / elapsed if elapsed > 0 else 0
        eta     = (691_406 - total_flows) / rate if rate > 0 else 0
        print(f"  {total_flows:>7,} flows | {elapsed:.1f}s | {rate:.0f} flows/sec | ETA ~{eta:.0f}s")

elapsed = time.time() - start_time
print(f"\n[STREAM BENCHMARK] Done. {total_flows:,} flows in {elapsed:.1f}s ({total_flows/elapsed:.0f} flows/sec)")
print(f"[STREAM BENCHMARK] Attack: {sum(true_labels):,} | Benign: {total_flows - sum(true_labels):,}")

label_names = ['BENIGN', 'ATTACK']

print("\n" + "="*60)
print("ARM 1: SVD / Subspace Geometric Engine")
print("="*60)
print(classification_report(true_labels, svd_predictions, target_names=label_names, digits=4))

print("="*60)
print("ARM 2: Monoid / Algebraic State Machine")
print("="*60)
print(classification_report(true_labels, mono_predictions, target_names=label_names, digits=4))

print("="*60)
print("COMBINED SYSTEM (SVD OR Monoid)")
print("="*60)
print(classification_report(true_labels, combined_preds, target_names=label_names, digits=4))

print("="*60)
print("CONFUSION MATRIX — Combined System")
print("Rows=True | Cols=Predicted | [BENIGN, ATTACK]")
print("="*60)
cm = confusion_matrix(true_labels, combined_preds)
print(pd.DataFrame(cm,
    index  =['True BENIGN', 'True ATTACK'],
    columns=['Pred BENIGN', 'Pred ATTACK']))

print("\n" + "="*60)
print("PER-CLASS DETECTION RATE — Combined System")
print("="*60)
df_results = pd.DataFrame({'true': raw_labels, 'detected': [bool(p) for p in combined_preds]})

for cls in sorted(df_results['true'].unique()):
    subset = df_results[df_results['true'] == cls]
    n = len(subset)
    if cls == 'BENIGN':
        correct = (~subset['detected']).sum()
        print(f"  {cls:<35} n={n:>7,}  correctly passed: {correct:>7,}  ({100*correct/n:.2f}%)")
    else:
        detected = subset['detected'].sum()
        print(f"  {cls:<35} n={n:>7,}  detected:         {detected:>7,}  ({100*detected/n:.2f}%)")

# --- Save results ---
import os
from datetime import datetime
os.makedirs('results', exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

report = {
    "run_timestamp": ts,
    "dataset": "Wednesday-workingHours.pcap_ISCX.csv",
    "epsilon": round(float(EPSILON), 6),
    "total_flows": total_flows,
    "throughput_flows_per_sec": round(total_flows / elapsed, 2),
    "elapsed_seconds": round(elapsed, 2),
}

json_path = f"results/benchmark_stream_{ts}.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=4)

txt_path = f"results/benchmark_stream_{ts}.txt"
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("SUBSPACE AEGIS — STREAM BENCHMARK REPORT\n")
    f.write(f"Run: {ts}\n")
    f.write(f"Total flows: {total_flows:,}\n")
    f.write(f"Elapsed: {elapsed:.1f}s\n")
    f.write(f"Throughput: {total_flows/elapsed:.0f} flows/sec\n\n")
    f.write("ARM 1: SVD\n")
    f.write(classification_report(true_labels, svd_predictions, target_names=label_names, digits=4))
    f.write("\nARM 2: Monoid\n")
    f.write(classification_report(true_labels, mono_predictions, target_names=label_names, digits=4))
    f.write("\nCOMBINED\n")
    f.write(classification_report(true_labels, combined_preds, target_names=label_names, digits=4))

print(f"\n[OUTPUT] Saved to {json_path} and {txt_path}")
