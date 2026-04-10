"""
benchmark_vectorized.py
-----------------------
Computes Metric 1 (Spectral SNR) and Metric 3 (Thermodynamic Efficiency)
across the full Wednesday dataset using vectorized numpy operations.

Metric 1 — Spectral SNR
    SNR_class = mean(||w|| attacks) / mean(||w|| benign)
    Proves the SVD filter creates a measurable geometric gap between
    equilibrium and perturbation. Higher = more disjoint distributions.

Metric 3 — Thermodynamic Efficiency
    efficiency_ratio = total_flows / flows_routed_to_LLM
    Quantifies the computational energy saved by the router node dropping
    benign traffic before it reaches the Groq physicist.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from algebraic_engine import AlgebraicMonoidEngine

os.makedirs('results', exist_ok=True)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Load persisted math artifacts ---
scaler     = joblib.load('subspace_models/geometric_scaler.joblib')
p_u_matrix = np.load('subspace_models/p_u_matrix.npy')

with open('subspace_models/subspace_config.json', 'r') as f:
    config = json.load(f)

EPSILON           = config['epsilon_threshold']
EXPECTED_FEATURES = scaler.feature_names_in_
logic_engine      = AlgebraicMonoidEngine()

# estimated tokens per LLM call (system prompt + human message + structured output)
TOKENS_PER_LLM_CALL = 350

print(f"[BENCHMARK] Epsilon: {EPSILON:.4f}")
print("[BENCHMARK] Loading full Wednesday dataset...")

df = pd.read_csv('Wednesday-workingHours.pcap_ISCX.csv')
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

labels = df['Label']
print(f"[BENCHMARK] {len(df):,} flows loaded.\n")

# ── Metric 1: Spectral SNR ─────────────────────────────────────────────────────
print("="*60)
print("METRIC 1: SPECTRAL SIGNAL-TO-NOISE RATIO (SNR)")
print("="*60)

df_geo   = df.reindex(columns=EXPECTED_FEATURES, fill_value=0.0)
x_scaled = scaler.transform(df_geo)
w_matrix = x_scaled - np.dot(x_scaled, p_u_matrix)
residuals = np.linalg.norm(w_matrix, axis=1)

benign_mask      = labels == 'BENIGN'
mean_benign_res  = residuals[benign_mask].mean()
std_benign_res   = residuals[benign_mask].std()

print(f"\n  Benign baseline:")
print(f"    mean ||w||  = {mean_benign_res:.4f}")
print(f"    std  ||w||  = {std_benign_res:.4f}")
print(f"    epsilon (ε) = {EPSILON:.4f}")
print(f"\n  Per-class SNR  (mean_attack / mean_benign):")
print(f"  {'Class':<35} {'n':>7}  {'mean ||w||':>10}  {'SNR':>8}  {'Separation'}")
print(f"  {'-'*75}")

attack_classes = [c for c in sorted(labels.unique()) if c != 'BENIGN']
for cls in attack_classes:
    mask         = labels == cls
    mean_atk_res = residuals[mask].mean()
    snr          = mean_atk_res / mean_benign_res
    # cohen's d style separation
    pooled_std   = np.sqrt((std_benign_res**2 + residuals[mask].std()**2) / 2)
    separation   = (mean_atk_res - mean_benign_res) / pooled_std if pooled_std > 0 else 0
    print(f"  {cls:<35} {mask.sum():>7,}  {mean_atk_res:>10.4f}  {snr:>8.3f}x  d={separation:.3f}")

# overall SNR across all attacks
attack_mask     = ~benign_mask
mean_all_attack = residuals[attack_mask].mean()
overall_snr     = mean_all_attack / mean_benign_res
print(f"\n  Overall SNR (all attacks vs benign): {overall_snr:.3f}x")

# ── Metric 3: Thermodynamic Efficiency ────────────────────────────────────────
print("\n" + "="*60)
print("METRIC 3: THERMODYNAMIC EFFICIENCY (COMPUTATIONAL ENERGY)")
print("="*60)

# vectorized monoid
syn = df.get('SYN Flag Count', pd.Series(0, index=df.index))
fin = df.get('FIN Flag Count', pd.Series(0, index=df.index))
rst = df.get('RST Flag Count', pd.Series(0, index=df.index))
psh = df.get('PSH Flag Count', pd.Series(0, index=df.index))
ack = df.get('ACK Flag Count', pd.Series(0, index=df.index))
urg = df.get('URG Flag Count', pd.Series(0, index=df.index))
dur = df.get('Flow Duration', pd.Series(0, index=df.index))

total_flags  = syn + fin + rst + psh + ack + urg
total_pkts = df.get('Total Fwd Packets', pd.Series(0, index=df.index)) + df.get('Total Backward Packets', pd.Series(0, index=df.index))

mono_anomaly = (
    # rule1 null scan disabled — no Protocol column in CICFlowMeter output,
    # so zero-flag flows include UDP/ICMP which are indistinguishable from TCP null scans
    # ((total_flags == 0) & (dur > 0)) |
    ((fin > 0) & (psh > 0) & (urg > 0) & (syn == 0) & (ack == 0)) |     # xmas scan
    ((syn > 20) & (ack == 0)) |                                           # syn flood
    ((syn > 0) & (fin == 0) & (rst == 0) & (dur > 30_000_000)) |         # orphaned loop
    ((psh > 50) & (fin == 0) & (rst == 0))                               # psh exhaustion
)

svd_anomaly = residuals > EPSILON
routed_to_llm = (svd_anomaly | mono_anomaly).sum()
total_flows   = len(df)
dropped       = total_flows - routed_to_llm

efficiency_ratio     = total_flows / routed_to_llm if routed_to_llm > 0 else float('inf')
tokens_actual        = routed_to_llm * TOKENS_PER_LLM_CALL
tokens_naive         = total_flows   * TOKENS_PER_LLM_CALL
token_savings_pct    = 100 * (1 - tokens_actual / tokens_naive)

print(f"\n  Total flows in dataset      : {total_flows:>10,}")
print(f"  Flows dropped at router     : {dropped:>10,}  ({100*dropped/total_flows:.2f}% of traffic)")
print(f"  Flows routed to LLM         : {routed_to_llm:>10,}  ({100*routed_to_llm/total_flows:.2f}% of traffic)")
print(f"\n  Efficiency ratio            : {efficiency_ratio:>10.2f}x")
print(f"  (1 LLM call per {efficiency_ratio:.1f} flows on average)")
print(f"\n  Token cost — Subspace Aegis : {tokens_actual:>10,} tokens")
print(f"  Token cost — Naive Pure LLM : {tokens_naive:>10,} tokens")
print(f"  Token savings               : {token_savings_pct:>9.2f}%")
print(f"\n  At $0.0001 per 1k tokens:")
cost_actual = tokens_actual / 1000 * 0.0001
cost_naive  = tokens_naive  / 1000 * 0.0001
print(f"    Subspace Aegis cost       : ${cost_actual:.4f}")
print(f"    Naive Pure LLM cost       : ${cost_naive:.4f}")
print(f"    Cost reduction            : {cost_naive/cost_actual:.1f}x cheaper" if cost_actual > 0 else "    N/A")

# ── Benign false positive rate (bonus) ────────────────────────────────────────
print("\n" + "="*60)
print("BONUS: FALSE POSITIVE RATE ON BENIGN TRAFFIC")
print("="*60)
benign_flagged = (svd_anomaly | mono_anomaly)[benign_mask].sum()
benign_total   = benign_mask.sum()
fpr            = benign_flagged / benign_total
print(f"\n  Benign flows incorrectly flagged: {benign_flagged:,} / {benign_total:,}  (FPR = {fpr:.4f})")

# ── Save results ──────────────────────────────────────────────────────────────
snr_per_class = {}
for cls in attack_classes:
    mask         = labels == cls
    mean_atk_res = residuals[mask].mean()
    pooled_std   = np.sqrt((std_benign_res**2 + residuals[mask].std()**2) / 2)
    snr_per_class[cls] = {
        "n":              int(mask.sum()),
        "mean_residual":  round(float(mean_atk_res), 6),
        "snr":            round(float(mean_atk_res / mean_benign_res), 6),
        "cohens_d":       round(float((mean_atk_res - mean_benign_res) / pooled_std) if pooled_std > 0 else 0, 6),
    }

report = {
    "run_timestamp": RUN_TIMESTAMP,
    "dataset": "Wednesday-workingHours.pcap_ISCX.csv",
    "epsilon": round(float(EPSILON), 6),
    "metric_1_spectral_snr": {
        "benign_baseline": {
            "n":             int(benign_mask.sum()),
            "mean_residual": round(float(mean_benign_res), 6),
            "std_residual":  round(float(std_benign_res), 6),
        },
        "per_class": snr_per_class,
        "overall_snr": round(float(overall_snr), 6),
    },
    "metric_3_thermodynamic_efficiency": {
        "total_flows":          int(total_flows),
        "flows_dropped":        int(dropped),
        "flows_routed_to_llm":  int(routed_to_llm),
        "pct_dropped":          round(100 * dropped / total_flows, 4),
        "efficiency_ratio":     round(float(efficiency_ratio), 4),
        "tokens_subspace_aegis": int(tokens_actual),
        "tokens_naive_llm":      int(tokens_naive),
        "token_savings_pct":     round(float(token_savings_pct), 4),
        "cost_subspace_aegis_usd": round(cost_actual, 6),
        "cost_naive_llm_usd":      round(cost_naive, 6),
    },
    "bonus_false_positive_rate": {
        "benign_total":   int(benign_total),
        "benign_flagged": int(benign_flagged),
        "fpr":            round(float(fpr), 6),
    }
}

json_path = f"results/benchmark_vectorized_{RUN_TIMESTAMP}.json"
with open(json_path, 'w') as f:
    json.dump(report, f, indent=4)
print(f"\n[OUTPUT] JSON report saved to {json_path}")

# plain text report
txt_path = f"results/benchmark_vectorized_{RUN_TIMESTAMP}.txt"
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write(f"SUBSPACE AEGIS — VECTORIZED BENCHMARK REPORT\n")
    f.write(f"Run: {RUN_TIMESTAMP}\n")
    f.write(f"Dataset: Wednesday-workingHours.pcap_ISCX.csv\n")
    f.write(f"Epsilon: {EPSILON:.4f}\n\n")

    f.write("="*60 + "\n")
    f.write("METRIC 1: SPECTRAL SNR\n")
    f.write("="*60 + "\n")
    f.write(f"Benign baseline — mean ||w||: {mean_benign_res:.4f}  std: {std_benign_res:.4f}\n\n")
    f.write(f"{'Class':<35} {'n':>7}  {'mean ||w||':>10}  {'SNR':>8}  {'Cohen d':>8}\n")
    f.write("-"*75 + "\n")
    for cls, v in snr_per_class.items():
        f.write(f"{cls:<35} {v['n']:>7,}  {v['mean_residual']:>10.4f}  {v['snr']:>8.3f}x  {v['cohens_d']:>8.3f}\n")
    f.write(f"\nOverall SNR (all attacks vs benign): {overall_snr:.3f}x\n\n")

    f.write("="*60 + "\n")
    f.write("METRIC 3: THERMODYNAMIC EFFICIENCY\n")
    f.write("="*60 + "\n")
    f.write(f"Total flows          : {total_flows:,}\n")
    f.write(f"Flows dropped        : {dropped:,}  ({100*dropped/total_flows:.2f}%)\n")
    f.write(f"Flows to LLM         : {routed_to_llm:,}  ({100*routed_to_llm/total_flows:.2f}%)\n")
    f.write(f"Efficiency ratio     : {efficiency_ratio:.2f}x\n")
    f.write(f"Token savings        : {token_savings_pct:.2f}%\n")
    f.write(f"Cost — Subspace Aegis: ${cost_actual:.4f}\n")
    f.write(f"Cost — Naive LLM     : ${cost_naive:.4f}\n\n")

    f.write("="*60 + "\n")
    f.write("BONUS: FALSE POSITIVE RATE\n")
    f.write("="*60 + "\n")
    f.write(f"Benign flagged: {benign_flagged:,} / {benign_total:,}  FPR = {fpr:.4f}\n")

print(f"[OUTPUT] Text report saved to {txt_path}")
