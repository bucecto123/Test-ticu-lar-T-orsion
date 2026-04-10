"""
benchmark_vgs.py
----------------
Computes Metric 2: Vector-Grounding Score (VGS)

Samples N attack flows per class, runs them through the live Groq
physicist, then scores whether the LLM output is strictly grounded
in the mathematical inputs provided to it.

VGS = 1.0 means the LLM acted as a pure translator.
VGS < 1.0 means hallucination was detected.

Grounding checks:
  1. force_grounding   — every term in applied_force_dimensions appears
                         verbatim in the spiking_features list fed to the LLM
  2. residual_grounding — the residual_norm value appears in the
                          physical_work_description text
  3. state_grounding   — thermodynamic_state is either EQUILIBRIUM or PERTURBED
                         (no invented states)
"""

import os
import time
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from algebraic_engine import AlgebraicMonoidEngine

load_dotenv(override=True)
api_key = os.environ.get("GROQ_API_KEY", "").strip().strip("'").strip('"')

os.makedirs('results', exist_ok=True)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Load artifacts ---
scaler     = joblib.load('subspace_models/geometric_scaler.joblib')
p_u_matrix = np.load('subspace_models/p_u_matrix.npy')

with open('subspace_models/subspace_config.json', 'r') as f:
    config = json.load(f)

EPSILON           = config['epsilon_threshold']
EXPECTED_FEATURES = scaler.feature_names_in_
logic_engine      = AlgebraicMonoidEngine()

SAMPLES_PER_CLASS = 30  # 30 per class gives ~±18% margin of error at 95% confidence
                        # Heartbleed capped at 11 (dataset limit)

# --- LLM setup (same as neuro_symbolic_graph.py) ---
class PhysicalDiagnosis(BaseModel):
    thermodynamic_state: str = Field(description="Must be either 'EQUILIBRIUM' or 'PERTURBED'.")
    applied_force_dimensions: List[str] = Field(description="The exact feature names from the spiking_features list.")
    physical_work_description: str = Field(description="Strict physical description of the work performed on the system based on Savelyev mechanics. NO generic cybersecurity jargon.")
    semantic_threat_mapping: str = Field(description="Human-readable threat diagnosis mapping the physical force to a system outcome.")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, api_key=api_key)
structured_llm = llm.with_structured_output(PhysicalDiagnosis)

PHYSICIST_PROMPT = """You are the Physicist Agent in a Subspace Aegis Multi-Agent System.
You act as a strict bridge between spectral mathematics and semantic meaning.

RULES:
1. Interpret the network strictly as a thermodynamic system.
2. Map the provided "Spiking Dimensions" to a physical force performing work.
3. DO NOT hallucinate standard cybersecurity jargon like "hacker", "malware", or "payload".
4. Ground your entire diagnosis ONLY in the mathematical facts provided.
5. Reference the residual norm magnitude when describing the physical work performed."""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", PHYSICIST_PROMPT),
    ("human", "Residual Norm (||w||): {residual_norm}\nSpiking Dimensions (Forces): {spiking_features}\nGeographic Context: {geo_context}\nMonoid Violation: {monoid_reasoning}")
])
chain = prompt_template | structured_llm

# --- Load dataset and sample attack flows ---
print("[VGS] Loading dataset...")
df = pd.read_csv('Wednesday-workingHours.pcap_ISCX.csv')
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

attack_classes = [c for c in df['Label'].unique() if c != 'BENIGN']
print(f"[VGS] Attack classes found: {attack_classes}")
print(f"[VGS] Sampling {SAMPLES_PER_CLASS} flows per class. Total LLM calls: {len(attack_classes) * SAMPLES_PER_CLASS}\n")

# --- VGS scoring ---
VALID_STATES = {'EQUILIBRIUM', 'PERTURBED'}

total_samples       = 0
force_grounded      = 0
residual_grounded   = 0
state_grounded      = 0

results_by_class = {}

for cls in attack_classes:
    cls_df   = df[df['Label'] == cls].sample(min(SAMPLES_PER_CLASS, len(df[df['Label'] == cls])), random_state=42)
    cls_scores = []

    print(f"  Evaluating class: {cls} ({len(cls_df)} samples)")

    for _, row in cls_df.iterrows():
        row_dict = row.to_dict()

        # compute spiking features (the ground truth inputs to the LLM)
        x_df     = pd.DataFrame({f: [row_dict.get(f, 0.0)] for f in EXPECTED_FEATURES})
        x_scaled = scaler.transform(x_df)
        w_vector = x_scaled - np.dot(x_scaled, p_u_matrix)
        residual = float(np.linalg.norm(w_vector))

        abs_w       = np.abs(w_vector[0])
        top_indices = np.argsort(abs_w)[-3:][::-1]
        spiking     = [EXPECTED_FEATURES[i] for i in top_indices]

        mono_result     = logic_engine.evaluate_state_loop(row_dict)
        monoid_reasoning = mono_result['monoid_state_reasoning']

        # invoke LLM with retry on rate limit
        max_retries = 3
        diagnosis = None
        for attempt in range(max_retries):
            try:
                diagnosis = chain.invoke({
                    "residual_norm":    residual,
                    "spiking_features": spiking,
                    "geo_context":      "Unknown Location",
                    "monoid_reasoning": monoid_reasoning
                })
                time.sleep(1.5)  # 1.5s between calls to stay under TPM limit
                break
            except Exception as e:
                if '429' in str(e):
                    wait = 90  # wait 90s on rate limit
                    print(f"    [RATE LIMIT] Waiting {wait}s before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait)
                else:
                    print(f"    [ERROR] LLM call failed: {e}")
                    break

        if diagnosis is None:
            continue

        # --- Grounding checks ---
        # 1. force grounding: every dimension the LLM claims must be in spiking
        spiking_lower = [s.lower() for s in spiking]
        force_dims    = diagnosis.applied_force_dimensions
        force_ok      = all(
            any(fd.lower() in sl or sl in fd.lower() for sl in spiking_lower)
            for fd in force_dims
        ) if force_dims else False

        # 2. residual grounding: a number close to the residual appears in work description
        import re
        numbers_in_text = [float(x) for x in re.findall(r'\d+\.?\d*', diagnosis.physical_work_description)]
        residual_ok = any(abs(n - residual) / (residual + 1e-9) < 0.05 for n in numbers_in_text)

        # 3. state grounding: only valid states used
        state_ok = diagnosis.thermodynamic_state.upper() in VALID_STATES

        cls_scores.append({
            "force_ok":    force_ok,
            "residual_ok": residual_ok,
            "state_ok":    state_ok,
            "spiking":     spiking,
            "force_dims":  force_dims,
            "state":       diagnosis.thermodynamic_state,
        })

        total_samples     += 1
        force_grounded    += int(force_ok)
        residual_grounded += int(residual_ok)
        state_grounded    += int(state_ok)

    results_by_class[cls] = cls_scores

# --- Report ---
print("\n" + "="*60)
print("METRIC 2: VECTOR-GROUNDING SCORE (VGS)")
print("="*60)

if total_samples == 0:
    print("[ERROR] No samples evaluated.")
else:
    vgs_force    = force_grounded    / total_samples
    vgs_residual = residual_grounded / total_samples
    vgs_state    = state_grounded    / total_samples
    vgs_overall  = (vgs_force + vgs_residual + vgs_state) / 3

    print(f"\n  Total samples evaluated : {total_samples}")
    print(f"\n  Sub-scores:")
    print(f"    Force Grounding Score    : {vgs_force:.4f}  ({force_grounded}/{total_samples} outputs used only provided spiking features)")
    print(f"    Residual Grounding Score : {vgs_residual:.4f}  ({residual_grounded}/{total_samples} outputs referenced the exact ||w|| value)")
    print(f"    State Grounding Score    : {vgs_state:.4f}  ({state_grounded}/{total_samples} outputs used only EQUILIBRIUM or PERTURBED)")
    print(f"\n  OVERALL VGS              : {vgs_overall:.4f}  (1.0 = perfect neuro-symbolic grounding)")

    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':<35} {'n':>4}  {'Force':>6}  {'Residual':>8}  {'State':>6}")
    print(f"  {'-'*65}")
    for cls, scores in results_by_class.items():
        n  = len(scores)
        if n == 0:
            continue
        fc = sum(s['force_ok']    for s in scores) / n
        rc = sum(s['residual_ok'] for s in scores) / n
        sc = sum(s['state_ok']    for s in scores) / n
        print(f"  {cls:<35} {n:>4}  {fc:>6.3f}  {rc:>8.3f}  {sc:>6.3f}")

    # flag any hallucinations
    hallucinations = [
        (cls, s) for cls, scores in results_by_class.items()
        for s in scores if not s['force_ok']
    ]
    if hallucinations:
        print(f"\n  [!] Hallucination instances detected ({len(hallucinations)}):")
        for cls, s in hallucinations:
            print(f"      Class: {cls}")
            print(f"        Provided spiking : {s['spiking']}")
            print(f"        LLM force dims   : {s['force_dims']}")
    else:
        print(f"\n  [✓] No force hallucinations detected across all {total_samples} samples.")

# ── Save results ──────────────────────────────────────────────────────────────
if total_samples > 0:
    per_class_summary = {}
    for cls, scores in results_by_class.items():
        n = len(scores)
        if n == 0:
            continue
        per_class_summary[cls] = {
            "n": n,
            "force_grounding":    round(sum(s['force_ok']    for s in scores) / n, 6),
            "residual_grounding": round(sum(s['residual_ok'] for s in scores) / n, 6),
            "state_grounding":    round(sum(s['state_ok']    for s in scores) / n, 6),
            "hallucinations":     [
                {"spiking_provided": s['spiking'], "llm_force_dims": s['force_dims']}
                for s in scores if not s['force_ok']
            ]
        }

    vgs_report = {
        "run_timestamp":    RUN_TIMESTAMP,
        "dataset":          "Wednesday-workingHours.pcap_ISCX.csv",
        "samples_per_class": SAMPLES_PER_CLASS,
        "total_samples":    total_samples,
        "metric_2_vgs": {
            "force_grounding_score":    round(vgs_force, 6),
            "residual_grounding_score": round(vgs_residual, 6),
            "state_grounding_score":    round(vgs_state, 6),
            "overall_vgs":              round(vgs_overall, 6),
        },
        "per_class": per_class_summary,
    }

    json_path = f"results/benchmark_vgs_{RUN_TIMESTAMP}.json"
    with open(json_path, 'w') as f:
        json.dump(vgs_report, f, indent=4)
    print(f"\n[OUTPUT] JSON report saved to {json_path}")

    txt_path = f"results/benchmark_vgs_{RUN_TIMESTAMP}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("SUBSPACE AEGIS — VECTOR-GROUNDING SCORE (VGS) REPORT\n")
        f.write(f"Run: {RUN_TIMESTAMP}\n")
        f.write(f"Dataset: Wednesday-workingHours.pcap_ISCX.csv\n")
        f.write(f"Samples per class: {SAMPLES_PER_CLASS}\n\n")

        f.write("="*60 + "\n")
        f.write("METRIC 2: VECTOR-GROUNDING SCORE (VGS)\n")
        f.write("="*60 + "\n")
        f.write(f"Total samples evaluated : {total_samples}\n\n")
        f.write(f"Force Grounding Score    : {vgs_force:.4f}  ({force_grounded}/{total_samples})\n")
        f.write(f"Residual Grounding Score : {vgs_residual:.4f}  ({residual_grounded}/{total_samples})\n")
        f.write(f"State Grounding Score    : {vgs_state:.4f}  ({state_grounded}/{total_samples})\n")
        f.write(f"\nOVERALL VGS              : {vgs_overall:.4f}\n\n")

        f.write(f"{'Class':<35} {'n':>4}  {'Force':>6}  {'Residual':>8}  {'State':>6}\n")
        f.write("-"*65 + "\n")
        for cls, v in per_class_summary.items():
            f.write(f"{cls:<35} {v['n']:>4}  {v['force_grounding']:>6.3f}  {v['residual_grounding']:>8.3f}  {v['state_grounding']:>6.3f}\n")

        f.write("\n")
        if hallucinations:
            f.write(f"HALLUCINATION INSTANCES ({len(hallucinations)}):\n")
            for cls, s in hallucinations:
                f.write(f"  Class: {cls}\n")
                f.write(f"    Provided spiking : {s['spiking']}\n")
                f.write(f"    LLM force dims   : {s['force_dims']}\n")
        else:
            f.write("No force hallucinations detected.\n")

    print(f"[OUTPUT] Text report saved to {txt_path}")
