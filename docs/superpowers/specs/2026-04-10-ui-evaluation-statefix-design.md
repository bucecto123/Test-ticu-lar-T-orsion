# Design: UI, Evaluation, and State Fix
**Date:** 2026-04-10  
**Project:** Subspace Aegis — Neuro-Symbolic IDS (COS30018 Option D)

---

## 1. Goals

Complete the three missing technical deliverables required by the assignment:

1. **UI** — Browser-accessible Gradio interface so users can interact with the IDS (satisfies assignment requirement for a UI + server-hosted demo)
2. **Formal evaluation** — Precision/recall/F1/confusion matrix against CIC-IDS2017 (satisfies benchmark evaluation requirement)
3. **State schema fix** — Align `agent_state.py` with fields actually used in `neuro_symbolic_graph.py` (fixes two silent bugs)

---

## 2. Files Changed / Created

### Modified
| File | Change |
|---|---|
| `dataCleaning/agent_state.py` | Add 5 missing fields; rename `final_verdict` → `final_diagnosis`; remove unused `agent_reasoning` |
| `dataCleaning/neuro_symbolic_graph.py` | Accept injected `llm` parameter in `build_neuro_symbolic_graph()`; remove module-level `ids_graph` singleton |
| `dataCleaning/main_stream_sim.py` | Fix `result.get('agent_reasoning')` → `result.get('final_diagnosis')` |

### Created
| File | Purpose |
|---|---|
| `dataCleaning/app.py` | Gradio web app — 3 tabs |
| `dataCleaning/evaluation.py` | Evaluation logic — metrics + confusion matrix |

---

## 3. Architecture

```
app.py (Gradio, port 7860)
├── Tab 1: CSV Batch     → run_batch(csv_path, n_flows, attack_filter, backend)
├── Tab 2: Manual Entry  → run_single(feature_json, backend)
└── Tab 3: Evaluation    → run_evaluation(csv_path, sample_size)

run_batch / run_single
└── build_neuro_symbolic_graph(llm=create_llm(backend))
    └── ids_graph.invoke(flow_state) → result dict

run_evaluation (evaluation.py)
├── streams N flows from CSV (chunked, no LLM)
├── runs continuous_sensor + discrete_sensor only
└── sklearn classification_report + confusion matrix heatmap (matplotlib)
```

**LLM injection:** `create_llm("groq")` returns `ChatGroq(model="llama-3.3-70b-versatile")`. `create_llm("ollama")` returns `ChatOllama(model="llama3.3")`. Graph is rebuilt per-call (stateless, cheap).

**Evaluation skips LLM:** Running cloud/local LLM on hundreds of flows is slow and costly. The binary anomaly signal comes from the two deterministic sensors; that is what we measure against ground truth labels.

---

## 4. UI Layout

### Tab 1 — CSV Batch
- `gr.File` — upload CSV (CIC-IDS2017 format)
- `gr.Dropdown` — attack filter: `All`, `BENIGN`, `DoS Hulk`, `DoS slowloris`, `DoS Slowhttptest`
- `gr.Slider` — number of flows to process (1–20, default 4)
- `gr.Radio` — LLM Backend: `Groq Cloud`, `Ollama Local`
- `gr.Button` — `Run IDS`
- `gr.Dataframe` — output columns: `flow_id`, `true_label`, `continuous_anomaly`, `monoid_state`, `algebraic_anomaly`, `final_diagnosis`

### Tab 2 — Manual Entry
- `gr.Code` (JSON) — pre-filled template with ~10 most diagnostic features and realistic default values
- `gr.Radio` — LLM Backend: `Groq Cloud`, `Ollama Local`
- `gr.Button` — `Analyze Flow`
- `gr.Textbox` — verdict card: thermodynamic state, applied forces, physical work description, threat mapping

### Tab 3 — Evaluation
- `gr.File` — upload CSV
- `gr.Slider` — sample size (100–2000, default 500)
- `gr.Button` — `Run Evaluation`
- `gr.Plot` — confusion matrix heatmap
- `gr.Dataframe` — per-class precision/recall/F1 table

---

## 5. State Schema Fix

### Fields to add to `NetworkFlowState`
```python
source_ip: Optional[str]          # read by orchestrator_node
geo_context: Optional[str]        # written by orchestrator_node, read by tuning_node + physicist_node
current_threshold: Optional[float] # written by tuning_node
spiking_features: Optional[List[str]]  # written by continuous_sensor_node, read by physicist_node
final_diagnosis: Optional[str]    # written by physicist_node
```

### Fields to remove / rename
- Remove `agent_reasoning` (never written by any node)
- Rename `final_verdict` → `final_diagnosis` (matches physicist node output)

### Bug fix in `main_stream_sim.py`
- Line 93: `result.get('agent_reasoning')` → `result.get('final_diagnosis')`

---

## 6. Manual Entry Feature Template

Pre-filled JSON with the 10 most diagnostic features:
```json
{
  "Flow Duration": 1000000,
  " Total Fwd Packets": 10,
  " Total Backward Packets": 8,
  " Total Length of Fwd Packets": 1200,
  " Fwd Packet Length Max": 300,
  " Bwd Packet Length Mean": 150,
  " Flow Bytes/s": 1200.0,
  " Flow Packets/s": 18.0,
  " SYN Flag Count": 1,
  " ACK Flag Count": 8,
  " PSH Flag Count": 2,
  " FIN Flag Count": 1,
  " RST Flag Count": 0,
  "Label": "BENIGN"
}
```
Missing features default to `0.0` in the continuous sensor.

---

## 7. Dependencies to Add

```
gradio
langchain-community   # for ChatOllama
```

Ollama itself must be installed separately by the user (`ollama pull llama3.3`).
