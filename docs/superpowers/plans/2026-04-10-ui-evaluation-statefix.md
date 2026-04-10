# UI, Evaluation, and State Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Gradio web UI (3 tabs: CSV batch, manual entry, evaluation), formal evaluation metrics, and fix the state schema mismatch that causes silent bugs.

**Architecture:** Gradio app (`app.py`) directly imports and calls `build_neuro_symbolic_graph(llm)` with an injected LLM (Groq or Ollama). Evaluation (`evaluation.py`) runs only the two deterministic sensors (no LLM) against ground truth labels to produce precision/recall/F1 and a confusion matrix. State schema (`agent_state.py`) is fixed to match what the graph nodes actually read/write.

**Tech Stack:** Python 3.x, Gradio, LangGraph, LangChain, langchain-community (ChatOllama), scikit-learn, matplotlib, numpy, pandas, joblib.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `dataCleaning/agent_state.py` | Modify | Add 5 missing fields; rename `final_verdict` → `final_diagnosis`; remove `agent_reasoning` |
| `dataCleaning/neuro_symbolic_graph.py` | Modify | Add `create_llm(backend)` factory; move `physicist_node` inside `build_neuro_symbolic_graph` as closure; remove module-level `llm`, `structured_llm`, `ids_graph` |
| `dataCleaning/main_stream_sim.py` | Modify | Fix import of `ids_graph`; fix `agent_reasoning` → `final_diagnosis` print |
| `dataCleaning/evaluation.py` | Create | Deterministic-only evaluation pipeline; returns confusion matrix fig + metrics DataFrame |
| `dataCleaning/app.py` | Create | Gradio app with 3 tabs wired to graph and evaluation |
| `dataCleaning/tests/test_state.py` | Create | Verify NetworkFlowState has all required fields |
| `dataCleaning/tests/test_evaluation.py` | Create | Verify `_predict_flow` returns correct types on synthetic data |

---

## Task 1: Fix State Schema

**Files:**
- Modify: `dataCleaning/agent_state.py`
- Modify: `dataCleaning/main_stream_sim.py:93`
- Create: `dataCleaning/tests/test_state.py`

- [ ] **Step 1.1: Create the test file**

Create `dataCleaning/tests/__init__.py` (empty) and `dataCleaning/tests/test_state.py`:

```python
# dataCleaning/tests/test_state.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent_state import NetworkFlowState
import typing

def test_all_required_fields_present():
    hints = typing.get_type_hints(NetworkFlowState)
    required = [
        "flow_id", "true_label",
        "geometric_features", "algebraic_features",
        "thermodynamic_residual", "continuous_anomaly",
        "spiking_features",
        "current_monoid_state", "algebraic_anomaly", "monoid_reasoning",
        "source_ip", "geo_context", "current_threshold",
        "final_diagnosis",
    ]
    for field in required:
        assert field in hints, f"Missing field: {field}"

def test_no_stale_fields():
    hints = typing.get_type_hints(NetworkFlowState)
    stale = ["agent_reasoning", "final_verdict"]
    for field in stale:
        assert field not in hints, f"Stale field still present: {field}"
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
cd dataCleaning
python -m pytest tests/test_state.py -v
```

Expected: FAIL — `Missing field: spiking_features` (and others)

- [ ] **Step 1.3: Rewrite `agent_state.py`**

Replace the entire file with:

```python
from typing import TypedDict, Dict, Any, Optional, List

class NetworkFlowState(TypedDict):
    """
    The shared memory manifold for the LangGraph neuro-symbolic agents.
    Dictates exactly what information can be passed between the deterministic
    sensors and the cognitive orchestrator.
    """
    flow_id: str
    true_label: str

    # Raw input data routed from the temporal stream
    geometric_features: Dict[str, Any]
    algebraic_features: Dict[str, Any]

    # Output from Sensor 1: The Mathematician (Continuous SVD)
    thermodynamic_residual: Optional[float]
    spiking_features: Optional[List[str]]
    continuous_anomaly: Optional[bool]

    # Output from Sensor 2: The Logician (Discrete Monoid)
    current_monoid_state: Optional[str]
    algebraic_anomaly: Optional[bool]
    monoid_reasoning: Optional[str]

    # Output from Orchestrator / Tuning Agent
    source_ip: Optional[str]
    geo_context: Optional[str]
    current_threshold: Optional[float]

    # Final output from the Physicist (LLM)
    final_diagnosis: Optional[str]
```

- [ ] **Step 1.4: Run test to verify it passes**

```bash
python -m pytest tests/test_state.py -v
```

Expected: PASS — 2 tests

- [ ] **Step 1.5: Fix `main_stream_sim.py` line 93**

Change:
```python
print(result.get('agent_reasoning'))
```
To:
```python
print(result.get('final_diagnosis', 'No LLM diagnosis (normal traffic).'))
```

- [ ] **Step 1.6: Commit**

```bash
git add dataCleaning/agent_state.py dataCleaning/main_stream_sim.py dataCleaning/tests/
git commit -m "fix: align NetworkFlowState schema with actual node I/O"
```

---

## Task 2: Refactor `neuro_symbolic_graph.py` for LLM Injection

**Files:**
- Modify: `dataCleaning/neuro_symbolic_graph.py`
- Modify: `dataCleaning/main_stream_sim.py` (import section)

The goal is to remove the module-level `llm`, `structured_llm`, and `ids_graph` singletons so the graph can be built with any LLM backend (Groq or Ollama) without triggering API connections on import.

- [ ] **Step 2.1: Replace `neuro_symbolic_graph.py` with the refactored version**

Replace the entire file with:

```python
import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv(override=True)
_api_key = os.environ.get("GROQ_API_KEY", "")
if _api_key:
    _api_key = _api_key.strip().strip("'").strip('"')

from agent_state import NetworkFlowState
from algebraic_engine import AlgebraicMonoidEngine

# --- Math artifacts (loaded once at module import, not per-call) ---
_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'subspace_models')
p_u_matrix = np.load(os.path.join(_MODELS_DIR, 'p_u_matrix.npy'))
scaler = joblib.load(os.path.join(_MODELS_DIR, 'geometric_scaler.joblib'))
logic_engine = AlgebraicMonoidEngine()

BASE_THRESHOLD = 1.40  # Temporarily lowered to catch Wednesday attacks

# --- Structured output schema ---
class PhysicalDiagnosis(BaseModel):
    thermodynamic_state: str = Field(
        description="Must be either 'EQUILIBRIUM' or 'PERTURBED'."
    )
    applied_force_dimensions: List[str] = Field(
        description="The exact feature names from the spiking_features list."
    )
    physical_work_description: str = Field(
        description="Strict physical description of the work performed on the system based on Savelyev mechanics. NO generic cybersecurity jargon."
    )
    semantic_threat_mapping: str = Field(
        description="Human-readable threat diagnosis mapping the physical force to a system outcome."
    )

# --- LLM factory ---
def create_llm(backend: str = "groq"):
    """
    Returns the appropriate LangChain chat model.
    backend: "groq" (cloud) or "ollama" (local).
    """
    if backend == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model="llama3.3", temperature=0.0)
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        api_key=_api_key
    )

# --- Deterministic sensor nodes (no LLM dependency) ---
def continuous_sensor_node(state: NetworkFlowState) -> Dict[str, Any]:
    """The Mathematician: Projects onto P_U and calculates residual w."""
    try:
        expected_features = scaler.feature_names_in_
    except AttributeError:
        return {"thermodynamic_residual": 0.0, "spiking_features": []}

    raw_continuous = {}
    clean_packet_features = {k.strip(): v for k, v in state["algebraic_features"].items()}

    for feat in expected_features:
        val = clean_packet_features.get(feat.strip(), 0.0)
        raw_continuous[feat] = [val]

    x_df = pd.DataFrame(raw_continuous)
    x_scaled = scaler.transform(x_df)
    x_proj = np.dot(x_scaled, np.dot(p_u_matrix, p_u_matrix.T))
    w_vector = x_scaled - x_proj
    residual = float(np.linalg.norm(w_vector))

    abs_w = np.abs(w_vector[0])
    top_indices = np.argsort(abs_w)[-3:][::-1]
    spiking_features = [expected_features[i] for i in top_indices]

    print(f"\n[MATHEMATICIAN] Residual: {residual:.4f} | Spiking: {spiking_features}")

    return {
        "thermodynamic_residual": residual,
        "spiking_features": spiking_features
    }

def discrete_sensor_node(state: NetworkFlowState) -> Dict[str, Any]:
    """The Logician: Evaluates TCP monoid rules."""
    raw_features = state.get("algebraic_features", {})
    result = logic_engine.evaluate_state_loop(raw_features)

    print(f"[LOGICIAN] Monoid State: {result['monoid_classification']}")

    return {
        "algebraic_anomaly": result["algebraic_anomaly"],
        "current_monoid_state": result["monoid_classification"],
        "monoid_reasoning": result["monoid_state_reasoning"]
    }

MOCK_GEO_DATABASE = {
    "192.168.1.100": "Corporate HQ",
    "10.0.0.50": "Airport Public WiFi"
}

def orchestrator_node(state: NetworkFlowState) -> Dict[str, Any]:
    """Fetches environmental context."""
    ip = state.get("source_ip", "10.0.0.50")
    context = MOCK_GEO_DATABASE.get(ip, "Unknown Location")
    print(f"\n[ORCHESTRATOR] IP {ip} -> Environment: {context}")
    return {"geo_context": context}

def tuning_node(state: NetworkFlowState) -> Dict[str, Any]:
    """Dynamically scales the thermodynamic threshold."""
    context = state.get("geo_context", "")
    current_eps = BASE_THRESHOLD

    if context == "Airport Public WiFi":
        current_eps *= 1.5
        print(f"[TUNING AGENT] Chaotic environment. Expanding epsilon to {current_eps:.4f}")
    else:
        print(f"[TUNING AGENT] Stable environment. Epsilon remains {current_eps:.4f}")

    residual = state.get("thermodynamic_residual", 0.0)
    continuous_anomaly = bool(residual > current_eps)

    return {
        "current_threshold": current_eps,
        "continuous_anomaly": continuous_anomaly
    }

def route_after_tuning(state: NetworkFlowState) -> str:
    """Conserves energy by skipping the LLM if traffic is benign."""
    if state.get("continuous_anomaly") or state.get("algebraic_anomaly"):
        print("[ROUTER] Perturbation detected. Routing to Physicist...")
        return "physicist"
    print("[ROUTER] System in equilibrium. Halting execution.")
    return "end"

PHYSICIST_PROMPT = """You are the Physicist Agent in a Subspace Aegis Multi-Agent System.
You act as a strict bridge between spectral mathematics and semantic meaning.

RULES:
1. Interpret the network strictly as a thermodynamic system.
2. Map the provided "Spiking Dimensions" to a physical force performing work.
3. DO NOT hallucinate standard cybersecurity jargon like "hacker", "malware", or "payload".
4. Ground your entire diagnosis ONLY in the mathematical facts provided."""

_prompt_template = ChatPromptTemplate.from_messages([
    ("system", PHYSICIST_PROMPT),
    ("human", "Residual Norm (||w||): {residual_norm}\nSpiking Dimensions (Forces): {spiking_features}\nGeographic Context: {geo_context}\nMonoid Violation: {monoid_reasoning}")
])

def build_neuro_symbolic_graph(llm=None):
    """
    Compiles the LangGraph neuro-symbolic IDS.
    llm: a LangChain chat model. If None, defaults to Groq.
    """
    if llm is None:
        llm = create_llm("groq")

    structured_llm = llm.with_structured_output(PhysicalDiagnosis)

    def physicist_node(state: NetworkFlowState) -> Dict[str, Any]:
        """Translates vector math into Savelyev physics constraints."""
        print("[*] Invoking Physicist Agent (Neuro-Symbolic Translation)...")

        chain = _prompt_template | structured_llm
        diagnosis = chain.invoke({
            "residual_norm": state.get("thermodynamic_residual", 0.0),
            "spiking_features": state.get("spiking_features", []),
            "geo_context": state.get("geo_context", "Unknown"),
            "monoid_reasoning": state.get("monoid_reasoning", "None")
        })

        final_text = (
            f"State: {diagnosis.thermodynamic_state} | "
            f"Work: {diagnosis.physical_work_description} | "
            f"Threat: {diagnosis.semantic_threat_mapping}"
        )
        print(f"\n=== FINAL SYSTEM VERDICT ===\n{final_text}\n============================\n")
        return {"final_diagnosis": final_text}

    workflow = StateGraph(NetworkFlowState)
    workflow.add_node("mathematician", continuous_sensor_node)
    workflow.add_node("logician", discrete_sensor_node)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("tuning_agent", tuning_node)
    workflow.add_node("physicist", physicist_node)

    workflow.add_edge(START, "mathematician")
    workflow.add_edge(START, "logician")
    workflow.add_edge("mathematician", "orchestrator")
    workflow.add_edge("logician", "orchestrator")
    workflow.add_edge("orchestrator", "tuning_agent")
    workflow.add_conditional_edges(
        "tuning_agent",
        route_after_tuning,
        {"physicist": "physicist", "end": END}
    )
    workflow.add_edge("physicist", END)

    return workflow.compile()
```

- [ ] **Step 2.2: Update the import in `main_stream_sim.py`**

Replace line 8:
```python
from neuro_symbolic_graph import ids_graph
```
With:
```python
from neuro_symbolic_graph import build_neuro_symbolic_graph, create_llm
ids_graph = build_neuro_symbolic_graph(create_llm("groq"))
```

- [ ] **Step 2.3: Verify the stream simulation still works (smoke test)**

```bash
cd dataCleaning
python -c "from neuro_symbolic_graph import build_neuro_symbolic_graph, create_llm; g = build_neuro_symbolic_graph(create_llm('groq')); print('Graph compiled OK:', g)"
```

Expected: prints `Graph compiled OK: <CompiledStateGraph ...>` with no errors.

- [ ] **Step 2.4: Commit**

```bash
git add dataCleaning/neuro_symbolic_graph.py dataCleaning/main_stream_sim.py
git commit -m "refactor: inject LLM into build_neuro_symbolic_graph, add create_llm factory"
```

---

## Task 3: Write `evaluation.py`

**Files:**
- Create: `dataCleaning/evaluation.py`
- Create: `dataCleaning/tests/test_evaluation.py`

- [ ] **Step 3.1: Write the failing test**

```python
# dataCleaning/tests/test_evaluation.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import _predict_flow

def test_predict_flow_returns_two_bools():
    # A synthetic benign-ish flow (all zeros → low residual, no monoid violation)
    row = {feat: 0.0 for feat in [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
        "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
        "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
        "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
        "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
        "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
        "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags",
        "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
        "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
        "Max Packet Length", "Packet Length Mean", "Packet Length Std",
        "Packet Length Variance", "CWE Flag Count", "ECE Flag Count",
        "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
        "Avg Bwd Segment Size", "Fwd Header Length.1", "Fwd Avg Bytes/Bulk",
        "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
        "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
        "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
        "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
        "min_seg_size_forward", "Active Mean", "Active Std", "Active Max",
        "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
        "SYN Flag Count", "FIN Flag Count", "RST Flag Count",
        "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    ]}
    cont_anom, alg_anom = _predict_flow(row)
    assert isinstance(cont_anom, bool), "continuous_anomaly must be bool"
    assert isinstance(alg_anom, bool), "algebraic_anomaly must be bool"

def test_xmas_scan_triggers_algebraic_anomaly():
    row = {
        "FIN Flag Count": 1, "PSH Flag Count": 1, "URG Flag Count": 1,
        "SYN Flag Count": 0, "ACK Flag Count": 0, "RST Flag Count": 0,
        "Flow Duration": 1000,
    }
    _, alg_anom = _predict_flow(row)
    assert alg_anom is True, "Xmas scan must trigger algebraic anomaly"
```

- [ ] **Step 3.2: Run test to verify it fails**

```bash
cd dataCleaning
python -m pytest tests/test_evaluation.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evaluation'`

- [ ] **Step 3.3: Create `evaluation.py`**

```python
# dataCleaning/evaluation.py
import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for Gradio
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from algebraic_engine import AlgebraicMonoidEngine
from neuro_symbolic_graph import BASE_THRESHOLD

# Load math artifacts once at module import
_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'subspace_models')
_p_u = np.load(os.path.join(_MODELS_DIR, 'p_u_matrix.npy'))
_scaler = joblib.load(os.path.join(_MODELS_DIR, 'geometric_scaler.joblib'))
_engine = AlgebraicMonoidEngine()


def _predict_flow(row_dict: dict) -> tuple:
    """
    Runs both deterministic sensors on a single flow dict.
    Returns (continuous_anomaly: bool, algebraic_anomaly: bool).
    """
    # --- Continuous sensor ---
    expected_features = _scaler.feature_names_in_
    raw = {feat: [row_dict.get(feat, 0.0)] for feat in expected_features}
    x_df = pd.DataFrame(raw)
    x_scaled = _scaler.transform(x_df)
    x_proj = np.dot(x_scaled, np.dot(_p_u, _p_u.T))
    w = x_scaled - x_proj
    residual = float(np.linalg.norm(w))
    continuous_anomaly = bool(residual > BASE_THRESHOLD)

    # --- Discrete sensor ---
    result = _engine.evaluate_state_loop(row_dict)
    algebraic_anomaly = bool(result["algebraic_anomaly"])

    return continuous_anomaly, algebraic_anomaly


def run_evaluation_pipeline(csv_path: str, sample_size: int = 500):
    """
    Evaluates both deterministic sensors against ground truth labels.

    Args:
        csv_path: Path to a CIC-IDS2017 format CSV.
        sample_size: Total number of flows to evaluate (spread across classes).

    Returns:
        (fig, metrics_df): matplotlib Figure (confusion matrix) and DataFrame (per-class metrics).
    """
    # Load and clean
    df = pd.read_csv(csv_path, nrows=sample_size * 10)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Sample evenly across classes
    classes = df['Label'].unique()
    per_class = max(1, sample_size // len(classes))
    sampled = pd.concat(
        [df[df['Label'] == lbl].head(per_class) for lbl in classes]
    ).reset_index(drop=True)

    y_true_binary = []
    y_pred_binary = []
    y_true_label = []
    y_pred_label = []

    for _, row in sampled.iterrows():
        row_dict = row.to_dict()
        cont_anom, alg_anom = _predict_flow(row_dict)
        predicted_attack = cont_anom or alg_anom
        true_label = row['Label']

        y_true_binary.append("BENIGN" if true_label == "BENIGN" else "ATTACK")
        y_pred_binary.append("ATTACK" if predicted_attack else "BENIGN")
        y_true_label.append(true_label)
        y_pred_label.append("ATTACK" if predicted_attack else "BENIGN")

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=["BENIGN", "ATTACK"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["BENIGN", "ATTACK"])
    ax.set_yticklabels(["BENIGN", "ATTACK"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Subspace Aegis IDS")
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=14, color=color)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    # --- Per-class metrics table ---
    report = classification_report(
        y_true_binary, y_pred_binary, output_dict=True, zero_division=0
    )
    metrics_df = pd.DataFrame(report).T.round(3)
    metrics_df = metrics_df[
        metrics_df.index.isin(["BENIGN", "ATTACK", "accuracy", "macro avg", "weighted avg"])
    ]

    return fig, metrics_df
```

- [ ] **Step 3.4: Run tests to verify they pass**

```bash
cd dataCleaning
python -m pytest tests/test_evaluation.py -v
```

Expected: PASS — 2 tests

- [ ] **Step 3.5: Commit**

```bash
git add dataCleaning/evaluation.py dataCleaning/tests/test_evaluation.py
git commit -m "feat: add evaluation pipeline with confusion matrix and per-class F1"
```

---

## Task 4: Write `app.py` (Gradio UI)

**Files:**
- Create: `dataCleaning/app.py`

- [ ] **Step 4.1: Install dependencies**

```bash
pip install gradio langchain-community
```

Verify:
```bash
python -c "import gradio; print('Gradio', gradio.__version__)"
python -c "from langchain_community.chat_models import ChatOllama; print('ChatOllama OK')"
```

- [ ] **Step 4.2: Create `app.py`**

```python
# dataCleaning/app.py
import os
import sys
import json
import pandas as pd
import numpy as np
import gradio as gr

sys.path.insert(0, os.path.dirname(__file__))

from neuro_symbolic_graph import build_neuro_symbolic_graph, create_llm
from evaluation import run_evaluation_pipeline

# Maps Gradio radio label → create_llm backend key
_BACKEND_MAP = {"Groq Cloud": "groq", "Ollama Local": "ollama"}

# Pre-filled JSON template for Tab 2 (Manual Entry)
_MANUAL_TEMPLATE = json.dumps({
    "Flow Duration": 1000000,
    "Total Fwd Packets": 10,
    "Total Backward Packets": 8,
    "Total Length of Fwd Packets": 1200,
    "Total Length of Bwd Packets": 960,
    "Fwd Packet Length Max": 300,
    "Bwd Packet Length Mean": 150.0,
    "Flow Bytes/s": 1200.0,
    "Flow Packets/s": 18.0,
    "Flow IAT Mean": 55556.0,
    "Packet Length Mean": 120.0,
    "Packet Length Std": 45.0,
    "SYN Flag Count": 1,
    "ACK Flag Count": 8,
    "PSH Flag Count": 2,
    "FIN Flag Count": 1,
    "RST Flag Count": 0,
    "URG Flag Count": 0,
    "Label": "BENIGN"
}, indent=2)


# ─── Tab 1: CSV Batch ───────────────────────────────────────────────────────

def run_batch(csv_file, attack_filter: str, n_flows: int, backend_label: str):
    if csv_file is None:
        return pd.DataFrame([{"error": "Please upload a CSV file."}])

    backend = _BACKEND_MAP[backend_label]
    ids_graph = build_neuro_symbolic_graph(create_llm(backend))

    df = pd.read_csv(csv_file.name, nrows=50000)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if attack_filter != "All":
        df = df[df['Label'] == attack_filter]

    if df.empty:
        return pd.DataFrame([{"error": f"No rows found for filter '{attack_filter}'."}])

    rows = []
    for i, (idx, row) in enumerate(df.iterrows()):
        if i >= int(n_flows):
            break
        row_dict = row.to_dict()
        flow_state = {
            "flow_id": f"flow_{idx}",
            "geometric_features": row_dict,
            "algebraic_features": row_dict,
            "true_label": row['Label'],
            "source_ip": "10.0.0.50",
        }
        result = ids_graph.invoke(flow_state)
        rows.append({
            "flow_id": flow_state["flow_id"],
            "true_label": row['Label'],
            "continuous_anomaly": result.get("continuous_anomaly"),
            "monoid_state": result.get("current_monoid_state"),
            "algebraic_anomaly": result.get("algebraic_anomaly"),
            "final_diagnosis": result.get("final_diagnosis", "—"),
        })

    return pd.DataFrame(rows)


# ─── Tab 2: Manual Entry ─────────────────────────────────────────────────────

def run_single(feature_json: str, backend_label: str):
    try:
        features = json.loads(feature_json)
    except json.JSONDecodeError as e:
        return f"JSON parse error: {e}"

    backend = _BACKEND_MAP[backend_label]
    ids_graph = build_neuro_symbolic_graph(create_llm(backend))

    flow_state = {
        "flow_id": "manual_entry",
        "geometric_features": features,
        "algebraic_features": features,
        "true_label": features.get("Label", "Unknown"),
        "source_ip": "10.0.0.50",
    }
    result = ids_graph.invoke(flow_state)

    verdict = (
        "=== SUBSPACE AEGIS VERDICT ===\n"
        f"Thermodynamic State   : {'PERTURBED' if result.get('continuous_anomaly') else 'EQUILIBRIUM'}\n"
        f"Continuous Anomaly    : {result.get('continuous_anomaly')}\n"
        f"Residual ||w||        : {result.get('thermodynamic_residual', 0.0):.4f}\n"
        f"Monoid Classification : {result.get('current_monoid_state')}\n"
        f"Algebraic Anomaly     : {result.get('algebraic_anomaly')}\n"
        "\n--- LLM DIAGNOSIS ---\n"
        f"{result.get('final_diagnosis', 'No anomaly detected — LLM not invoked.')}"
    )
    return verdict


# ─── Tab 3: Evaluation ───────────────────────────────────────────────────────

def run_evaluation_ui(csv_file, sample_size: int):
    if csv_file is None:
        return None, pd.DataFrame([{"error": "Please upload a CSV file."}])
    fig, metrics_df = run_evaluation_pipeline(csv_file.name, int(sample_size))
    return fig, metrics_df


# ─── Gradio Layout ───────────────────────────────────────────────────────────

with gr.Blocks(title="Subspace Aegis IDS") as demo:
    gr.Markdown("# Subspace Aegis — Neuro-Symbolic Network IDS")
    gr.Markdown(
        "A hybrid LLM-powered intrusion detection system using SVD geometry "
        "and TCP monoid algebra. COS30018 Option D."
    )

    with gr.Tab("CSV Batch"):
        gr.Markdown("### Upload a CIC-IDS2017 CSV and analyse multiple flows")
        with gr.Row():
            batch_csv = gr.File(label="Network Flow CSV", file_types=[".csv"])
            with gr.Column():
                batch_filter = gr.Dropdown(
                    choices=["All", "BENIGN", "DoS Hulk", "DoS slowloris", "DoS Slowhttptest"],
                    value="All",
                    label="Attack Filter"
                )
                batch_n = gr.Slider(minimum=1, maximum=20, step=1, value=4, label="Number of Flows")
                batch_backend = gr.Radio(
                    choices=["Groq Cloud", "Ollama Local"],
                    value="Groq Cloud",
                    label="LLM Backend"
                )
                batch_btn = gr.Button("Run IDS", variant="primary")
        batch_out = gr.Dataframe(label="Results", wrap=True)
        batch_btn.click(
            fn=run_batch,
            inputs=[batch_csv, batch_filter, batch_n, batch_backend],
            outputs=batch_out
        )

    with gr.Tab("Manual Entry"):
        gr.Markdown("### Manually specify a single network flow as JSON")
        with gr.Row():
            with gr.Column():
                manual_json = gr.Code(
                    value=_MANUAL_TEMPLATE,
                    language="json",
                    label="Flow Features (JSON)"
                )
                manual_backend = gr.Radio(
                    choices=["Groq Cloud", "Ollama Local"],
                    value="Groq Cloud",
                    label="LLM Backend"
                )
                manual_btn = gr.Button("Analyse Flow", variant="primary")
            manual_out = gr.Textbox(label="Verdict", lines=12)
        manual_btn.click(
            fn=run_single,
            inputs=[manual_json, manual_backend],
            outputs=manual_out
        )

    with gr.Tab("Evaluation"):
        gr.Markdown("### Benchmark the deterministic sensors against ground truth labels")
        gr.Markdown(
            "_Note: The LLM is not used in evaluation — only the SVD and TCP monoid sensors "
            "are measured. This avoids rate-limit costs and gives clean, reproducible metrics._"
        )
        with gr.Row():
            eval_csv = gr.File(label="Network Flow CSV", file_types=[".csv"])
            with gr.Column():
                eval_n = gr.Slider(minimum=100, maximum=2000, step=100, value=500, label="Sample Size")
                eval_btn = gr.Button("Run Evaluation", variant="primary")
        with gr.Row():
            eval_plot = gr.Plot(label="Confusion Matrix")
            eval_table = gr.Dataframe(label="Precision / Recall / F1")
        eval_btn.click(
            fn=run_evaluation_ui,
            inputs=[eval_csv, eval_n],
            outputs=[eval_plot, eval_table]
        )


if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
```

- [ ] **Step 4.3: Verify Gradio app launches**

```bash
cd dataCleaning
python app.py
```

Expected: terminal output like:
```
Running on local URL:  http://127.0.0.1:7860
```
Open `http://127.0.0.1:7860` in a browser and confirm 3 tabs are visible. Stop with Ctrl+C.

- [ ] **Step 4.4: Commit**

```bash
git add dataCleaning/app.py
git commit -m "feat: add Gradio UI with CSV batch, manual entry, and evaluation tabs"
```

---

## Task 5: Smoke Test End-to-End

- [ ] **Step 5.1: Run all tests**

```bash
cd dataCleaning
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5.2: Verify main_stream_sim still runs (optional — requires dataset CSV)**

If `Wednesday-workingHours.pcap_ISCX.csv` is present:
```bash
cd dataCleaning
python main_stream_sim.py
```

Expected: flows processed, final output prints `final_diagnosis` (not `None`).

- [ ] **Step 5.3: Final commit**

```bash
git add -A
git commit -m "chore: final integration — all deliverables wired up"
```

---

## Dependencies Summary

Install if not already present:
```bash
pip install gradio langchain-community
```

For Ollama local LLM (optional — only needed if using "Ollama Local" in the UI):
```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3.3
```
