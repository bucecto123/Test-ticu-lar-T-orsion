# Subspace Aegis

> A Neuro-Symbolic Network Intrusion Detection System powered by LangGraph and LLM-constrained physics translation.

**COS30018 — Intelligent Systems | Option D**

---

## Overview

Subspace Aegis is a hybrid intrusion detection system (IDS) that combines deterministic mathematical sensors with a Large Language Model (LLM) to detect and diagnose network threats. Instead of relying purely on pattern matching or black-box ML, the system decomposes network traffic into two mathematical spaces — a **continuous geometric subspace** (SVD) and a **discrete algebraic monoid** (TCP state logic) — and only invokes an LLM when anomalous perturbations are detected.

The core innovation is a **dual-sensor gating architecture** that acts as a thermodynamic filter: benign traffic is dropped at the router level, and only anomalous flows are forwarded to the LLM "Physicist" agent for hallucination-resistant semantic diagnosis.

---

## Architecture

The system is built as a **five-agent LangGraph pipeline**:

```
                    ┌──────────────────┐
                    │   Network Flow   │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌─────────────────────┐     ┌──────────────────────┐
   │   The Mathematician │     │     The Logician      │
   │   (SVD Geometric    │     │   (TCP Monoid State   │
   │    Sensor)          │     │    Machine)           │
   └──────────┬──────────┘     └──────────┬────────────┘
              │                           │
              └──────────┬────────────────┘
                         ▼
              ┌──────────────────────┐
              │   The Orchestrator   │
              │  (Geo-Context Agent) │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │   The Tuning Agent   │
              │  (Threshold Scaler)  │
              └──────────┬───────────┘
                         │
                 ┌───────┴────────┐
                 │    Router      │
                 │ Anomaly? ──No──┼──▶ END (Equilibrium)
                 └───────┬────────┘
                    Yes  │
                         ▼
              ┌──────────────────────┐
              │   The Physicist      │
              │  (LLM Translation   │
              │   via Groq/Ollama)  │
              └──────────────────────┘
```

| Agent | Role | Type |
|---|---|---|
| **Mathematician** | Projects flows onto the SVD invariant subspace; calculates residual ‖w‖ | Deterministic |
| **Logician** | Evaluates TCP flag monoid rules (Xmas, SYN flood, Slowloris, etc.) | Deterministic |
| **Orchestrator** | Injects environmental context (geo-IP) | Deterministic |
| **Tuning Agent** | Dynamically scales the ε threshold based on environment | Deterministic |
| **Physicist** | Translates vector math into Savelyev physics-constrained diagnosis | LLM (Groq/Ollama) |

---

## Dataset

The system is evaluated against the **CIC-IDS2017 Wednesday** benchmark dataset (`Wednesday-workingHours.pcap_ISCX.csv`), which contains:

- ~692K network flows
- BENIGN traffic + 5 attack classes: DoS Hulk, DoS Slowloris, DoS Slowhttptest, DoS GoldenEye, Heartbleed

> **Note:** The CSV dataset is excluded from version control via `.gitignore` due to file size. Download it from [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html).

---

## Project Structure

```
Test-ticu-lar-T-orsion/
├── dataCleaning/
│   ├── app.py                          # Gradio web UI (3 tabs: Batch, Manual, Evaluation)
│   ├── neuro_symbolic_graph.py         # LangGraph pipeline definition (5-agent graph)
│   ├── agent_state.py                  # NetworkFlowState TypedDict (shared memory manifold)
│   ├── algebraic_engine.py             # TCP monoid state machine (The Logician)
│   ├── evaluation.py                   # Deterministic sensor evaluation pipeline
│   ├── wednesday_data_cleaner.py       # Dataset preprocessing + SVD baseline fitting
│   ├── verify_subspace.py              # Visualization of thermodynamic landscape
│   ├── adversarial_test.py             # Acid test: benign vs attack residual distributions
│   ├── main_stream_sim.py              # Live temporal stream simulation
│   │
│   ├── benchmark_vectorized.py         # Metric 1 (Spectral SNR) + Metric 3 (Efficiency)
│   ├── benchmark_stream.py             # Full-dataset streaming benchmark w/ per-class rates
│   ├── benchmark_vgs.py                # Metric 2 (Vector-Grounding Score) — LLM hallucination test
│   │
│   ├── subspace_models/                # Persisted math artifacts
│   │   ├── geometric_scaler.joblib     # Fitted StandardScaler
│   │   ├── p_u_matrix.npy              # SVD projection matrix (P_U)
│   │   └── subspace_config.json        # Epsilon threshold + feature names
│   │
│   ├── results/                        # Benchmark output (JSON + TXT reports)
│   ├── tests/                          # Unit tests for evaluation and state modules
│   │
│   ├── notes_000.md — notes_008.md     # Architectural Decision Records (ADRs)
│   ├── requirements.txt                # Python dependencies
│   ├── .env.example                    # API key template
│   └── .gitignore
│
├── MachineLearningCVE/                 # ML/CVE exploration (dataset excluded)
├── docs/superpowers/                   # Design specs and implementation plans
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/) (free tier) **or** a local [Ollama](https://ollama.com/) instance with `llama3.3`

### Installation

```bash
# Clone the repository
git clone https://github.com/bucecto123/Test-ticu-lar-T-orsion.git
cd Test-ticu-lar-T-orsion/dataCleaning

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy the environment template
cp .env.example .env

# Edit .env and add your API key
# GROQ_API_KEY=gsk_your_key_here
```

### Preparing the Dataset

1. Download `Wednesday-workingHours.pcap_ISCX.csv` from the [CIC-IDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html).
2. Place it in the `dataCleaning/` directory.
3. *(Optional)* Re-fit the SVD baseline — pre-fitted models are already included in `subspace_models/`:

```bash
python wednesday_data_cleaner.py
```

---

## Usage

### Gradio Web Interface

Launch the interactive web UI with three analysis tabs:

```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

| Tab | Description |
|---|---|
| **CSV Batch** | Upload a CIC-IDS2017 CSV, filter by attack class, analyse multiple flows |
| **Manual Entry** | Input a single flow as JSON and receive a full system verdict |
| **Evaluation** | Run deterministic sensor evaluation with confusion matrix and F1 metrics |

### Stream Simulation

Run the temporal stream simulator to process flows sequentially:

```bash
python main_stream_sim.py
```

---

## Benchmarks

### Metric 1 — Spectral Signal-to-Noise Ratio (SNR)

Measures the geometric separation between benign and attack residual distributions.

```bash
python benchmark_vectorized.py
```

### Metric 2 — Vector-Grounding Score (VGS)

Evaluates LLM hallucination resistance by scoring whether the Physicist's output is strictly grounded in mathematical inputs.

```bash
python benchmark_vgs.py
```

### Metric 3 — Thermodynamic Efficiency

Quantifies token savings achieved by the dual-sensor router dropping benign traffic before LLM invocation.

```bash
# Included in the vectorized benchmark
python benchmark_vectorized.py
```

### Key Results

| Metric | Value |
|---|---|
| Overall Spectral SNR | 5.30x |
| Vector-Grounding Score (VGS) | 0.8889 |
| Thermodynamic Efficiency | 6.17x token savings |
| False Positive Rate (FPR) | 0.0190 |
| Combined F1 (Attack) | 0.9028 |

---

## Algebraic Monoid Rules

The Logician evaluates four active TCP state rules:

| Rule | Classification | Trigger Condition |
|---|---|---|
| Xmas Scan | `SCAN_XMAS` | FIN > 0, PSH > 0, URG > 0, SYN = 0, ACK = 0 |
| SYN Flood | `DOS_SYN_FLOOD` | SYN > 20, ACK = 0 |
| Orphaned Loop | `DOS_SLOWLORIS_STEALTH` | SYN > 0, FIN = 0, RST = 0, Duration > 30s |
| PSH Exhaustion | `DOS_L7_EXHAUSTION` | PSH > 50, FIN = 0, RST = 0 |

> **Note:** Null Scan and Stealth PSH rules are disabled for flow-level CICFlowMeter data — see `notes_006` for the full root-cause analysis.

---

## Research Notes (ADRs)

| File | Topic |
|---|---|
| `notes_000.md` | Dataset rationale and topology context |
| `notes_001.md` | Implications of infinity in flow telemetry |
| `notes_002.md` | Geometric subspace decomposition strategy |
| `notes_003.md` | Algebraic monoid formalization |
| `notes_004.md` | Agent pipeline design decisions |
| `notes_005.md` | Three-tiered benchmark taxonomy |
| `notes_006.md` | Monoid rule corrections from vectorized test |
| `notes_007.md` | Detection rate analysis from stream benchmark |
| `notes_008.md` | VGS benchmark partial results and pending |

---

## Tech Stack

- **Framework:** [LangGraph](https://github.com/langchain-ai/langgraph) (multi-agent orchestration)
- **LLM:** [Groq Cloud](https://groq.com/) (Llama 3.3 70B) / Ollama (local)
- **Math:** NumPy, Scikit-learn (PCA/SVD), Joblib
- **UI:** [Gradio](https://gradio.app/)
- **Data:** Pandas, Matplotlib

---

## Team

| Member | Contribution |
|---|---|
| **Hoang Nhat Thanh** | Core SVD geometry engine, LangGraph pipeline, benchmark suite |
| **Dinh Danh Nam** | Algebraic monoid engine, Gradio UI, stream simulator |
| **Ngo Nguyen Phuc** | Research documentation, evaluation methodology, ADRs |
| **Nguyen Quy Hung** | Dataset analysis, benchmark interpretation, ADRs |

---

## License

This project was developed as part of **COS30018 Intelligent Systems** at Swinburne University of Technology.
