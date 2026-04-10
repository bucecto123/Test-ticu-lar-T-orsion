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
        result = ids_graph.invoke(flow_state, config={"configurable": {"thread_id": flow_state["flow_id"]}})
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
    result = ids_graph.invoke(flow_state, config={"configurable": {"thread_id": flow_state["flow_id"]}})

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
