import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
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
    kwargs = {"model": "llama-3.3-70b-versatile", "temperature": 0.0}
    if _api_key:
        kwargs["api_key"] = _api_key
    return ChatGroq(**kwargs)

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
4. Ground your entire diagnosis ONLY in the mathematical facts provided.
5. Reference the residual norm magnitude when describing the physical work performed."""

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
            "spiking_features": state.get("spiking_features") or [],
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

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
