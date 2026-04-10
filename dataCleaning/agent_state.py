from typing import TypedDict, Dict, Any, Optional, List

class NetworkFlowState(TypedDict):
    """
    The shared memory manifold for the LangGraph neuro-symbolic agents.
    Dictates exactly what information can be passed between the deterministic
    sensors and the cognitive orchestrator.
    """
    flow_id: str
    true_label: str

    # Raw input data routed from the temporal stream.
    # NOTE: In the current simulation (main_stream_sim.py), both dicts receive
    # the full row_dict so all nodes can access all features. The algebraic_engine
    # reads TCP flag columns; the continuous sensor reads the 71 geometric columns.
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
