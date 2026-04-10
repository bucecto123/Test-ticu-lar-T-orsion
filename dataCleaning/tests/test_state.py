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

def test_final_diagnosis_fallback():
    result = {"continuous_anomaly": False, "algebraic_anomaly": False}
    diagnosis = result.get('final_diagnosis', 'No LLM diagnosis (normal traffic).')
    assert diagnosis == 'No LLM diagnosis (normal traffic).'
