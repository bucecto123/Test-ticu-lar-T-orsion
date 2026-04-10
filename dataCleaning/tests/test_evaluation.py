# dataCleaning/tests/test_evaluation.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import _predict_flow

def test_predict_flow_returns_two_bools():
    # A synthetic flow with all zeros → low residual, no monoid violation → both False
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
