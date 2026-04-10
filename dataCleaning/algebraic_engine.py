from typing import Dict, Any

class AlgebraicMonoidEngine:
    """
    the deterministic discrete state-machine (the logician).
    evaluates the strict topological rules of TCP communication.

    note: two rules are disabled for flow-level CICFlowMeter data:
      - null scan (rule 1): no Protocol column, so zero-flag flows include
        UDP/ICMP which are indistinguishable from TCP null scans.
      - stealth PSH (rule 6): CICFlowMeter counts flags per direction, so
        benign flows legitimately show ACK=0 at flow level.
    both rules remain valid for raw packet-level inspection.
    """
    def __init__(self):
        self.duration_threshold  = 30_000_000  # 30 seconds in microseconds
        self.syn_flood_threshold = 20

    def evaluate_state_loop(self, features: Dict[str, Any]) -> Dict[str, Any]:
        syn      = features.get('SYN Flag Count', 0)
        fin      = features.get('FIN Flag Count', 0)
        rst      = features.get('RST Flag Count', 0)
        psh      = features.get('PSH Flag Count', 0)
        ack      = features.get('ACK Flag Count', 0)
        urg      = features.get('URG Flag Count', 0)
        duration = features.get('Flow Duration', 0)

        anomaly        = False
        reasoning      = "tcp state loop closed or maintained normally."
        classification = "BENIGN"

        # --- RULE 2: THE XMAS SCAN (Reconnaissance) ---
        if fin > 0 and psh > 0 and urg > 0 and syn == 0 and ack == 0:
            anomaly        = True
            classification = "SCAN_XMAS"
            reasoning      = "topological violation: illegal combination of FIN, PSH, and URG without ACK (xmas tree scan)."

        # --- RULE 3: THE SYN FLOOD (Denial of Service) ---
        elif syn > self.syn_flood_threshold and ack == 0:
            anomaly        = True
            classification = "DOS_SYN_FLOOD"
            reasoning      = f"topological violation: extreme SYN asymmetry ({syn} SYNs to 0 ACKs). state-table exhaustion."

        # --- RULE 4: THE ORPHANED LOOP (Slowloris / Stealth) ---
        elif syn > 0 and fin == 0 and rst == 0 and duration > self.duration_threshold:
            anomaly        = True
            classification = "DOS_SLOWLORIS_STEALTH"
            reasoning      = f"topological violation: orphaned connection held open for {duration / 1_000_000:.2f} seconds without FIN/RST."

        # --- RULE 5: PUSH EXHAUSTION (Slowhttptest / L7 Floods) ---
        elif psh > 50 and fin == 0 and rst == 0:
            anomaly        = True
            classification = "DOS_L7_EXHAUSTION"
            reasoning      = f"topological violation: excessive PSH flags ({psh}) without loop closure."

        return {
            "algebraic_anomaly":      anomaly,
            "monoid_classification":  classification,
            "monoid_state_reasoning": reasoning
        }

if __name__ == "__main__":
    engine = AlgebraicMonoidEngine()
    stealth_flow = {
        'SYN Flag Count': 2,
        'FIN Flag Count': 0,
        'RST Flag Count': 0,
        'PSH Flag Count': 5,
        'ACK Flag Count': 5,
        'Flow Duration': 55_000_000
    }
    result = engine.evaluate_state_loop(stealth_flow)
    print(f"anomaly detected: {result['algebraic_anomaly']}")
    print(f"logician reasoning: {result['monoid_state_reasoning']}")
