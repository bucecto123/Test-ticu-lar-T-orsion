import pandas as pd
import numpy as np
from typing import Iterator, Dict, Any
import json
import time

# importing the compiled neuro-symbolic graph from step 4
from neuro_symbolic_graph import build_neuro_symbolic_graph, create_llm
ids_graph = build_neuro_symbolic_graph(create_llm("groq"))

def temporal_flow_generator(filepath: str, target_attack: str = None) -> Iterator[Dict[str, Any]]:
    """
    the ingestion manifold.
    simulates the irreversible arrow of time by yielding a single network flow 
    state vector per tick. if a target_attack is specified, it fast-forwards 
    the temporal stream directly to that topological event to save compute.
    """
    print(f"initializing temporal stream. opening {filepath}...")
    
    # chunking the read to handle the massive manifold efficiently
    chunk_size = 10000
    
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        chunk.columns = chunk.columns.str.strip()
        
        # fast-forwarding the time-stream if a specific anomaly is targeted
        if target_attack:
            mask = chunk['Label'].str.contains(target_attack, case=False, na=False)
            chunk = chunk[mask]
            
            if chunk.empty:
                continue # jump to the next block of time
                
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.dropna(inplace=True)
        
        for index, row in chunk.iterrows():
            # THE FIX: convert the entire row into a dictionary
            row_dict = row.to_dict()
            
            # constructing the dual-space dictionary expected by the langgraph agents.
            # we pass the full row_dict into both so the continuous engine can find 
            # its 71 dimensions, and the discrete engine can find its 7 flags.
            flow_state = {
                "flow_id": f"flow_{index}",
                "geometric_features": row_dict, 
                "algebraic_features": row_dict, 
                "true_label": row['Label'] # hidden from the ai, used only for terminal output
            }
            
            yield flow_state

def execute_neuro_symbolic_ids():
    """
    the main driver loop.
    wakes the langgraph agent, streams the network reality into it, 
    and prints the cognitive synthesis.
    """
    dataset_path = 'Wednesday-workingHours.pcap_ISCX.csv'
    
    # THE FIX: remove the target_attack filter so the stream yields ALL traffic.
    # this allows our loop below to actually find BENIGN and Hulk packets.
    stream = temporal_flow_generator(dataset_path, target_attack=None)
    
    print("\n--- NEURO-SYMBOLIC IDS ONLINE ---")
    print("listening to network interface...\n")
    
    # define what we want to test
    targets = ["BENIGN", "DoS Hulk", "DoS slowloris", "DoS Slowhttptest"]
    
    try:
        for flow_pulse in stream:
            label = flow_pulse['true_label']
            
            # if the packet matches one of our targets, test it
            if label in targets:
                print(f"\n{'='*60}")
                print(f"[*] intercepted packet {flow_pulse['flow_id']} | True Label: {label}")
                print("[*] routing state to deterministic sensors (continuous math & discrete logic) parallelly...")
                
                start_time = time.time()

                # thread_id ties this invocation to a session in MemorySaver.
                # using flow_id as the thread so each flow gets its own checkpoint.
                # swap to source_ip if you want per-IP session continuity instead.
                config = {"configurable": {"thread_id": flow_pulse["flow_id"]}}
                result = ids_graph.invoke(flow_pulse, config=config)
                
                latency = time.time() - start_time
                print(f"[*] neuro-symbolic collapse complete in {latency:.2f} seconds.")
                
                print(f"\n=== SYSTEM VERDICT ===")
                print(f"Continuous SVD Anomaly : {result.get('continuous_anomaly')}")
                print(f"Discrete Monoid State  : {result.get('current_monoid_state')}")
                print(f"Topological Anomaly    : {result.get('algebraic_anomaly')}")
                print(f"\n--- PHYSICIST DIAGNOSIS ---")
                print(result.get('final_diagnosis', 'No LLM invoked (equilibrium state).'))
                
                # remove it from targets so we only test one of each class
                targets.remove(label)
                
            # stop once we've tested everything cleanly
            if not targets:
                print("\n[*] all target topologies evaluated. shutting down ingestion manifold.")
                break
                
    except StopIteration:
        print("end of temporal stream reached.")

if __name__ == "__main__":
    execute_neuro_symbolic_ids()