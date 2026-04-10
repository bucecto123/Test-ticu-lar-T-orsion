import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

def run_acid_test(filepath: str, sample_size: int = 100000):
    print('initiating the acid test: colliding attacks with the invariant subspace...')
    
    # 1. load the raw battlefield
    # i felt funny so i just removed the sample size altogether
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    labels = df['Label']
    
    # 2. isolating continuous space
    algebraic_features = [
        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
        'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count'
    ]
    df_geometric = df.drop(columns=algebraic_features + ['Label', 'Destination Port'], errors='ignore')
    
    # 3. loading our saved reality
    scaler = joblib.load('subspace_models/geometric_scaler.joblib')
    p_u = np.load('subspace_models/p_u_matrix.npy')
    with open('subspace_models/subspace_config.json', 'r') as f:
        config = json.load(f)
    epsilon = config['epsilon_threshold']
    
    # 4. the collision
    x_scaled = scaler.transform(df_geometric)
    x_projected = np.dot(x_scaled, p_u)
    
    w_matrix = x_scaled - x_projected
    residuals = np.linalg.norm(w_matrix, axis=1)
    
    # 5. separate the forces for visualization (NOW INCLUDING SLOWHTTPTEST)
    mask_benign = labels == 'BENIGN'
    mask_hulk = labels == 'DoS Hulk'
    mask_slowloris = labels == 'DoS slowloris'
    mask_slowhttptest = labels == 'DoS Slowhttptest'
    
    res_benign = residuals[mask_benign]
    res_hulk = residuals[mask_hulk]
    res_slowloris = residuals[mask_slowloris]
    res_slowhttptest = residuals[mask_slowhttptest]
    
    print(f"max benign residual: {np.max(res_benign):.2f} (should be near {epsilon:.2f})")
    print(f"avg hulk residual: {np.mean(res_hulk):.2f}")
    print(f"avg slowloris residual: {np.mean(res_slowloris):.2f}")
    # checking if our hypothesis holds
    if len(res_slowhttptest) > 0:
        print(f"avg slowhttptest residual: {np.mean(res_slowhttptest):.2f}")
    else:
        print("no slowhttptest samples found in this chunk. you may need to increase sample_size.")
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(res_benign, bins=50, color='#00ff9f', alpha=0.6, label='benign (equilibrium)', density=True)
    ax.hist(res_hulk, bins=50, color='#ff003c', alpha=0.6, label='dos hulk (rupture)', density=True)
    ax.hist(res_slowloris, bins=50, color='#ffaa00', alpha=0.6, label='dos slowloris (rupture)', density=True)
    
    # adding the new vector
    if len(res_slowhttptest) > 0:
        ax.hist(res_slowhttptest, bins=50, color='#00aaff', alpha=0.6, label='dos slowhttptest (rupture)', density=True)
    
    ax.axvline(epsilon, color='white', linestyle='--', linewidth=2, label=f'\u03B5 boundary ({epsilon:.2f})')
    
    ax.set_title('the acid test: thermodynamic residuals of benign vs. attack flows', fontsize=12)
    ax.set_xlabel('residual magnitude ||w|| (kinetic energy)')
    ax.set_ylabel('normalized density')
    
    ax.legend()
    plt.grid(True, linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('acid_test_results.png', dpi=300)
    print("saved collision data to 'acid_test_results.png'")

if __name__ == "__main__":
    run_acid_test('Wednesday-workingHours.pcap_ISCX.csv', sample_size=500000) # increased sample size to ensure we catch it