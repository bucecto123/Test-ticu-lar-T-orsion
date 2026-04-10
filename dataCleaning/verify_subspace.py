import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib
import json

def visualize_persisted_geometry(filepath: str, sample_size: int = 100000):
    print('waking up the visualizer and loading persisted physics...')
    
    # 1. load the raw data again (just for testing the saved models)
    df = pd.read_csv(filepath, nrows=sample_size)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # isolate the normal traffic labels so we can plot the heavy tail
    labels = df['Label']
    normal_mask = labels == 'BENIGN'
    
    # grab continuous dims, ignore the categorical monoid flags
    algebraic_features = [
        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
        'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count'
    ]
    df_geometric = df.drop(columns=algebraic_features + ['Label', 'Destination Port'], errors='ignore')
    
    # ---------------------------------------------------------
    # THE CONNECTION: loading reality from disk
    # ---------------------------------------------------------
    # rather than fitting a new scaler, we load the saved one
    # to enforce identical spatial normalization.
    scaler = joblib.load('subspace_models/geometric_scaler.joblib')
    x_scaled = scaler.transform(df_geometric)
    x_normal = x_scaled[normal_mask]
    
    # loadinggeometric blueprint (the invariant subspace matrix)
    p_u = np.load('subspace_models/p_u_matrix.npy')
    
    # loading the strict boundary we calculated earlier
    with open('subspace_models/subspace_config.json', 'r') as f:
        config = json.load(f)
    saved_epsilon = config['epsilon_threshold']

    # ---------------------------------------------------------
    # THE MATHEMATICS: projecting and measuring
    # ---------------------------------------------------------
    # project normal traffic onto the loaded p_u subspace
    x_normal_projected = np.dot(x_normal, p_u)
    
    # w = v - p_u(v) ... the kinetic energy that didn't fit in the blueprint
    w_normal = x_normal - x_normal_projected
    residual_norms = np.linalg.norm(w_normal, axis=1)

    # ---------------------------------------------------------
    # VISUALIZATION: plotting the thermodynamic landscape
    # ---------------------------------------------------------
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(18, 5))
    
    # plot 1: the heavy tail of benign traffic
    ax1 = fig.add_subplot(131)
    ax1.hist(residual_norms, bins=150, color='#00ff9f', alpha=0.7)
    
    # dropping the anchor lines to prove wheresaved epsilon sits
    p90 = np.percentile(residual_norms, 90)
    ax1.axvline(p90, color='yellow', linestyle='--', label=f'90th: {p90:.2f}')
    
    # this red line should match saved_epsilon perfectly
    ax1.axvline(saved_epsilon, color='red', linestyle='-', linewidth=2, label=f'saved \u03B5: {saved_epsilon:.2f}')
    
    ax1.set_title('thermodynamic distribution of benign flows', fontsize=10)
    ax1.set_xlabel('residual magnitude ||w||')
    ax1.set_ylabel('frequency')
    ax1.set_xlim(0, 15) 
    ax1.legend()

    # plot 2: the epsilon sensitivity curve
    ax2 = fig.add_subplot(132)
    percentiles = np.arange(80, 100, 0.5)
    epsilons = [np.percentile(residual_norms, p) for p in percentiles]
    
    ax2.plot(percentiles, epsilons, color='#ff007f', linewidth=2)
    ax2.axvline(90, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(95, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(99, color='white', linestyle='--', label='99th % (set threshold)')
    
    ax2.set_title('epsilon sensitivity to percentile choice', fontsize=10)
    ax2.set_xlabel('percentile')
    ax2.set_ylabel('calculated \u03B5 threshold')
    ax2.legend()
    
    # plot 3: dimensional compression vs. threshold
    # dynamically calculating what happens if we squeezed the subspace tighter
    ax3 = fig.add_subplot(133)
    variances = [0.80, 0.85, 0.90, 0.95, 0.99]
    eps_by_var = []
    dims = []
    
    for var in variances:
        pca = PCA(n_components=var, svd_solver='full')
        pca.fit(x_normal)
        test_p_u = np.dot(pca.components_.T, pca.components_)
        test_w = x_normal - np.dot(x_normal, test_p_u)
        eps_by_var.append(np.percentile(np.linalg.norm(test_w, axis=1), 99))
        dims.append(pca.n_components_)

    ax3.plot(variances, eps_by_var, color='#00b8ff', marker='o', label='\u03B5 threshold')
    ax3.set_xlabel('variance kept in subspace')
    ax3.set_ylabel('\u03B5 threshold', color='#00b8ff')
    ax3.tick_params(axis='y', labelcolor='#00b8ff')
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(variances, dims, color='white', linestyle='--', marker='x')
    ax3_twin.set_ylabel('dimensions ($k$)', color='white')
    ax3_twin.tick_params(axis='y', labelcolor='white')
    
    ax3.set_title('subspace variance vs. threshold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('subspace_visualizations.png', dpi=300)
    print("saved reality to 'subspace_visualizations.png'")

if __name__ == "__main__":
    visualize_persisted_geometry('Wednesday-workingHours.pcap_ISCX.csv')