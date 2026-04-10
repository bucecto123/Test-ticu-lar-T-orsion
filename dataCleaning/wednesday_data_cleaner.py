import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import json
import os

# as discussed, we need to persist our math artifacts to disk.
# so langgraph doesnt recalculate the geometry for every single packet later.
os.makedirs('subspace_models', exist_ok=True)

def load_and_split_data(filepath: str, sample_size: int = 100000):
    print('dataset loading')
    
    # grabbing a chunk of the dataset to spare the ram
    df = pd.read_csv(filepath, nrows=sample_size)
    
    # column names, firstly
    # quite mind bogging on how python has this flexibility of allowing
    # such self-referenced statements like this to clean the headers
    df.columns = df.columns.str.strip()
    
    # replaces infty with nan... then drops them.
    # if a flow duration is 0 (like a single ping packet), velocity is inf.
    # if inf gets into the svd, the covariance matrix mathematically collapses
    # because you cant draw an orthogonal projection from an infinite distance.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    df.reset_index(drop=True, inplace=True)
    
    # save labels for baseline fitting later
    labels = df['Label']
    
    # splitting phase
    print('splitting algebraic and geometric spaces')
    
    # the discrete space (think of it as lang's monoid)
    # these are purely topological state transitions, not physical forces
    algebraic_features = [
        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
        'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count'
    ]
    df_algebraic = df[algebraic_features]
    
    # continuous space (axler's geometry, and we will deal with thermo-
    # dynamically stable states later. i.e., anything that isnt continous)
    # we drop categorical flags and identities to prevent spatial overfitting
    df_geometric = df.drop(columns=algebraic_features + ['Label', 'Destination Port'], errors='ignore')

    # scaling features
    scaler = StandardScaler()
    
    # fit and transform the geometric data to unit variance, z score
    # so massive variables like flow duration dont mathematically eclipse smaller ones
    geometric_matrix_scaled = scaler.fit_transform(df_geometric)
    
    # re-wrap in a df to keep column name attached for later translation
    df_geometric_scaled = pd.DataFrame(geometric_matrix_scaled, columns=df_geometric.columns)
    
    # persisting the scaler for the langgraph node to use during live inference
    joblib.dump(scaler, 'subspace_models/geometric_scaler.joblib')
    
    print(f"algebraic matrix shape: {df_algebraic.shape}")
    print(f"geometric matrix shape: {df_geometric_scaled.shape}")
    print(f"total labels: {labels.value_counts().to_dict()}")
    
    return df_algebraic, df_geometric_scaled, labels

def fit_and_save_baseline(df_geometric_scaled: pd.DataFrame, labels: pd.Series):
    print('isolating benign traffic to define equilibrium')
    
    # we only feed normal traffic to the svd so it learns the pure baseline
    normal_mask = labels == 'BENIGN'
    x_normal = df_geometric_scaled[normal_mask].values
    
    # finding the invariant subspace using pca (which wraps svd behind the scenes). 
    # keeping 95 percent of the variance.
    pca = PCA(n_components=0.95, svd_solver='full')
    pca.fit(x_normal)
    
    # u is our basis matrix, p_u is the actual projection matrix
    u = pca.components_.T 
    p_u = np.dot(u, u.T)
    
    # projecting normal traffic to calculate the thermodynamic residual ||w||
    x_normal_projected = np.dot(x_normal, p_u)
    w_normal = x_normal - x_normal_projected
    
    # calculating the savelyev magnitude across the row axis
    residual_norms = np.linalg.norm(w_normal, axis=1)
    
    # epsilon threshold set to 99th percentile to allow for microscopic noise
    # it is set at 99th percentile, for testing purposes
    epsilon = np.percentile(residual_norms, 99)
    
    # saving the mathematical laws to disk for the agents
    np.save('subspace_models/p_u_matrix.npy', p_u)
    
    # saving a json config so the llm has access to the feature names
    config = {
        "epsilon_threshold": float(epsilon),
        "geometric_feature_names": df_geometric_scaled.columns.tolist()
    }
    with open('subspace_models/subspace_config.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"strict epsilon threshold (\u03B5): {epsilon:.4f}")

# pipeline execution
if __name__ == "__main__":
    filepath = 'Wednesday-workingHours.pcap_ISCX.csv'
    df_alg, df_geom, labels = load_and_split_data(filepath)
    fit_and_save_baseline(df_geom, labels)