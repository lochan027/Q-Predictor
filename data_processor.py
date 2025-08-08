import pandas as pd
import numpy as np
import requests
import io
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# URLs for sample datasets
SAMPLE_DATA_URLS = {
    "CICIDS2017 (Subset)": "https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/sample/Monday-WorkingHours.pcap_ISCX.csv",
    "KDD99 (Subset)": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B_20Percent.txt"
}

# Column names for KDD99 dataset
KDD99_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label'
]

def load_sample_data(dataset_name):
    """Load a sample dataset for demonstration purposes.
    
    Args:
        dataset_name (str): Name of the dataset to load
        
    Returns:
        pd.DataFrame: Loaded and preprocessed sample dataset
    """
    try:
        if dataset_name == "CICIDS2017 (Subset)":
            # Load CICIDS2017 data
            url = SAMPLE_DATA_URLS[dataset_name]
            response = requests.get(url)
            data = pd.read_csv(io.StringIO(response.text), encoding='latin1')
            
            # Basic preprocessing for CICIDS2017
            # Handle missing values
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.dropna()
            
            # Add a binary label column (1 for attack, 0 for benign)
            data['label'] = data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
            
            # Limit to a subset for faster processing
            benign = data[data['label'] == 0].sample(min(1000, len(data[data['label'] == 0])))
            attacks = data[data['label'] == 1].sample(min(500, len(data[data['label'] == 1])))
            data = pd.concat([benign, attacks])
            
            return data
            
        elif dataset_name == "KDD99 (Subset)":
            # Load KDD99 data
            url = SAMPLE_DATA_URLS[dataset_name]
            response = requests.get(url)
            data = pd.read_csv(io.StringIO(response.text), header=None, names=KDD99_COLUMNS)
            
            # Basic preprocessing for KDD99
            # Convert label to binary (normal=0, attack=1)
            data['label'] = data['label'].apply(lambda x: 0 if x == 'normal' else 1)
            
            # Limit to a subset for faster processing
            benign = data[data['label'] == 0].sample(min(1000, len(data[data['label'] == 0])))
            attacks = data[data['label'] == 1].sample(min(500, len(data[data['label'] == 1])))
            data = pd.concat([benign, attacks])
            
            return data
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
    except Exception as e:
        # If there's an error loading from URL, use a small synthetic dataset
        print(f"Error loading sample data: {e}. Using synthetic data instead.")
        return generate_synthetic_data()

def generate_synthetic_data(n_samples=1500):
    """Generate synthetic network log data for demonstration.
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Synthetic dataset
    """
    # Generate timestamps
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='30s')
    
    # Generate features
    np.random.seed(42)
    data = {
        'timestamp': timestamps,
        'duration': np.random.exponential(30, n_samples),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
        'service': np.random.choice(['http', 'ftp', 'smtp', 'ssh', 'dns'], n_samples),
        'src_bytes': np.random.exponential(500, n_samples),
        'dst_bytes': np.random.exponential(300, n_samples),
        'count': np.random.poisson(3, n_samples),
        'srv_count': np.random.poisson(2, n_samples),
        'same_srv_rate': np.random.beta(5, 2, n_samples),
        'diff_srv_rate': np.random.beta(2, 5, n_samples),
        'dst_host_count': np.random.poisson(10, n_samples),
        'dst_host_srv_count': np.random.poisson(8, n_samples),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate some anomalies (10% of the data)
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples*0.1), replace=False)
    
    # Modify anomalous samples
    df.loc[anomaly_indices, 'src_bytes'] *= 10  # Much larger data transfer
    df.loc[anomaly_indices, 'dst_bytes'] *= 5
    df.loc[anomaly_indices, 'count'] *= 3
    df.loc[anomaly_indices, 'same_srv_rate'] = np.random.beta(1, 5, len(anomaly_indices))
    
    # Add label column (1 for anomaly, 0 for normal)
    df['label'] = 0
    df.loc[anomaly_indices, 'label'] = 1
    
    return df

def preprocess_data(df):
    """Preprocess network log data for anomaly detection.
    
    Args:
        df (pd.DataFrame): Raw network log data
        
    Returns:
        np.ndarray: Preprocessed feature matrix ready for model input
    """
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Drop the label column if it exists (unsupervised learning)
    if 'label' in data.columns:
        data = data.drop('label', axis=1)
    
    # Identify column types
    timestamp_cols = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
    id_cols = [col for col in data.columns if 'id' in col.lower() or 'ip' in col.lower() or 'address' in col.lower()]
    
    # Drop non-feature columns
    cols_to_drop = timestamp_cols + id_cols + ['Label'] if 'Label' in data.columns else timestamp_cols + id_cols
    data = data.drop(cols_to_drop, axis=1, errors='ignore')
    
    # Identify numeric and categorical columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Apply preprocessing
    try:
        processed_data = preprocessor.fit_transform(data)
        return processed_data
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        # Fallback to just using numeric columns if preprocessing fails
        return data[numeric_cols].fillna(0).values