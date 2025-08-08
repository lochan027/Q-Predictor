import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
import io
import requests
from datetime import datetime

# Import our custom modules
from data_processor import preprocess_data, load_sample_data
from quantum_model import QuantumAnomalyDetector
from visualization import plot_anomaly_scores, plot_confusion_matrix, plot_feature_importance

# Set page configuration
st.set_page_config(page_title="Q-Predictor: Quantum-Enhanced Anomaly Detection", 
                   layout="wide",
                   initial_sidebar_state="expanded")

# Custom CSS for a modern, dark theme with improved styling
st.markdown("""
<style>
    /* General Theme */
    body {
        color: #E0E0E0;
        background-color: #121212;
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #121212;
    }
    
    /* Main Headers */
    .main-header {
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00A8E8, #00E4FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2.5rem;
        text-shadow: 0px 2px 4px rgba(0, 0, 0, 0.5);
        letter-spacing: -0.5px;
    }
    
    /* Sub Headers */
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #00C8FF;
        border-bottom: 2px solid #00C8FF;
        padding-bottom: 0.6rem;
        margin-top: 2.5rem;
        margin-bottom: 1.8rem;
        letter-spacing: -0.3px;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1cypcdb, .css-1oe6wy4 {
        background-color: #1A1A1A !important;
        border-right: 2px solid #00C8FF;
    }
    .css-1d391kg .st-emotion-cache-10trblm, .css-1cypcdb .st-emotion-cache-10trblm, .css-1oe6wy4 .st-emotion-cache-10trblm {
        color: #F0F0F0;
    }
    
    /* Radio buttons and Selectbox */
    .stRadio > label, .stSelectbox > label, .stCheckbox > label {
        font-weight: 600;
        color: #00C8FF;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Sliders */
    .stSlider > label {
        color: #00C8FF;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .stSlider .st-emotion-cache-1q3jlrr {
        background-color: #00C8FF;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #007BFF, #00C8FF);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 0.5rem;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0, 168, 232, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0056b3, #00A8E8);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 168, 232, 0.5);
    }
    
    /* Info Box */
    .info-box {
        background-color: #1A1A1A;
        border: 1px solid #00C8FF;
        padding: 1.8rem;
        border-radius: 0.8rem;
        margin-bottom: 2.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        line-height: 1.6;
    }
    
    /* Expander */
    .st-expander {
        border: 1px solid #00C8FF;
        border-radius: 0.8rem;
        background-color: #1A1A1A;
        margin-bottom: 1.5rem;
    }
    
    /* Dataframe */
    .stDataFrame {
        border: 1px solid #00C8FF;
        border-radius: 0.5rem;
        overflow: hidden;
    }
    .stDataFrame table {
        background-color: #1A1A1A;
    }
    .stDataFrame th {
        background-color: #00A8E8 !important;
        color: white !important;
        font-weight: 600;
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #00C8FF;
        border-radius: 0.8rem;
        padding: 1rem;
    }
    
    /* Code blocks */
    code {
        background-color: #1A1A1A;
        color: #00C8FF;
        padding: 0.2rem 0.4rem;
        border-radius: 0.3rem;
    }
    
    /* Links */
    a {
        color: #00C8FF;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
        color: #00E4FF;
    }
    
    /* Markdown text */
    p, li {
        line-height: 1.6;
        font-size: 1rem;
    }
    
    /* Bold text */
    strong {
        color: #00E4FF;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">Q-Predictor: Quantum-Enhanced Network Anomaly Detection</h1>', unsafe_allow_html=True)
st.markdown('<div class="info-box">This application uses a hybrid classical-quantum approach to detect anomalies in network traffic data. Upload your network logs or use our sample dataset to identify potential threats.</div>', unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'anomaly_scores' not in st.session_state:
    st.session_state.anomaly_scores = None
if 'quantum_enhanced_scores' not in st.session_state:
    st.session_state.quantum_enhanced_scores = None
if 'threshold' not in st.session_state:
    st.session_state.threshold = -0.5  # Default threshold for Isolation Forest

# Sidebar for data loading and model parameters
with st.sidebar:
    st.markdown('<h2 class="sub-header">Data & Model Settings</h2>', unsafe_allow_html=True)
    
    # Data loading options
    data_option = st.radio("Select data source:", ["Upload CSV", "Use sample data"])
    
    if data_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload network log file (CSV)", type=["csv"])
        if uploaded_file is not None:
            try:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.success(f"Loaded data with {st.session_state.data.shape[0]} rows and {st.session_state.data.shape[1]} columns")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    else:
        dataset_option = st.selectbox("Select sample dataset:", ["CICIDS2017 (Subset)", "KDD99 (Subset)"])
        if st.button("Load Sample Data"):
            with st.spinner("Loading sample data..."):
                st.session_state.data = load_sample_data(dataset_option)
                st.success(f"Loaded {dataset_option} with {st.session_state.data.shape[0]} rows")
    
    # Model parameters
    st.markdown("### Model Parameters")
    
    # Classical model parameters
    n_estimators = st.slider("Number of estimators (Isolation Forest)", 50, 500, 100, 50)
    contamination = st.slider("Expected contamination rate", 0.01, 0.5, 0.1, 0.01)
    
    # Quantum model parameters
    use_quantum = st.checkbox("Enable quantum enhancement", True)
    if use_quantum:
        quantum_backend = st.selectbox("Quantum Backend", ["qasm_simulator", "statevector_simulator"])
        quantum_shots = st.slider("Number of shots", 100, 2000, 1000, 100)
    
    # Threshold for anomaly detection
    st.session_state.threshold = st.slider("Anomaly threshold", -1.0, 0.0, -0.5, 0.05)
    
    # Run analysis button
    run_button = st.button("Run Analysis")

# Main content area
if st.session_state.data is not None:
    # Display data overview
    st.markdown('<h2 class="sub-header">Data Overview</h2>', unsafe_allow_html=True)
    
    # Show data sample and info
    with st.expander("View Data Sample"):
        st.dataframe(st.session_state.data.head(10))
        
        # Display basic statistics
        col1, col2 = st.columns(2)
        with col1:
            st.write("Data Shape:", st.session_state.data.shape)
        with col2:
            st.write("Data Types:")
            st.write(st.session_state.data.dtypes)
    
    # Run analysis when button is clicked
    if run_button:
        with st.spinner("Processing data and running anomaly detection..."):
            # Preprocess the data
            st.session_state.processed_data = preprocess_data(st.session_state.data)
            
            # Run classical anomaly detection (Isolation Forest)
            model = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=42
            )
            
            # Fit the model and get anomaly scores
            model.fit(st.session_state.processed_data)
            st.session_state.anomaly_scores = model.decision_function(st.session_state.processed_data)
            
            # If quantum enhancement is enabled, apply quantum model
            if use_quantum:
                quantum_detector = QuantumAnomalyDetector(
                    backend=quantum_backend,
                    shots=quantum_shots
                )
                st.session_state.quantum_enhanced_scores = quantum_detector.enhance_anomaly_detection(
                    st.session_state.processed_data, 
                    st.session_state.anomaly_scores
                )
                # Use quantum-enhanced scores for predictions
                predictions = (st.session_state.quantum_enhanced_scores < st.session_state.threshold).astype(int)
            else:
                # Use classical scores for predictions
                predictions = (st.session_state.anomaly_scores < st.session_state.threshold).astype(int)
                st.session_state.quantum_enhanced_scores = None
            
            # Add predictions to the original data
            results_df = st.session_state.data.copy()
            results_df['anomaly_score'] = st.session_state.anomaly_scores
            if st.session_state.quantum_enhanced_scores is not None:
                results_df['quantum_enhanced_score'] = st.session_state.quantum_enhanced_scores
            results_df['is_anomaly'] = predictions
            
            # Display results
            st.markdown('<h2 class="sub-header">Analysis Results</h2>', unsafe_allow_html=True)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", st.session_state.data.shape[0])
            with col2:
                st.metric("Detected Anomalies", predictions.sum())
            with col3:
                st.metric("Anomaly Rate", f"{predictions.sum() / len(predictions):.2%}")
            
            # Visualizations
            st.markdown('<h3 class="sub-header">Visualization</h3>', unsafe_allow_html=True)
            
            # Anomaly score distribution
            fig_scores = plot_anomaly_scores(
                classical_scores=st.session_state.anomaly_scores,
                quantum_scores=st.session_state.quantum_enhanced_scores,
                threshold=st.session_state.threshold
            )
            st.plotly_chart(fig_scores, use_container_width=True)
            
            # Time series plot if timestamp column exists
            timestamp_cols = [col for col in results_df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if timestamp_cols:
                try:
                    time_col = timestamp_cols[0]
                    # Convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(results_df[time_col]):
                        results_df[time_col] = pd.to_datetime(results_df[time_col], errors='coerce')
                    
                    # Create time series plot
                    fig_time = px.scatter(
                        results_df.sort_values(time_col),
                        x=time_col,
                        y='anomaly_score' if st.session_state.quantum_enhanced_scores is None else 'quantum_enhanced_score',
                        color='is_anomaly',
                        color_discrete_map={0: 'blue', 1: 'red'},
                        title="Anomaly Scores Over Time",
                        labels={"is_anomaly": "Is Anomaly"},
                        hover_data=results_df.columns[:5].tolist()
                    )
                    st.plotly_chart(fig_time, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create time series plot: {e}")
            
            # Display anomalous records
            st.markdown('<h3 class="sub-header">Detected Anomalies</h3>', unsafe_allow_html=True)
            anomalies = results_df[results_df['is_anomaly'] == 1]
            if len(anomalies) > 0:
                st.dataframe(anomalies)
                
                # Download results button
                csv_buffer = io.StringIO()
                anomalies.to_csv(csv_buffer, index=False)
                csv_str = csv_buffer.getvalue()
                st.download_button(
                    label="Download Anomalies CSV",
                    data=csv_str,
                    file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No anomalies detected with the current threshold.")

else:
    # Display instructions when no data is loaded
    st.info("Please upload a network log file (CSV) or use the sample data to begin analysis.")
    
    # Show example of expected data format
    st.markdown('<h3 class="sub-header">Expected Data Format</h3>', unsafe_allow_html=True)
    st.markdown("""
    Your CSV file should contain network log data with features such as:
    - Timestamp information
    - Source/Destination IP addresses
    - Protocol information
    - Connection details (duration, bytes transferred, etc.)
    - Service types
    - Flag information
    
    The sample datasets (CICIDS2017 and KDD99) are already formatted appropriately.
    """)
    
    # Display app workflow
    st.markdown('<h3 class="sub-header">How It Works</h3>', unsafe_allow_html=True)
    st.markdown("""
    The Q-Predictor application follows a four-step process to detect anomalies in your network data:
    
    1. **Data Ingestion & Preprocessing**: 
        - You can either upload your own CSV file or use one of the provided sample datasets (CICIDS2017 or KDD99).
        - The application automatically preprocesses the data by handling missing values, scaling numerical features, and one-hot encoding categorical features. This ensures the data is in the right format for the anomaly detection models.

    2. **Classical Anomaly Detection**: 
        - The preprocessed data is fed into an Isolation Forest model, a classical machine learning algorithm that is highly effective for anomaly detection.
        - The model calculates an anomaly score for each data point, where lower scores indicate a higher likelihood of being an anomaly.

    3. **Quantum Enhancement (Optional)**:
        - If you enable quantum enhancement, the classical anomaly scores are further refined using a quantum computing-inspired approach.
        - The application uses a Quantum-Enhanced Clustering or a Quantum Boltzmann Machine (QBM) method to improve the detection of subtle anomalies that may be missed by classical methods.

    4. **Visualization & Analysis**:
        - The application provides a comprehensive set of visualizations to help you analyze the results, including:
            - Anomaly score distributions
            - Time series plots of anomaly scores
            - A table of detected anomalies
        - You can also download the list of detected anomalies for further investigation.
    """)
    
    # Display future features
    st.markdown('<h3 class="sub-header">Future Features</h3>', unsafe_allow_html=True)
    st.markdown("""
    We're continuously improving Q-Predictor with exciting new capabilities:
    
    1. **Real-time Monitoring**:
        - Live network traffic analysis with instant alerts for detected anomalies
        - Dashboard with real-time visualization of network health metrics
    
    2. **Advanced Quantum Algorithms**:
        - Integration with IBM Quantum hardware for true quantum advantage
        - Implementation of Quantum Neural Networks for enhanced detection accuracy
    
    3. **Threat Intelligence Integration**:
        - Automatic correlation with known threat databases
        - Contextual information about detected anomalies and potential mitigation strategies
    
    4. **Explainable AI**:
        - Detailed explanations of why specific traffic was flagged as anomalous
        - Feature importance visualization for better understanding of detection decisions
    
    5. **Custom Alert System**:
        - Configurable notification thresholds and delivery methods
        - Integration with popular security information and event management (SIEM) systems
    """)