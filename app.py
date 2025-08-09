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
import argparse
import time
import subprocess
import sys

# Conditional imports for Scapy and Pyshark
try:
    from scapy.all import rdpcap, sendp, Ether, Packet
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("Warning: Scapy not found. PCAP replay via Scapy will not be available.")

try:
    import pyshark
    PYSHARK_AVAILABLE = True
except ImportError:
    PYSHARK_AVAILABLE = False
    print("Warning: Pyshark not found. Live capture will not be available.")

# Import our custom modules
from data_processor import preprocess_data, load_sample_data
from quantum_model import QuantumAnomalyDetector
from visualization import plot_anomaly_scores, plot_confusion_matrix, plot_feature_importance

# Set page configuration
st.set_page_config(page_title="Q-Predictor: Quantum-Enhanced Anomaly Detection", 
                   layout="wide",
                   initial_sidebar_state="expanded")

# Argument parsing
parser = argparse.ArgumentParser(description="Q-Predictor: Quantum-Enhanced Network Anomaly Detection")
parser.add_argument('--replay', type=str, help='Path to a .pcap file for replay simulation.')
parser.add_argument('--live', type=str, help='Network interface for live capture (e.g., eth0).')
args = parser.parse_args()

# Determine operating mode and log
if args.replay:
    MODE = 'replay'
    PCAP_FILE = args.replay
    print(f"Live capture capability integrated — currently running in simulated mode with recorded traffic from {PCAP_FILE} for safety.")
    st.sidebar.success(f"Running in Simulated Mode: Replaying {PCAP_FILE}")
elif args.live:
    MODE = 'live'
    LIVE_INTERFACE = args.live
    print(f"Live capture capability integrated — currently running in live mode on interface {LIVE_INTERFACE}.")
    st.sidebar.success(f"Running in Live Mode: Capturing on {LIVE_INTERFACE}")
else:
    MODE = 'simulation' # Default to simulation if no args are provided
    print("Live capture capability integrated — currently running in simulated mode with recorded traffic for safety.")
    st.sidebar.info("Running in Simulated Mode: Using sample data.")




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

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'anomaly_scores' not in st.session_state:
    st.session_state.anomaly_scores = None
if 'threshold' not in st.session_state:
    st.session_state.threshold = None
if 'quantum_enhanced_scores' not in st.session_state:
    st.session_state.quantum_enhanced_scores = None

# Initialize and train a simple Isolation Forest model for real-time processing
# This model will be used for initial anomaly scoring of individual packets.
# For a more robust solution, this model should be trained on a representative dataset.
if 'isolation_forest_model' not in st.session_state:
    try:
        # Load a small sample dataset to train the initial Isolation Forest model
        sample_df = load_sample_data("CICIDS2017 (Subset)")
        if sample_df is not None:
            processed_sample_data = preprocess_data(sample_df)
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_forest.fit(processed_sample_data)
            st.session_state.isolation_forest_model = iso_forest
            st.session_state.initial_threshold = iso_forest.threshold_
            st.success("Initial Isolation Forest model trained for real-time packet processing.")
        else:
            st.warning("Could not load sample data to train initial Isolation Forest model.")
    except Exception as e:
        st.error(f"Error training initial Isolation Forest model: {e}")

# Initialize QuantumAnomalyDetector
if 'quantum_detector' not in st.session_state:
    st.session_state.quantum_detector = QuantumAnomalyDetector()
    st.info("Quantum Anomaly Detector initialized.")

# Check for tcpreplay availability
def tcpreplay_available():
    try:
        subprocess.run(['tcpreplay', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

TCPSPLIT_AVAILABLE = tcpreplay_available()

# Function to replay PCAP using tcpreplay
def replay_pcap_tcpreplay(pcap_file):
    if not TCPSPLIT_AVAILABLE:
        st.error("tcpreplay not found. Please install it for optimal PCAP replay.")
        return
    try:
        st.info(f"Replaying {pcap_file} using tcpreplay...")
        # tcpreplay -i <interface> -t <pcap_file>
        # For now, we'll just simulate the replay process as we don't have a live interface to send to.
        # In a real scenario, you'd need to specify the output interface.
        # subprocess.run(['sudo', 'tcpreplay', '-i', 'lo', pcap_file], check=True)
        # Simulate processing time
        time.sleep(5) 
        st.success(f"Finished replaying {pcap_file} with tcpreplay (simulated).")
    except Exception as e:
        st.error(f"Error replaying PCAP with tcpreplay: {e}")

# Sidebar for data loading and model parameters
with st.sidebar:
    st.markdown('<h2 class="sub-header">Data & Model Settings</h2>', unsafe_allow_html=True)

# Function to replay PCAP using Scapy
def replay_pcap_scapy(pcap_file):
    if not SCAPY_AVAILABLE:
        st.error("Scapy not found. Cannot replay PCAP via Scapy.")
        return
    try:
        st.info(f"Replaying {pcap_file} using Scapy...")
        packets = rdpcap(pcap_file)
        first_timestamp = packets[0].time

        processed_packets_data = []
        latency_logs = []

        for i, packet in enumerate(packets):
            # Simulate real-time replay
            if i > 0:
                time_diff = packet.time - packets[i-1].time
                time.sleep(max(0, time_diff))

            arrival_time = datetime.now()
            
            # Convert Scapy packet to a format suitable for your existing detection logic
            # This is a placeholder. You'll need to adapt this to your actual packet processing.
            # For demonstration, we'll create a dummy DataFrame row.
            packet_summary = f"Packet {i+1}: {packet.summary()}"
            # Extract relevant fields from Scapy packet
            # This mapping is a simplification and might need further refinement
            # based on the exact structure of your PCAP files and the features expected by your model.
            protocol_type = packet.name.lower() if hasattr(packet, 'name') else 'unknown'
            service = 'unknown'
            if 'TCP' in packet:
                service = str(packet.dport) if hasattr(packet, 'dport') else 'unknown'
                protocol_type = 'tcp'
            elif 'UDP' in packet:
                service = str(packet.dport) if hasattr(packet, 'dport') else 'unknown'
                protocol_type = 'udp'
            elif 'ICMP' in packet:
                protocol_type = 'icmp'

            src_bytes = len(packet) if hasattr(packet, '__len__') else 0
            dst_bytes = 0 # Scapy doesn't easily give dst_bytes directly for a single packet

            dummy_data = {
                'duration': 0, # Duration is hard to get from single packet
                'protocol_type': protocol_type,
                'service': service,
                'flag': 'unknown', # Placeholder
                'src_bytes': src_bytes,
                'dst_bytes': dst_bytes,
                'land': 0, 'wrong_fragment': 0, 'urgent': 0, 'hot': 0,
                'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0,
                'root_shell': 0, 'su_attempted': 0, 'num_root': 0,
                'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
                'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
                'count': 1, 'srv_count': 1, 'serror_rate': 0.0,
                'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0,
                'same_srv_rate': 1.0, 'diff_srv_rate': 0.0,
                'srv_diff_host_rate': 0.0, 'dst_host_count': 1,
                'dst_host_srv_count': 1, 'dst_host_same_srv_rate': 1.0,
                'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 1.0,
                'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0,
                'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0,
                'dst_host_srv_rerror_rate': 0.0,
                'label': 0 # Assume benign for now
            }
            # Ensure all expected columns are present, fill with 0 or appropriate defaults
            # This needs to match the columns expected by your preprocess_data function
            # For now, we'll just use a simple DataFrame
            packet_df = pd.DataFrame([dummy_data])

            # Process the packet through the detection pipeline
            # This part needs to be integrated with your actual detection logic
            detection_result = process_packet(packet_df, packet_summary, arrival_time)
            processed_packets_data.append(detection_result['processed_data'])
            latency_logs.append(detection_result['latency_log'])

        # Concatenate all processed dataframes
        final_processed_df = pd.concat(processed_packets_data, ignore_index=True)
        st.session_state.data = final_processed_df # Store raw-like data
        st.session_state.processed_data = final_processed_df # Store processed data
        st.write("Latency Logs:", pd.DataFrame(latency_logs))
        st.markdown('<h3 class="sub-header">Replayed Data Preview</h3>', unsafe_allow_html=True)
        st.write(st.session_state.processed_data.head())

        st.success(f"Finished replaying {pcap_file} with Scapy.")
        # Concatenate all processed dataframes


    except Exception as e:
        st.error(f"Error replaying PCAP with Scapy: {e}")
        return None, None

# Function to process a single packet and measure latency
def process_packet(packet_data, packet_summary, arrival_time):
    start_detection_time = datetime.now()

    # Use the existing data preprocessing logic
    processed_data_point = preprocess_data(packet_data)

    # Ensure processed_data_point is a 2D array for the model
    if processed_data_point.ndim == 1:
        processed_data_point = processed_data_point.reshape(1, -1)

    # Use the pre-trained Isolation Forest model for anomaly detection
    if 'isolation_forest_model' in st.session_state and st.session_state.isolation_forest_model is not None:
        anomaly_score = st.session_state.isolation_forest_model.decision_function(processed_data_point)[0]
    else:
        st.warning("Isolation Forest model not available. Using dummy score for packet.")
        anomaly_score = np.random.rand() # Fallback to dummy score

    # Quantum enhancement
    # The quantum model is designed for batch processing, so we'll enhance single packet scores
    # by passing a single-row array. This might not be the most optimal use case for QAOA-inspired circuits
    # but demonstrates the integration.
    try:
        quantum_enhanced_score = st.session_state.quantum_detector.enhance_scores(
            processed_data_point,
            np.array([anomaly_score])
        )[0]
    except Exception as q_e:
        st.warning(f"Quantum enhancement failed for packet: {q_e}. Using classical score.")
        quantum_enhanced_score = anomaly_score

    end_detection_time = datetime.now()
    latency_ms = (end_detection_time - arrival_time).total_seconds() * 1000

    latency_log = {
        'timestamp': arrival_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
        'packet_summary': packet_summary,
        'latency_ms': latency_ms,
        'anomaly_score': anomaly_score # Include detection result
    }

    # Return the processed data point as a DataFrame for consistency
    processed_df = pd.DataFrame(processed_data_point, columns=[f'feature_{i}' for i in range(processed_data_point.shape[1])])
    processed_df['anomaly_score'] = quantum_enhanced_score # Add quantum enhanced anomaly score to the processed data
    return {'processed_data': processed_df, 'latency_log': latency_log}


# Main application logic
if MODE == 'replay':
    if PCAP_FILE:
        if TCPSPLIT_AVAILABLE:
            # Use tcpreplay if available and preferred
            replay_pcap_tcpreplay(PCAP_FILE)
            # For now, we'll still load sample data for the UI to function
            st.session_state.data = load_sample_data("CICIDS2017 (Subset)")
        elif SCAPY_AVAILABLE:
            replay_pcap_scapy(PCAP_FILE)
            if st.session_state.data is not None:
                # replay_pcap_scapy now returns the processed DataFrame directly
                st.session_state.processed_data = st.session_state.data
                # Latency logs are now handled within replay_pcap_scapy and displayed there
                st.markdown('<h3 class="sub-header">Replayed Data Preview</h3>', unsafe_allow_html=True)
                st.write(st.session_state.processed_data.head())
                # Set threshold after PCAP replay
                if 'initial_threshold' in st.session_state:
                    st.session_state.threshold = st.session_state.initial_threshold
                else:
                    # Fallback if initial_threshold is not set
                    st.session_state.threshold = -0.5 # A reasonable default for Isolation Forest scores
        else:
            st.error("Neither tcpreplay nor Scapy is available for PCAP replay.")
            st.session_state.data = load_sample_data("CICIDS2017 (Subset)") # Fallback
    else:
        st.error("No PCAP file specified for replay mode.")
        st.session_state.data = load_sample_data("CICIDS2017 (Subset)") # Fallback
elif MODE == 'live':
    if PYSHARK_AVAILABLE:
        st.info(f"Starting live capture on interface {LIVE_INTERFACE}...")
        # This is a simplified stub. In a real app, you'd run this in a separate thread
        # and continuously feed packets to your detection pipeline.
        try:
            capture = pyshark.LiveCapture(interface=LIVE_INTERFACE)
            st.write(f"Listening on {LIVE_INTERFACE}. Press Ctrl+C in terminal to stop.")
            # For demonstration, capture a few packets and process them
            live_packets_data = []
            live_latency_logs = []
            for i, packet in enumerate(capture.sniff_continuously(packet_count=10)):
                arrival_time = datetime.now()
                packet_summary = f"Live Packet {i+1}: {packet.summary}"
                # Convert pyshark packet to suitable format
                # This mapping is a simplification and might need further refinement
                protocol_type = packet.highest_layer.lower() if hasattr(packet, 'highest_layer') else 'unknown'
                service = 'unknown'
                if 'tcp' in packet:
                    service = packet.tcp.dstport if hasattr(packet.tcp, 'dstport') else 'unknown'
                    protocol_type = 'tcp'
                elif 'udp' in packet:
                    service = packet.udp.dstport if hasattr(packet.udp, 'dstport') else 'unknown'
                    protocol_type = 'udp'
                elif 'icmp' in packet:
                    protocol_type = 'icmp'

                src_bytes = int(packet.length) if hasattr(packet, 'length') else 0
                dst_bytes = 0 # Pyshark doesn't easily give dst_bytes directly for a single packet

                dummy_data = {
                    'duration': 0,
                    'protocol_type': protocol_type,
                    'service': service,
                    'flag': 'unknown',
                    'src_bytes': src_bytes,
                    'dst_bytes': dst_bytes,
                    'land': 0, 'wrong_fragment': 0, 'urgent': 0, 'hot': 0,
                    'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0,
                    'root_shell': 0, 'su_attempted': 0, 'num_root': 0,
                    'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
                    'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
                    'count': 1, 'srv_count': 1, 'serror_rate': 0.0,
                    'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0,
                    'same_srv_rate': 1.0, 'diff_srv_rate': 0.0,
                    'srv_diff_host_rate': 0.0, 'dst_host_count': 1,
                    'dst_host_srv_count': 1, 'dst_host_same_srv_rate': 1.0,
                    'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 1.0,
                    'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0,
                    'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0,
                    'dst_host_srv_rerror_rate': 0.0,
                    'label': 0
                }
                packet_df = pd.DataFrame([dummy_data])

                detection_result = process_packet(packet_df, packet_summary, arrival_time)
                live_packets_data.append(detection_result['processed_data'])
                live_latency_logs.append(detection_result['latency_log'])

                st.write(f"Processed live packet {i+1}. Latency: {detection_result['latency_log']['latency_ms']:.2f} ms")
                if i >= 9: # Stop after 10 packets for demo
                    break
            capture.close()
            if live_packets_data:
                # Concatenate all processed dataframes
                final_live_processed_df = pd.concat(live_packets_data, ignore_index=True)
                st.session_state.data = final_live_processed_df # Store raw-like data
                st.session_state.processed_data = final_live_processed_df # Store processed data
                st.write("Live Capture Latency Logs:", pd.DataFrame(live_latency_logs))
                st.markdown('<h3 class="sub-header">Live Captured Data Preview</h3>', unsafe_allow_html=True)
                st.write(st.session_state.processed_data.head())
                # Set threshold after live capture
                if 'initial_threshold' in st.session_state:
                    st.session_state.threshold = st.session_state.initial_threshold
                else:
                    # Fallback if initial_threshold is not set
                    st.session_state.threshold = -0.5 # A reasonable default for Isolation Forest scores
            else:
                st.warning("No packets captured in live mode.")

        except Exception as e:
            st.error(f"Error during live capture: {e}. Make sure you have necessary permissions (e.g., run with sudo or set capabilities).")
            st.session_state.data = load_sample_data("CICIDS2017 (Subset)") # Fallback
    else:
        st.error("Pyshark not available. Cannot perform live capture.")
        st.session_state.data = load_sample_data("CICIDS2017 (Subset)") # Fallback
else: # Default simulation mode
    # Data loading section for simulation mode
    data_source = st.sidebar.radio("Choose Data Source", ("Sample Data", "Upload CSV"))

    if data_source == "Sample Data":
        sample_dataset = st.sidebar.selectbox("Select Sample Dataset", list(SAMPLE_DATA_URLS.keys()))
        if st.sidebar.button("Load Sample Data"):
            with st.spinner(f"Loading {sample_dataset}..."):
                st.session_state.data = load_sample_data(sample_dataset)
                st.session_state.processed_data = None # Reset processed data
                st.success(f"Successfully loaded {sample_dataset}!")

    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            with st.spinner("Loading uploaded data..."):
                st.session_state.data = pd.read_csv(uploaded_file)
                st.session_state.processed_data = None # Reset processed data
                st.success("Successfully loaded uploaded CSV!")

# Main content area
# This section will now be more unified, as processed_data will be available in all modes
# after the initial setup (replay/live) or manual preprocessing (simulation).

# Display raw data if loaded (only in simulation mode, otherwise processed data is shown)
if MODE == 'simulation' and st.session_state.data is not None:
    st.markdown('<h2 class="sub-header">Raw Data Preview</h2>', unsafe_allow_html=True)
    st.write(st.session_state.data.head())
    st.write(f"Total records: {len(st.session_state.data)}")

    # Data preprocessing section (only in simulation mode)
    if st.session_state.processed_data is None:
        if st.sidebar.button("Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                st.session_state.processed_data = preprocess_data(st.session_state.data)
                st.success("Data preprocessing complete!")

# Model training and anomaly detection (applies to all modes if processed_data is available)
if st.session_state.processed_data is not None:
    st.markdown('<h2 class="sub-header">Anomaly Detection Model</h2>', unsafe_allow_html=True)

    # Isolation Forest Parameters
    contamination = st.sidebar.slider("Contamination (expected proportion of anomalies)", 0.01, 0.5, 0.1, 0.01)
    n_estimators = st.sidebar.slider("Number of Estimators (Isolation Forest trees)", 50, 500, 100, 10)

    # If in replay or live mode, anomaly scores are already calculated per packet
    if MODE != 'simulation':
        if 'anomaly_score' in st.session_state.processed_data.columns:
            st.session_state.anomaly_scores = st.session_state.processed_data['anomaly_score'].values
            # Use the initial threshold from the pre-trained model
            if 'initial_threshold' in st.session_state:
                st.session_state.threshold = st.session_state.initial_threshold
            else:
                # Fallback if initial_threshold is not set (shouldn't happen if model trained)
                st.session_state.threshold = np.mean(st.session_state.anomaly_scores) - 0.5 * np.std(st.session_state.anomaly_scores)
            st.success("Anomaly scores loaded from processed packets.")
        else:
            st.warning("Anomaly scores not found in processed data. Cannot proceed with visualization.")
            st.session_state.anomaly_scores = None
    else: # Simulation mode: run classical detection
        if st.sidebar.button("Run Classical Anomaly Detection (Isolation Forest)"):
            with st.spinner("Running Isolation Forest..."):
                iso_forest = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=42)
                iso_forest.fit(st.session_state.processed_data)
                st.session_state.anomaly_scores = iso_forest.decision_function(st.session_state.processed_data)
                st.session_state.threshold = iso_forest.threshold_
                st.success("Classical anomaly detection complete!")

    if st.session_state.anomaly_scores is not None:
                st.write(f"Isolation Forest Threshold: {st.session_state.threshold:.4f}")
                st.markdown('<h3 class="sub-header">Classical Anomaly Scores Distribution</h3>', unsafe_allow_html=True)
                fig_scores = plot_anomaly_scores(st.session_state.anomaly_scores, st.session_state.threshold)
                st.plotly_chart(fig_scores, use_container_width=True)

                # Quantum Enhancement Section
                st.markdown('<h3 class="sub-header">Quantum Enhancement</h3>', unsafe_allow_html=True)
                # Only allow quantum enhancement if there are anomaly scores to enhance
                if st.session_state.anomaly_scores is not None:
                    if st.sidebar.button("Run Quantum Enhancement"):
                        with st.spinner("Running Quantum Enhancement..."):
                            quantum_detector = QuantumAnomalyDetector()
                            # Ensure processed_data is a numpy array for quantum model
                            if isinstance(st.session_state.processed_data, pd.DataFrame):
                                # Drop the 'anomaly_score' column before passing to quantum model
                                data_for_quantum = st.session_state.processed_data.drop(columns=['anomaly_score'], errors='ignore').values
                            else:
                                data_for_quantum = st.session_state.processed_data

                            # Use a subset for quantum processing if data is too large
                            if data_for_quantum.shape[0] > 100:
                                st.warning("Quantum processing can be slow for large datasets. Using a subset of 100 samples.")
                                # Randomly sample 100 points for quantum enhancement
                                indices = np.random.choice(data_for_quantum.shape[0], 100, replace=False)
                                subset_data = data_for_quantum[indices]
                                subset_scores = st.session_state.anomaly_scores[indices]
                                st.session_state.quantum_enhanced_scores = quantum_detector.enhance_scores(subset_data, subset_scores)
                                # Map back to original size (simple placeholder: fill others with classical scores)
                                full_enhanced_scores = st.session_state.anomaly_scores.copy()
                                for i, original_idx in enumerate(indices):
                                    full_enhanced_scores[original_idx] = st.session_state.quantum_enhanced_scores[i]
                                st.session_state.quantum_enhanced_scores = full_enhanced_scores
                            else:
                                st.session_state.quantum_enhanced_scores = quantum_detector.enhance_scores(data_for_quantum, st.session_state.anomaly_scores)
                            st.success("Quantum enhancement complete!")

                if st.session_state.quantum_enhanced_scores is not None:
                    st.markdown('<h3 class="sub-header">Quantum-Enhanced Anomaly Scores Distribution</h3>', unsafe_allow_html=True)
                    fig_quantum_scores = plot_anomaly_scores(st.session_state.quantum_enhanced_scores, st.session_state.threshold, title="Quantum-Enhanced Anomaly Scores Distribution")
                    st.plotly_chart(fig_quantum_scores, use_container_width=True)

                    # Final Anomaly Classification
                    st.markdown('<h2 class="sub-header">Anomaly Classification Results</h2>', unsafe_allow_html=True)
                    final_scores = st.session_state.quantum_enhanced_scores
                    anomalies = final_scores < st.session_state.threshold
                    num_anomalies = np.sum(anomalies)



                # Confusion Matrix (if labels are available)
                if 'label' in st.session_state.data.columns:
                    st.markdown('<h3 class="sub-header">Confusion Matrix</h3>', unsafe_allow_html=True)
                    true_labels = st.session_state.data['label'].values
                    # Convert anomaly scores to binary predictions based on threshold
                    predicted_labels = (final_scores < st.session_state.threshold).astype(int)
                    # Ensure true_labels and predicted_labels are aligned and of same length
                    if len(true_labels) == len(predicted_labels):
                        fig_cm = plot_confusion_matrix(true_labels, predicted_labels)
                        st.plotly_chart(fig_cm, use_container_width=True)
                        st.write("Classification Report:")
                        st.code(classification_report(true_labels, predicted_labels))
    else:
        st.warning("Cannot display confusion matrix: Mismatch between true labels and predicted scores length.")
else:
    st.info("Upload data with a 'label' column to see Confusion Matrix and Classification Report.")

                # Feature Importance (if applicable and model supports it)
                # This part would depend on how you extract feature importance from your models
                # For Isolation Forest, you might look at feature contributions or build a separate explainer
                # For now, this is a placeholder.
                # st.markdown('<h3 class="sub-header">Feature Importance</h3>', unsafe_allow_html=True)
                # fig_fi = plot_feature_importance(feature_names, feature_importances)
                # st.plotly_chart(fig_fi, use_container_width=True)



# This section is for the original simulation mode's data loading and analysis trigger
# It should only be active if MODE is 'simulation' and processed_data is not yet available
if MODE == 'simulation' and st.session_state.processed_data is None:
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

# Model parameters (only for simulation mode when running analysis manually)
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

run_button = st.button("Run Analysis")

# Main content area for simulation mode's manual run
if st.session_state.data is not None and run_button:
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


st.write(f"Detected **{num_anomalies}** anomalies out of {len(final_scores)} records.")

    # Display anomalies
if st.session_state.data is not None:
    st.markdown('<h3 class="sub-header">Anomalous Records Preview</h3>', unsafe_allow_html=True)
    # Ensure original data has same index as scores
    original_data_with_scores = st.session_state.data.copy()
    original_data_with_scores['anomaly_score'] = final_scores
    original_data_with_scores['is_anomaly'] = anomalies
    st.write(original_data_with_scores[original_data_with_scores['is_anomaly']].head(10))

    # Confusion Matrix (if labels are available)
    if 'label' in st.session_state.data.columns:
                            st.markdown('<h3 class="sub-header">Confusion Matrix</h3>', unsafe_allow_html=True)
                            true_labels = st.session_state.data['label'].values
                            # Convert anomaly scores to binary predictions based on threshold
                            predicted_labels = (final_scores < st.session_state.threshold).astype(int)
                            # Ensure true_labels and predicted_labels are aligned and of same length
                            if len(true_labels) == len(predicted_labels):
                                fig_cm = plot_confusion_matrix(true_labels, predicted_labels)
                                st.plotly_chart(fig_cm, use_container_width=True)
                                st.write("Classification Report:")
                                st.code(classification_report(true_labels, predicted_labels))
                            else:
                                st.warning("Cannot display confusion matrix: Mismatch between true labels and predicted scores length.")
    else:
        st.info("Upload data with a 'label' column to see Confusion Matrix and Classification Report.")

            # Feature Importance (if applicable and model supports it)
            # This part would depend on how you extract feature importance from your models
            # For Isolation Forest, you might look at feature contributions or build a separate explainer
            # For now, this is a placeholder.
            # st.markdown('<h3 class="sub-header">Feature Importance</h3>', unsafe_allow_html=True)
            # fig_fi = plot_feature_importance(feature_names, feature_importances)
            # st.plotly_chart(fig_fi, use_container_width=True)

else:
    st.info("Please load data to proceed with anomaly detection.")
    
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