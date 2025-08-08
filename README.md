# Q-Predictor: Quantum-Enhanced Network Anomaly Detection

Q-Predictor is a full-stack web application that combines classical machine learning with quantum computing techniques to detect anomalies in network traffic data. The application uses Qiskit to implement quantum algorithms that enhance traditional anomaly detection methods.

## Features

- **Interactive Web Interface**: Upload and analyze network log files through a user-friendly Streamlit interface
- **Classical Anomaly Detection**: Isolation Forest algorithm to identify potential anomalies
- **Quantum Enhancement**: Quantum computing techniques to improve detection accuracy
- **Visualization**: Interactive charts and graphs to explore and understand results
- **Sample Datasets**: Built-in support for CICIDS2017 and KDD99 datasets

## Architecture

The application consists of the following components:

1. **Frontend**: Streamlit-based web interface for data upload and visualization
2. **Data Processing**: Preprocessing pipeline for network log data
3. **Classical Model**: Isolation Forest for initial anomaly detection
4. **Quantum Model**: Quantum algorithms (QAOA-inspired and Quantum Boltzmann Machine) to enhance detection
5. **Visualization**: Interactive plots and charts for result analysis

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/q-predictor.git
   cd q-predictor
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Use the application:
   - Upload a network log file (CSV format)
   - Or use one of the provided sample datasets
   - Adjust model parameters as needed
   - Run the analysis
   - Explore the results through visualizations

## Data Format

The application expects network log data in CSV format with features such as:

- Timestamp information
- Source/Destination IP addresses
- Protocol information
- Connection details (duration, bytes transferred, etc.)
- Service types
- Flag information

The sample datasets (CICIDS2017 and KDD99) are already formatted appropriately.

## Quantum Enhancement

The application uses two quantum computing approaches to enhance classical anomaly detection:

1. **Quantum-Enhanced Clustering**: Uses quantum kernels to improve clustering of normal vs. anomalous data points

2. **Quantum Boltzmann Machine (QBM)**: A simplified implementation that uses quantum circuits to model complex distributions and enhance anomaly scores

Both methods run on Qiskit's quantum simulators and do not require access to real quantum hardware.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Qiskit](https://qiskit.org/) - IBM's open-source quantum computing framework
- [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) - Intrusion Detection Evaluation Dataset
- [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) - Network Intrusion Detection Dataset