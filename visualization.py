import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

def plot_anomaly_scores(classical_scores, quantum_scores=None, threshold=-0.5):
    """Plot the distribution of anomaly scores.
    
    Args:
        classical_scores (np.ndarray): Anomaly scores from classical model
        quantum_scores (np.ndarray, optional): Quantum-enhanced anomaly scores
        threshold (float): Threshold for anomaly detection
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    # Add classical scores histogram
    fig.add_trace(go.Histogram(
        x=classical_scores,
        name='Classical Scores',
        opacity=0.7,
        marker_color='blue',
        nbinsx=50
    ))
    
    # Add quantum scores if available
    if quantum_scores is not None:
        fig.add_trace(go.Histogram(
            x=quantum_scores,
            name='Quantum-Enhanced Scores',
            opacity=0.7,
            marker_color='purple',
            nbinsx=50
        ))
    
    # Add threshold line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Anomaly Threshold",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        title="Distribution of Anomaly Scores",
        xaxis_title="Anomaly Score (lower = more anomalous)",
        yaxis_title="Count",
        barmode='overlay',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    return fig

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix for anomaly detection results.
    
    Args:
        y_true (np.ndarray): True labels (0 for normal, 1 for anomaly)
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create heatmap
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Normal', 'Anomaly'],
        y=['Normal', 'Anomaly'],
        text_auto=True,
        color_continuous_scale='Blues'
    )
    
    # Update layout
    fig.update_layout(
        title="Confusion Matrix",
        height=400,
        width=400
    )
    
    return fig

def plot_feature_importance(model, feature_names):
    """Plot feature importance for anomaly detection.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): Names of features
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Check if model has feature_importances_
    if not hasattr(model, 'feature_importances_'):
        return None
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Create bar chart
    fig = px.bar(
        x=sorted_importances[:20],  # Show top 20 features
        y=sorted_feature_names[:20],
        orientation='h',
        labels={'x': 'Importance', 'y': 'Feature'},
        title='Top 20 Feature Importances'
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def plot_time_series_anomalies(df, time_column, score_column, is_anomaly_column):
    """Plot time series data with anomalies highlighted.
    
    Args:
        df (pd.DataFrame): DataFrame with time series data
        time_column (str): Name of the time/date column
        score_column (str): Name of the anomaly score column
        is_anomaly_column (str): Name of the binary anomaly indicator column
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Sort by time
    df_sorted = df.sort_values(time_column)
    
    # Create figure
    fig = go.Figure()
    
    # Add normal points
    normal_df = df_sorted[df_sorted[is_anomaly_column] == 0]
    fig.add_trace(go.Scatter(
        x=normal_df[time_column],
        y=normal_df[score_column],
        mode='markers',
        name='Normal',
        marker=dict(color='blue', size=6, opacity=0.6)
    ))
    
    # Add anomaly points
    anomaly_df = df_sorted[df_sorted[is_anomaly_column] == 1]
    fig.add_trace(go.Scatter(
        x=anomaly_df[time_column],
        y=anomaly_df[score_column],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=10, symbol='x')
    ))
    
    # Update layout
    fig.update_layout(
        title="Anomaly Scores Over Time",
        xaxis_title="Time",
        yaxis_title="Anomaly Score",
        height=500,
        hovermode="closest"
    )
    
    return fig

def plot_3d_clusters(X_reduced, anomaly_scores, threshold):
    """Create a 3D scatter plot of data points colored by anomaly score.
    
    Args:
        X_reduced (np.ndarray): Dimensionality-reduced data (3 dimensions)
        anomaly_scores (np.ndarray): Anomaly scores
        threshold (float): Threshold for anomaly detection
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': X_reduced[:, 0],
        'y': X_reduced[:, 1],
        'z': X_reduced[:, 2],
        'anomaly_score': anomaly_scores,
        'is_anomaly': (anomaly_scores < threshold).astype(int)
    })
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        df,
        x='x',
        y='y',
        z='z',
        color='anomaly_score',
        color_continuous_scale='Viridis_r',  # Reversed so darker = more anomalous
        symbol='is_anomaly',
        symbol_map={0: 'circle', 1: 'x'},
        size=[8 if a == 1 else 5 for a in df['is_anomaly']],
        opacity=0.8,
        title="3D Visualization of Anomalies"
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        scene=dict(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3'
        )
    )
    
    return fig

def plot_quantum_circuit(circuit):
    """Create a visualization of a quantum circuit.
    
    Args:
        circuit: Qiskit QuantumCircuit object
        
    Returns:
        matplotlib.figure.Figure: Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Draw circuit
    circuit.draw(output='mpl', ax=ax)
    
    # Update layout
    plt.tight_layout()
    
    return fig