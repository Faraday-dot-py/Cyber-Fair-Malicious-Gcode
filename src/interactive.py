import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the trained model
model = tf.keras.models.load_model(r'C:\Users\awebb\Documents\Programming\Work\SIIL\Cyber Fair Malicious Gcode\src\model_files\gcode_malicious_detection_model.keras')

# Load the scaler
scaler = joblib.load(r'C:\Users\awebb\Documents\Programming\Work\SIIL\Cyber Fair Malicious Gcode\src\model_files\scaler.pkl')

def extract_features_from_gcode(gcode_file):
    # This function should implement your feature extraction logic
    # For this example, we'll just create dummy features
    return pd.DataFrame(np.random.rand(1, 10))  # Adjust the number of features as needed

class LayerActivations(tf.keras.Model):
    def __init__(self, model):
        super(LayerActivations, self).__init__()
        self.model = model
        self.layer_outputs = []

    def call(self, inputs):
        self.layer_outputs = [inputs]
        x = inputs
        for layer in self.model.layers:
            x = layer(x)
            self.layer_outputs.append(x)
        return x

def get_layer_outputs(model, input_data):
    activation_model = LayerActivations(model)
    activation_model(input_data)
    return activation_model.layer_outputs

def draw_interactive_neural_network(layer_outputs):
    edge_x = []
    edge_y = []
    node_x = []
    node_y = []
    node_text = []
    node_colors = []

    # Normalize activations across all layers
    all_activations = np.concatenate([act.numpy().flatten() for act in layer_outputs])
    min_activation, max_activation = np.min(all_activations), np.max(all_activations)

    layer_y_positions = []

    for i, activations in enumerate(layer_outputs):
        layer_size = activations.shape[1]
        layer_y = np.linspace(-layer_size/2, layer_size/2, layer_size)
        layer_y_positions.append(layer_y)
        node_x.extend([i] * layer_size)
        node_y.extend(layer_y)
        
        layer_activations_flat = activations.numpy().flatten()
        for j, activation in enumerate(layer_activations_flat):
            normalized_activation = (activation - min_activation) / (max_activation - min_activation)
            node_colors.append(normalized_activation)
            node_text.append(f"Layer {i}, Neuron {j}<br>Activation: {activation:.4f}")

        if i > 0:
            prev_layer_size = layer_outputs[i-1].shape[1]
            current_layer_y = layer_y_positions[i]
            prev_layer_y = layer_y_positions[i-1]
            for j in range(layer_size):
                for k in range(prev_layer_size):
                    edge_x.extend([i-1, i, None])
                    edge_y.extend([prev_layer_y[k], current_layer_y[j], None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            reversescale=True,
            color=node_colors,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Neuron Activation',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_trace .text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Neural Network Visualization',
                        titlefont_size=16,
                 showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20,l=5,r=5,t=40),
                         annotations=[
                             dict(
                                 text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs'> https://plotly.com/ipython-notebooks/network-graphs</a>",
                                 showarrow=False,
                                 xref="paper", yref="paper",
                                 x=0.005, y=-0.002 ) ],
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    fig.show()

# Main execution
gcode_file = r'C:\Users\awebb\Documents\Programming\Work\SIIL\Cyber Fair Malicious Gcode\demo\cube.gcode'  # Replace with the path to your G-code file

# Extract features from the G-code file
features = extract_features_from_gcode(gcode_file)

# Preprocess the features
features_scaled = scaler.transform(features)

# Convert to tensor
input_tensor = tf.convert_to_tensor(features_scaled, dtype=tf.float32)

# Get layer outputs
layer_outputs = get_layer_outputs(model, input_tensor)

# Draw the interactive neural network
draw_interactive_neural_network(layer_outputs)