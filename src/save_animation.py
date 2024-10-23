import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import joblib
import pandas as pd
import networkx as nx

# Load the trained model
model = tf.keras.models.load_model(r'C:\Users\awebb\Documents\Programming\Work\SIIL\Cyber Fair Malicious Gcode\src\model_files\gcode_malicious_detection_model.keras')

# Load the scaler
scaler = joblib.load(r'C:\Users\awebb\Documents\Programming\Work\SIIL\Cyber Fair Malicious Gcode\src\model_files\scaler.pkl')

# Load the G-code features as a pandas dataframe
gcode_features = pd.read_csv(r'C:\Users\awebb\Documents\Programming\Work\SIIL\Cyber Fair Malicious Gcode\src\data\gcode_features.csv')

def get_gcode_features(i):
    return pd.DataFrame(gcode_features.iloc[i, :]).drop('label').T

class LayerActivations(tf.keras.Model):
    def __init__(self, model):
        super(LayerActivations, self).__init__()
        self.model = model
        self.layer_outputs = []

    def call(self, inputs):
        self.layer_outputs = [inputs]  # Include the input layer
        x = inputs
        for layer in self.model.layers:
            x = layer(x)
            self.layer_outputs.append(x)
        return x

def get_layer_outputs(model, input_data):
    activation_model = LayerActivations(model)
    activation_model(input_data)
    return activation_model.layer_outputs

def draw_neural_network(layer_sizes, layer_activations, ax):
    G = nx.Graph()
    node_colors = []
    pos = {}
    node_labels = {}

    # Normalize activations across all layers
    all_activations = np.concatenate([act.numpy().flatten() for act in layer_activations])
    min_activation, max_activation = np.min(all_activations), np.max(all_activations)

    for i, (size, activations) in enumerate(zip(layer_sizes, layer_activations)):
        layer_activations_flat = activations.numpy().flatten()
        for j in range(size):
            node_id = f'{i}_{j}'
            G.add_node(node_id)
            pos[node_id] = (i, j - size / 2)

            # Normalize activation and set color
            activation = layer_activations_flat[j]
            normalized_activation = (activation - min_activation) / (max_activation - min_activation)
            node_colors.append(plt.cm.viridis(normalized_activation))

            # Set node label (activation value)
            node_labels[node_id] = f'{activation:.2f}'

        if i > 0:
            for j in range(size):
                for k in range(layer_sizes[i-1]):
                    G.add_edge(f'{i-1}_{k}', f'{i}_{j}')

    ax.clear()
    nx.draw(G, pos, node_color=node_colors, node_size=1000, with_labels=False, arrows=False)
    nx.draw_networkx_labels(G, pos, node_labels, font_size=8)
    ax.set_title('Neural Network Visualization')
    ax.axis('off')

# Prepare data for animation
all_layer_outputs = []

for i in range(len(gcode_features)):
    # Extract features from the G-code file
    features = get_gcode_features(i)

    # Preprocess the features
    features_scaled = scaler.transform(features)

    # Convert to tensor
    input_tensor = tf.convert_to_tensor(features_scaled, dtype=tf.float32)

    # Get layer outputs
    layer_outputs = get_layer_outputs(model, input_tensor)

    # Save the layer outputs
    all_layer_outputs.append(layer_outputs)

# Set up the figure and axis for animation
fig, ax = plt.subplots(figsize=(12, 8))

# Create the animation function
def update(frame):
    draw_neural_network([features_scaled.shape[1], 64, 32, 1], all_layer_outputs[frame], ax)

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(all_layer_outputs), repeat=False)

# Save the animation to a file
ani.save('neural_network_activation_animation.gif', writer='ffmpeg', fps=5)  # You can change 'fps' to desired value

# Display a message after saving
print("Animation saved as 'neural_network_activation_animation.gif'")
