import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib

# Load the extracted feature data
data = pd.read_csv('gcode_features.csv')

# Split features and labels
X = data.drop('label', axis=1)  # Features
y = data['label']  # Labels (0 for good, 1 for malicious)

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize (normalize) the feature data for better performance of the neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model using TensorFlow/Keras
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train_scaled.shape[1],)),  # Input layer
    tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
    tf.keras.layers.Dense(64, activation='relu'),   # Hidden layer with 64 neurons and ReLU activation
    tf.keras.layers.Dense(32, activation='relu'),   # Hidden layer with 32 neurons and ReLU activation
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer (1 neuron for binary classification)
])

# Compile the model (binary classification task)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 20

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

# Save the model
model.save('gcode_malicious_detection_model.keras')
# model.export('gcode_malicious_detection_model.pb')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')