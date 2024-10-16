import os
import pandas as pd
import re
from tqdm import tqdm  # Import tqdm for progress bar

# Define the maximum limits (you can adjust these based on your printer specs)
MAX_X = 220  # Max X-axis limit
MAX_Y = 220  # Max Y-axis limit
MAX_Z = 250  # Max Z-axis limit
MAX_TEMP = 260  # Safe extruder temp limit
MAX_BED_TEMP = 120  # Safe bed temp limit
MAX_SPEED = 5000  # Safe movement speed limit

# Directories containing the good and malicious G-code files
good_gcode_dir = "good_gcode"
malicious_gcode_dir = 'malicious_gcode'

# Function to extract features from a single G-code file
def extract_gcode_features(file_path, label):
    features = {
        'temp_extruder': 0,
        'temp_bed': 0,
        'max_x': 0,
        'max_y': 0,
        'max_z': 0,
        'max_speed': 0,
        'max_extrusion': 0,
        'fan_speed': 0,
        'num_commands': 0,
        'out_of_bounds_moves': 0,
        'label': label  # 0 for good, 1 for malicious
    }

    with open(file_path, 'r') as f:
        for line in f:
            features['num_commands'] += 1

            # Extract temperature commands
            if line.startswith('M104') or line.startswith('M109'):
                temp = extract_value_from_line(line, 'S')
                if temp:
                    features['temp_extruder'] = max(features['temp_extruder'], float(temp))
            
            if line.startswith('M140') or line.startswith('M190'):
                temp = extract_value_from_line(line, 'S')
                if temp:
                    features['temp_bed'] = max(features['temp_bed'], float(temp))
            
            # Extract movement commands
            if line.startswith('G0') or line.startswith('G1'):
                x = extract_value_from_line(line, 'X')
                y = extract_value_from_line(line, 'Y')
                z = extract_value_from_line(line, 'Z')
                f = extract_value_from_line(line, 'F')
                e = extract_value_from_line(line, 'E')

                if x: features['max_x'] = max(features['max_x'], float(x))
                if y: features['max_y'] = max(features['max_y'], float(y))
                if z: features['max_z'] = max(features['max_z'], float(z))
                if f: features['max_speed'] = max(features['max_speed'], float(f))
                if e: features['max_extrusion'] = max(features['max_extrusion'], float(e))

                # Check for out-of-bounds movements
                if (x and float(x) > MAX_X) or (y and float(y) > MAX_Y) or (z and float(z) > MAX_Z):
                    features['out_of_bounds_moves'] += 1
            
            # Extract fan speed
            if line.startswith('M106'):
                fan_speed = extract_value_from_line(line, 'S')
                if fan_speed:
                    features['fan_speed'] = max(features['fan_speed'], float(fan_speed))

    return features

# Helper function to extract a parameter value (e.g., X, Y, Z, S) from a G-code line
def extract_value_from_line(line, param):
    match = re.search(rf"{param}([-\d.]+)", line)
    if match:
        return match.group(1)
    return None

# Function to process all G-code files in a directory and return a feature dataframe
def process_gcode_files(directory, label):
    feature_list = []
    gcode_files = [filename for filename in os.listdir(directory) if filename.endswith('.gcode')]

    # Add progress bar for file processing
    for filename in tqdm(gcode_files, desc=f"Processing G-code files in {directory}"):
        # print(f"Processing file: {filename}")
        try:
            if filename.endswith('.gcode'):
                file_path = os.path.join(directory, filename)
                features = extract_gcode_features(file_path, label)
                feature_list.append(features)
        except Exception as e:
            print(f"Error processing file '{filename}': {str(e)}")

    return pd.DataFrame(feature_list)

# Process good and malicious G-code files
good_gcode_df = process_gcode_files(good_gcode_dir, label=0)
malicious_gcode_df = process_gcode_files(malicious_gcode_dir, label=1)

# Combine good and malicious G-code data into a single dataset
gcode_data = pd.concat([good_gcode_df, malicious_gcode_df], ignore_index=True)

# Save the dataset to a CSV file (for future use)
gcode_data.to_csv('gcode_features.csv', index=False)

print("Feature extraction complete. Data saved to 'gcode_features.csv'.")
