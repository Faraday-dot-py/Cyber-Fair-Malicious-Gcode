# G-Code Feature Extraction and Malicious Modification Detection

## Overview

This project aims to analyze G-code files used for 3D printing by extracting key features and detecting potentially malicious modifications. The script processes two directories: one containing "good" G-code files (valid and safe) and another with "malicious" G-code files (intentionally modified to exhibit dangerous behavior). The extracted features are saved into a CSV file for further analysis by AI.

## Features

- **Feature Extraction**: Extracts various features from G-code files, including:
  - Maximum extruder and bed temperatures
  - Maximum movement limits on the X, Y, and Z axes
  - Maximum movement speed and extrusion values
  - Fan speeds
  - Counts of commands and out-of-bounds movements
  
- **Malicious Modification Simulation**: Modifies G-code files to introduce malicious behaviors, such as:
  - Setting dangerously high temperatures
  - Exceeding movement limits
  - Turning off cooling fans

- **Progress Tracking**: Displays a progress bar during file processing for user feedback.

- **AI Malicious Code Detection**: A multi-layer perceptron, or MLP is trained to recognize anomalies in gcode files based on several key traits. 

## Applications
The MLP trained off this data can be used to alert server admins that someone is trying to upload a potentially malicious gcode file, or to deny the upload in the first place. This protects the 3D printers from potential damage by bad actors

## Prerequisites

- Python 3.x
- Required libraries:
  - `pandas`
  - `tqdm`

You can install the required libraries using pip:

```bash
pip install pandas tqdm
```

## Directory Structure
*Note: Raw gcode used for training is not in github due to size and privacy concerns*

```
├ Home/
├── src/
├──── data/
├────── good_gcode/              # Directory containing safe G-code files
├────── malicious_gcode/         # Directory for modified malicious G-code
|
├──── model/
├────── gcode_malicious_detection_model.keras    #Trained model file
├────── scaler.pkl               # Saved scaler file
|
├──── demo.ipymb                 # Demo jupyter notebook for the fair
├──── mal.py                     # Used to generate malicious gcode
├──── model.py                   # Where the model is trained
├──── preprocess.py              # Extract significant features to csv
├── demo/
├──── cube.gcode                 # A "good" gcode file for demo
├──── malicious_cube.gcode       # A corrupted gcode file for demo
```

## Usage

1. Place your valid G-code files in the `good_gcode` directory.
2. Run the script to process the good G-code files and generate malicious G-code files:

   ```bash
   # Generate bad gcode files
   python mal.py
   ```

   ```bash
   # Perform feature extraction on gcode
   python preprocess.py
   ```

3. The script will extract features from both good and malicious G-code files and save them in `gcode_features.csv` for analysis.

## Functionality Overview

### Feature Extraction

The script extracts features from each G-code file, including temperatures, movements, and command counts. It checks for:
- Temperature commands (`M104`, `M109`, `M140`, `M190`)
- Movement commands (`G0`, `G1`)
- Fan speed commands (`M106`)

### Malicious G-code Modification

The script modifies G-code lines randomly based on a set of predefined behaviors, simulating potential malicious modifications that could occur in a compromised G-code file.

# Made In Association With:
![MS Logo|300](https://github.com/Faraday-dot-py/Cyber-Fair-Malicious-Gcode/blob/main/logos/MS%20Vertical%20White.png?raw=true)
![SIIL Logo](https://github.com/Faraday-dot-py/Cyber-Fair-Malicious-Gcode/blob/main/logos/CPP-SIIL%20Vert-Horiz-cropped.png?raw=true)



## BSD 3-Clause License

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions, and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the author nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

**DISCLAIMER:**

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
