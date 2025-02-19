import os
import sys
import time
import argparse
import logging
import numpy as np

# Get the base directory from the environment variable
base_dir = os.getenv("PULSELINE_VER0_DIR")
if not base_dir:
    print("Error: PULSELINE_VER0_DIR environment variable is not set.")
    sys.exit(1)

# List of relative paths to add dynamically
relative_paths = [
    "input_file_dir_init/scripts",
    "SPOTLIGHT_PULSELINE/scripts",
]

# Loop through and add each path to sys.path
for relative_path in relative_paths:
    full_path = os.path.join(base_dir, relative_path)
    sys.path.insert(0, full_path)

# Import necessary functions
from beam_level_candidate_sifting import *
try:
    from read_input_file_dir import load_parameters
except ImportError as e:
    print("Error importing 'read_input_file_dir'. Ensure the script exists in the specified path.")
    print(e)
    sys.exit(1)

# Define configuration file path
config_file_path = os.path.join(base_dir, "input_file_dir_init/input_dir/input_file_directory.txt")

def final_stage_candidate_sifting():
    # Load configuration file for importing paths
    if not os.path.exists(config_file_path):
        print(f"Configuration file not found: {config_file_path}")
        sys.exit(1)

    try:
        params = load_parameters(config_file_path)
    except Exception as e:
        print(f"Error loading parameters from configuration file: {e}")
        sys.exit(1)

    # Extract parameters from config for running beam level sifting
    environ_init_script = params.get('environ_init_script')
    raw_input_dir = params.get('raw_input_dir')
    fil_output_dir = params.get('fil_output_dir') 
    pulseline_input_file_dir = params.get('pulseline_input_file_dir')
    pulseline_output_dir = params.get('pulseline_output_dir')
    pulseline_log_dir = params.get('pulseline_log_dir')

    # Path for getting beam sifting parameters
    beam_sifting_par_file_path = os.path.join(pulseline_input_file_dir, "pulseline_master.txt")
    
    # Loading the beam sifting parameter file
    try:
        sift_params = load_parameters(beam_sifting_par_file_path)
    except Exception as e:
        print(f"Error loading beam sifting parameters: {e}")
        sys.exit(1)

    # Extract additional parameters for sifting
    harmonic_opt_flag = sift_params.get('harmonic_opt_flag')
    period_tol_beam_sort = sift_params.get('period_tol_beam_sort', 0.1)
    min_beam_cut = sift_params.get('min_beam_cut', 2)

    # Executing the beam sifting function
    os.system(f"source {environ_init_script}")
    
    beam_level_candidate_sifting(
        pulseline_output_dir, pulseline_output_dir, raw_input_dir, fil_output_dir, 
        pulseline_input_file_dir, pulseline_log_dir, harmonic_opt_flag, period_tol_beam_sort, min_beam_cut
    )

# Run the function if the script is executed
if __name__ == "__main__":
    final_stage_candidate_sifting()