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
from aa_output_rename import *
from search_level_candidate_sifting import *
from harmonic_optimization import *
try:
    from read_input_file_dir import load_parameters
except ImportError as e:
    print("Error importing 'read_input_file_dir'. Ensure the script exists in the specified path.")
    print(e)
    sys.exit(1)

# Define configuration file path
config_file_path = os.path.join(base_dir, "input_file_dir_init/input_dir/input_file_directory.txt")


def first_stage_candidate_sifting(file, node_alias, gpu_id):
    # Load configuration file for importing paths
    if not os.path.exists(config_file_path):
        print(f"Configuration file not found: {config_file_path}")
        sys.exit(1)

    try:
        params = load_parameters(config_file_path)
    except Exception as e:
        print(f"Error loading parameters from configuration file: {e}")
        sys.exit(1)

    # Extract parameters from config
    environ_init_script = params.get('environ_init_script')
    aa_output_dir = params.get('aa_output_dir')
    pulseline_input_file_dir = params.get('pulseline_input_file_dir')
    pulseline_input_dir = params.get('pulseline_input_dir')
    pulseline_output_dir = params.get('pulseline_output_dir')
    pulseline_log_dir = params.get('pulseline_log_dir')

    # PULSELINE string setup
    PULSELINE = "PULSELINE"

    # Generate pulseline file name
    pulseline_file_name = f"{PULSELINE}_{file.replace('AA_2dhs_', '').strip()}"
    print(f"Pulseline file path: {os.path.join(pulseline_input_file_dir, pulseline_file_name)}")
    
    # Check if pulseline input file exists
    if not os.path.exists(os.path.join(pulseline_input_file_dir, pulseline_file_name)):
        print(f"Pulseline input file not found: {os.path.join(pulseline_input_file_dir, pulseline_file_name)}")
        sys.exit(1)

    # Reload parameters from pulseline input file
    try:
        params = load_parameters(os.path.join(pulseline_input_file_dir, pulseline_file_name))
    except Exception as e:
        print(f"Error loading parameters from configuration file: {e}")
        sys.exit(1)
    
    # Extract and print 'fil_file' from params
    fil_file_path = params.get('fil_file')
    if fil_file_path:
        fil_file = os.path.basename(fil_file_path)
        print(f"Extracted fil file name: {fil_file}")
    else:
        print("fil_file parameter not found in the loaded parameters.")
    file_name = fil_file.replace(".fil", "")

    # Define rename paths
    rename_input_path = os.path.join(aa_output_dir, "output", file.replace(".txt", ""))
    rename_output_path = os.path.join(pulseline_input_dir, file.replace(".txt", ""))

    # Perform AA output renaming
    aa_output_rename(rename_input_path, rename_output_path, fil_file, params.get('harmonic_sum_flag'))

    # Set up parameters for candidate sifting
    sifting_input_path = os.path.join(pulseline_input_dir, file.replace(".txt", ""))
    sifting_output_path = pulseline_output_dir

    # Run candidate sifting
    search_level_sift_candidates(sifting_input_path, sifting_output_path, file_name, 
                                  params.get('start_DM'), params.get('end_DM'), 
                                  params.get('low_period'), params.get('high_period'),
                                  params.get('dm_step'), params.get('DM_filtering_cut_10'),
                                  params.get('DM_filtering_cut_1000'), params.get('SNR_cut'),
                                  params.get('period_tol_init_sort'))

    # Handle harmonic optimization based on flag
    if params.get('harmonic_opt_flag') == 1:
        harmonic_optimization(pulseline_output_dir, pulseline_output_dir, file_name, params.get('period_tol_harm'))
    elif params.get('harmonic_opt_flag') == 0:
        print("Skipping harmonic optimization.")
    else:
        print("Invalid harmonic flag in input file.")
        sys.exit(1)

    print(f"CPU function for first stage sorting executed successfully for file {file}.")


if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(description="Run the PULSELINE function with specified parameters.")
    parser.add_argument("file", type=str, help="Input file name to process.")
    parser.add_argument("node_alias", type=str, help="Node alias where the function is running.")
    parser.add_argument("gpu_id", type=int, help="GPU ID to use for processing.")

    # Parse arguments and run the function
    args = parser.parse_args()
    first_stage_candidate_sifting(args.file, args.node_alias, args.gpu_id)
