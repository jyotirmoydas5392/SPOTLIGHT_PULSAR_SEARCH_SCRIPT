import os
import sys
import time
import argparse
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s]: %(message)s',
                    handlers=[logging.StreamHandler()])

# Get the base directory from the environment variable
base_dir = os.getenv("PULSELINE_VER0_DIR")
if not base_dir:
    logging.error("PULSELINE_VER0_DIR environment variable is not set.")
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
try:
    from candidate_folding import *
    from batch_convert_ps_to_png import *
    from read_input_file_dir import load_parameters
except ImportError as e:
    logging.error("Error importing required modules. Ensure the scripts exist in the specified paths.")
    logging.error(e)
    sys.exit(1)

# Define configuration file path
config_file_path = os.path.join(base_dir, "input_file_dir_init/input_dir/input_file_directory.txt")


def beam_level_candidate_folding(file, node_alias, gpu_id):
    """
    Handles candidate folding operations and PNG generation for a given file.

    :param file: Input file name to process.
    :param node_alias: Alias of the node where processing is performed.
    :param gpu_id: ID of the GPU to use for processing.
    """
    # Load configuration file for importing paths
    if not os.path.exists(config_file_path):
        logging.error(f"Configuration file not found: {config_file_path}")
        sys.exit(1)

    try:
        params = load_parameters(config_file_path)
    except Exception as e:
        logging.error(f"Error loading parameters from configuration file: {e}")
        sys.exit(1)

    # Extract parameters from config
    environ_init_script = params['environ_init_script']
    pulseline_input_file_dir = params['pulseline_input_file_dir']
    pulseline_output_dir = params['pulseline_output_dir']
    pulseline_log_dir = params['pulseline_log_dir']

    # PULSELINE string setup
    PULSELINE = "PULSELINE"

    # Generate pulseline file name
    pulseline_file_name = f"{PULSELINE}_{file.replace('AA_2dhs_', '').strip()}"
    logging.info(f"Pulseline file path: {os.path.join(pulseline_input_file_dir, pulseline_file_name)}")

    # Check if pulseline input file exists
    pulseline_file_path = os.path.join(pulseline_input_file_dir, pulseline_file_name)
    if not os.path.exists(pulseline_file_path):
        logging.error(f"Pulseline input file not found: {pulseline_file_path}")
        sys.exit(1)

    # Reload parameters from pulseline input file
    try:
        params = load_parameters(pulseline_file_path)
    except Exception as e:
        logging.error(f"Error loading parameters from pulseline file: {e}")
        sys.exit(1)

    # Extract and validate 'fil_file'
    fil_file_path = params.get('fil_file')
    if not fil_file_path:
        logging.error("The 'fil_file' parameter is missing from the configuration.")
        sys.exit(1)

    fil_file = os.path.basename(fil_file_path)
    logging.info(f"Extracted fil file name: {fil_file}")
    file_name = fil_file.replace(".fil", "")

    # Set up parameters for candidate folding
    folded_outputs = "folded_outputs/"  # Define the subdirectory name
    folding_input_path = pulseline_output_dir
    folding_output_path = os.path.join(pulseline_output_dir, folded_outputs)

    # Create the output directory if it doesn't exist
    os.makedirs(folding_output_path, exist_ok=True)

    # Run candidate folding
    try:
        folding_operation(
            folding_input_path, folding_output_path, fil_file_path, file_name,
            params.get('workers_per_node'), params.get('harmonic_opt_flag'), params.get('beam_sort_flag'),
            params.get('fold_soft'), params.get('fold_type')
        )
        logging.info(f"Folding completed successfully for file {file_name}.")
    except Exception as e:
        logging.error(f"Error during folding operation: {e}")
        sys.exit(1)

    # Generate PNG files from the folded PS files
    input_ps_files_path = folding_output_path
    output_png_files_path = folding_output_path

    try:
        batch_convert_ps_to_png(
            input_ps_files_path, output_png_files_path,
            params.get('worker_per_node'), keyword=file_name
        )
        logging.info("PNG files created successfully for all folded PS files.")
    except Exception as e:
        logging.error(f"Error during PNG conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(description="Run the PULSELINE function with specified parameters.")
    parser.add_argument("file", type=str, help="Input file name to process.")
    parser.add_argument("node_alias", type=str, help="Node alias where the function is running.")
    parser.add_argument("gpu_id", type=int, help="GPU ID to use for processing.")

    # Parse arguments and run the function
    args = parser.parse_args()
    beam_level_candidate_folding(args.file, args.node_alias, args.gpu_id)