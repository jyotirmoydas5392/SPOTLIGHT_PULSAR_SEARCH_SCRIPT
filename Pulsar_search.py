import os
import re
import sys
import time
import logging
from pathlib import Path
import subprocess
import numpy as np
from multiprocessing import Process

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

# Get the base directory from the environment variable
base_dir = os.getenv("PULSELINE_VER0_DIR")
if not base_dir:
    logging.error("Error: PULSELINE_VER0_DIR environment variable is not set.")
    sys.exit(1)

# Add required paths to sys.path
required_paths = [
    "input_file_dir_init/scripts",
    "SPOTLIGHT_PULSELINE/scripts",
    "input_file_dir_init",
    "SPOTLIGHT_PULSELINE",
    "raw_to_filterbank",
    "scripts"
]
for relative_path in required_paths:
    full_path = os.path.join(base_dir, relative_path)
    if os.path.exists(full_path):
        sys.path.insert(0, full_path)
        logging.info(f"Added to sys.path: {full_path}")
    else:
        logging.warning(f"Path does not exist: {full_path}")

# Import required modules
try:
    from read_input_file_dir import load_parameters  # Import load_parameters
    from raw_to_fil import *  # Import raw_to_fil functions
    from input_file_generator_functions import *
    from multi_gpu_multi_node_functions import *
    from raw_to_fil_functions import *
    logging.info("Modules imported successfully.")
except ImportError as e:
    logging.error("Error importing required modules.", exc_info=True)
    sys.exit(1)

# Define configuration file path and input file generator script path
config_file_path = os.path.join(base_dir, "input_file_dir_init/input_dir/input_file_directory.txt")
input_file_generator_dir = os.path.join(base_dir, "input_file_dir_init")

# Check if configuration file exists
if not os.path.exists(config_file_path):
    logging.error(f"Configuration file not found: {config_file_path}")
    sys.exit(1)


def main():
    """
    Main function to execute astro-accelerate jobs on available GPU nodes.
    """

    # Step 1: Load configuration
    if not os.path.exists(config_file_path):
        print(f"Configuration file not found: {config_file_path}")
        sys.exit(1)

    try:
        params = load_parameters(config_file_path)
    except Exception as e:
        print(f"Error loading parameters from configuration file: {e}")
        sys.exit(1)

    # Extract parameters
    aa_executable_file_dir = params.get('aa_executable_file_dir')
    aa_input_file_dir = params.get('aa_input_file_dir')
    aa_output_dir = params.get('aa_output_dir')
    avail_gpus_file_dir = params.get('avail_gpus_file_dir')
    aa_log_dir = params.get('aa_log_dir')
    environ_init_script = params.get('environ_init_script')

    gpu_0_start_delay = int(params.get('gpu_0_start_delay', 5))  # Default: 5 seconds
    gpu_1_start_delay = int(params.get('gpu_1_start_delay', 10))  # Default: 10 seconds
    file_processing_delay = int(params.get('file_processing_delay', 5))  # Default: 5 seconds

    #first_stage_candidate_sifting module runner path
    first_stage_sifting_path = params.get('first_stage_sifting_path')

    #beam level sifting flag
    beam_level_sifting = params.get('beam_level_sifting')

    #final_stage_candidate_sifting module runner path
    final_stage_sifting_path = params.get('final_stage_sifting_path')
    #beam_level_candidate_sifting module runner path
    beam_level_folding_path = params.get('beam_level_folding_path')

    #Input dara type flag
    data_type = params.get('data_type')

    # Step 2: Generte input data if needed
    if data_type == 0:
        process_raw_files(config_file_path)


    # Step 3: Generate input files if needed
    generate_input_files(input_file_generator_dir)

    # Step 4: Validate directories
    required_dirs = [aa_executable_file_dir, aa_input_file_dir, aa_output_dir]
    for directory in required_dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Created missing directory: {directory}")
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")
        else:
            print(f"Directory already exists: {directory}")


    # Step 5: Load available GPU nodes
    try:
        avail_gpu_nodes = np.loadtxt(
            os.path.join(avail_gpus_file_dir, 'avail_gpu_nodes.txt'), dtype=str
        )
    except Exception as e:
        print(f"Error reading available GPU nodes: {e}")
        sys.exit(1)

    if not isinstance(avail_gpu_nodes, (list, np.ndarray)) or len(avail_gpu_nodes) == 0:
        print("No available GPU nodes found in the file.")
        sys.exit(1)

    # Step 6: Construct the SSH commands for GPU processing and same node CPU processing for search, shift, fold, etc.

    command_template_1 = (
        "ssh -X {node_alias} \"source {environ_init_script} && "
        "source {aa_executable_file_dir}/environment.sh && "
        "bash {aa_executable_file_dir}/astro-accelerate.sh {input_file_path} {output_dir} "
        "> {log_dir}/gpu_log_{node_alias}_gpu_{gpu_id}.log 2>&1 && "
        "python3 {first_stage_sifting_path} {file} {node_alias} {gpu_id} "
        ">> {log_dir}/cpu_log_{node_alias}_gpu_{gpu_id}.log 2>&1\""
    )

    command_template_2 = (
        "ssh -X {node_alias} \"source {environ_init_script} && "
        "python3 {beam_level_folding_path} {file} {node_alias} {gpu_id} "
        ">> {log_dir}/cpu_log_{node_alias}_gpu_{gpu_id}.log 2>&1\""
    )

    # Step 7: Process each node concurrently for the command template 1 for running AA pulsar search and CPU first stage sorting
    processes = []
    for node_alias in avail_gpu_nodes:
        p = Process(
            target=process_node,
            args=(
                node_alias, aa_input_file_dir, aa_output_dir, aa_log_dir, gpu_0_start_delay, gpu_1_start_delay, file_processing_delay, command_template_1)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Step 8: Process all the search level sifted candidates output at once to do a beam level candidate filtration
    if beam_level_sifting == 1:
        os.system(f"python3 {final_stage_sifting_path}")

    # Step 9: Process each node concurrently for the command template 2 for multinode CPU folding
    processes = []
    for node_alias in avail_gpu_nodes:
        p = Process(
            target=process_node,
            args=(
                node_alias, aa_input_file_dir, aa_output_dir, aa_log_dir, gpu_0_start_delay, gpu_1_start_delay, file_processing_delay, command_template_2)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
