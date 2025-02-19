import os
import re
import sys
import subprocess
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load base directory from environment variable
base_dir = os.getenv("PULSELINE_VER0_DIR")
if not base_dir:
    logging.error("Error: PULSELINE_VER0_DIR environment variable is not set.")
    sys.exit(1)

# Add required paths to sys.path
required_paths = [
    "input_file_dir_init/scripts",
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
    logging.info("Modules imported successfully.")
except ImportError as e:
    logging.error("Error importing required modules.", exc_info=True)
    sys.exit(1)

def process_raw_files(config_path):
    # Load parameters
    params = load_parameters(config_path)
    input_dir = params.get("raw_input_dir")
    output_dir = params.get("fil_output_dir")
    nbeams = params.get("nbeams_per_raw", 10)
    njobs = params.get("num_jobs", -1)
    raw_to_fil_runner_path = params.get("raw_to_fil_runner_path")

    # Ensure the directories exist
    if not Path(input_dir).exists():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Find all matching `.raw.<number>` files with their absolute paths
    pattern = re.compile(r".*\.raw\.\d+$")
    raw_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and pattern.match(f)
    ]

    if not raw_files:
        raise ValueError("No matching `.raw.<number>` files found in the input directory.")

    # Build the command
    command = [
        "python3",
        raw_to_fil_runner_path,
        *raw_files,  # Unpack the raw_files list into separate arguments
        "-n", str(njobs),
        "-b", str(nbeams),
        "-o", output_dir
    ]

    # Execute the command
    try:
        subprocess.run(command, check=True)
        print("Processing completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during processing: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")