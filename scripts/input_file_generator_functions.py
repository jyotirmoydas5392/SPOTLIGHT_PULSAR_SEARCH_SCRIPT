import numpy as np
import subprocess

def generate_input_files(input_file_generator_dir):
    """
    Generate input files.
    """
    print("Generating input files .....")
    try:
        # Run the input_file_generator script and pass the directory as an argument
        subprocess.run(
            ["python3", f"{input_file_generator_dir}/input_file_generator.py", input_file_generator_dir],
            check=True,
        )
        print("Input files generated successfully.")

    except subprocess.CalledProcessError as e:
        print(f"Error generating input files: {e}")
        sys.exit(1)