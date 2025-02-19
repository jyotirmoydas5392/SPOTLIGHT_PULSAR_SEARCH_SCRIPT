import os
import numpy as np
from multiprocessing import Pool


# Define the function to execute system commands
def folding(command):
    """
    Executes the folding operation via a system call.
    :param command: Command string to execute.
    """
    try:
        os.system(command)
    except Exception as e:
        print(f"Error during folding operation: {e}")

# Function to load candidates based on the harmonic_opt_flag and beam_sort_flag
def candidates(input_dir, file_name, harmonic_opt_flag, beam_sort_flag):
    """
    Loads the candidates based on the harmonic_opt_flag and beam_sort_flag, and returns the reversed array.
    
    :param file_name: The base name for the candidates file.
    :param harmonic_opt_flag: The flag to determine which candidates file to load.
    :param beam_sort_flag: The flag for beam sorting.
    :param input_dir: The directory where the input files are located.
    :return: Reversed candidate array or None if an error occurs.
    """
    # Determine the candidate file name based on flags
    if harmonic_opt_flag == 0.0 and beam_sort_flag == 0.0:
        candidate_file = f"{file_name}_all_sifted_candidates.txt"
    elif harmonic_opt_flag == 0.0 and beam_sort_flag == 1.0:
        candidate_file = f"{file_name}_all_sifted_beam_sorted_candidates.txt"
    elif harmonic_opt_flag == 1.0 and beam_sort_flag == 0.0:
        candidate_file = f"{file_name}_all_sifted_harmonic_removed_candidates.txt"
    elif harmonic_opt_flag == 1.0 and beam_sort_flag == 1.0:
        candidate_file = f"{file_name}_all_sifted_harmonic_removed_beam_sorted_candidates.txt"
    else:
        print("Invalid combination of harmonic_opt_flag and beam_sort_flag.")
        return None

    # Load the candidate file and reverse the array
    try:
        file_path = os.path.join(input_dir, candidate_file)
        data = np.loadtxt(file_path, dtype=str, skiprows=1)  # Skip header
        A = np.array(data, dtype=float)  # Convert to float array
        return A[::-1]  # Reverse the array
    except Exception as e:
        print(f"Error loading candidate file {candidate_file}: {e}")
        return None

# Folding operation function
def folding_operation(input_dir, output_dir, fil_file_path, file_name, workers, harmonic_opt_flag, beam_sort_flag, fold_soft, fold_type):
    """
    Executes the folding operation based on candidate data and fold type.
    :param input_dir: The directory where the input files are located.
    :param output_dir: The directory where output files will be saved.
    :param fil_file_path: Path to the .fil file for folding.
    :param file_name: The base name for output files (without extension).
    :param workers: Number of workers for parallel execution.
    :param harmonic_opt_flag: Flag to select the type of candidate data.
    :param fold_soft: Flag to select folding software type (0.0 for prepfold, 1.0 for custom).
    :param fold_type: Flag to select the type of folding (0.0 for dat, 1.0 for fil, 2.0 for both).
    """
    print(f"Folding operation initiated for file: {file_name}")

    # Validate input parameters
    if fold_soft not in [0.0, 1.0]:
        raise ValueError("Invalid fold_soft value. Please use 0.0 or 1.0.")
    if fold_type not in [0.0, 1.0, 2.0]:
        raise ValueError("Invalid fold_type value. Please use 0.0, 1.0, or 2.0.")

    # Load candidates based on harmonic flag
    candidate_array = candidates(input_dir, file_name, harmonic_opt_flag, beam_sort_flag)
    if candidate_array is None:
        print("No candidates to process. Exiting...")
        return

    # Extract candidate data
    Folding_period = candidate_array[:, 0]
    Folding_period_dot = candidate_array[:, 1]
    Folding_DM = candidate_array[:, 2]

    # Prepare folding commands
    dat_folding_strings = []
    fil_folding_strings = []

    num_candidates = len(Folding_period)
    for i in range(num_candidates):
        DM_value = f"{Folding_DM[i]:.2f}"  # Format DM_value to 2 decimal places
        period_value = f"{Folding_period[i]:.10f}"  # Format Folding_period to 10 decimal places
        period_dot_value = f"{Folding_period_dot[i]:.6f}"  # Format Folding_period_dot to 6 decimal places

        # Prepare the dat folding command
        dat_folding_strings.append(
            f"prepfold -p {period_value} -pd {period_dot_value} -dm {DM_value} "
            f"-noxwin -zerodm -nosearch -o {file_name}_DM{DM_value}_DAT {input_dir}/{file_name}_DM{DM_value}.dat"
        )

        # Prepare the fil folding command
        fil_folding_strings.append(
            f"prepfold -p {period_value} -pd {period_dot_value} -dm {DM_value} "
            f"-noxwin -zerodm -nosearch -o {file_name}_DM{DM_value}_FIL {fil_file_path}"
        )

    def execute_folding(commands):
        """
        Executes folding commands in parallel within the same output directory.
        :param commands: List of command strings to execute.
        """
        original_dir = os.getcwd()
        try:
            os.chdir(output_dir)
            print(f"Logged into directory: {output_dir}")
            with Pool(workers) as pool:
                pool.map(folding, commands)
        finally:
            os.chdir(original_dir)
            print(f"Returned to directory: {original_dir}")

    # Execute folding commands based on fold type
    if fold_soft == 0.0:
        if fold_type == 0.0:
            print(f"Executing dat folding for {num_candidates} candidates.")
            execute_folding(dat_folding_strings)
        elif fold_type == 1.0:
            print(f"Executing fil folding for {num_candidates} candidates.")
            execute_folding(fil_folding_strings)
        elif fold_type == 2.0:
            print(f"Executing dat and fil folding for {num_candidates} candidates.")
            execute_folding(dat_folding_strings)
            execute_folding(fil_folding_strings)
    elif fold_soft == 1.0:
        print("Custom folding software to be implemented.")

    print("Folding operation completed.")