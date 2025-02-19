import os
import re
import shutil
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
]

# Loop through and add each path to sys.path
for relative_path in relative_paths:
    full_path = os.path.join(base_dir, relative_path)
    sys.path.insert(0, full_path)

# Import necessary functions
try:
    from read_input_file_dir import load_parameters
except ImportError as e:
    print("Error importing 'read_input_file_dir'. Ensure the script exists in the specified path.")
    print(e)
    sys.exit(1)
    

def extract_beam_id(filename):
    """
    Extracts the Beam ID (e.g., BM098) from a given filename.
    
    :param filename: The name of the file.
    :return: The extracted Beam ID or None if not found.
    """
    match = re.search(r"(BM\d+)", filename)
    if match:
        return match.group(1)  # Return the matched BM ID
    return None  # Return None if no match is found

def copy_ahdr_files(input_dir, output_dir):
    """
    Copies all .ahdr files from the input directory to the output directory.
    
    :param input_dir: Path to the input directory containing .ahdr files.
    :param output_dir: Path to the output directory where files will be copied.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # List all files in the input directory
        files_copied = 0
        for filename in os.listdir(input_dir):
            # Check if the file has an .ahdr extension
            if filename.lower().endswith('.ahdr'):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)
                
                # Copy the file
                shutil.copy(input_path, output_path)
                files_copied += 1
                print(f"Copied: {filename} -> {output_path}")

        # Final summary
        if files_copied == 0:
            print(f"No .ahdr files found in {input_dir}.")
        else:
            print(f"Successfully copied {files_copied} .ahdr files to {output_dir}.")

    except Exception as e:
        print(f"Error while copying files: {e}")


def extract_ra_dec_beam(input_dir, output_dir):
    """
    Extract RA, Dec, and Beam Index values from .ahdr files, construct a Beam ID, 
    and save them to a single output file.

    :param input_dir: Directory containing .ahdr files.
    :param output_dir: Directory where the output file will be saved.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Define the output file path
        output_file = os.path.join(output_dir, "Extracted_RA_Dec_beam_index.txt")
        
        # Open the output file in write mode
        with open(output_file, "w") as out_file:
            # Write the header line for clarity
            out_file.write("# Extracted data: RA, Dec, Beam Index, Beam ID\n")
            out_file.write("RA (Rad), Dec (Rad), Beam Index, Beam ID\n")
            
            # Track if any valid data was found
            data_found = False
            
            # Iterate through all files in the input directory
            for filename in os.listdir(input_dir):
                if filename.endswith(".ahdr"):
                    file_path = os.path.join(input_dir, filename)
                    print(f"Processing file: {filename}")
                    
                    try:
                        with open(file_path, "r") as file:
                            # Flag to indicate when to start processing beam data
                            processing_data = False
                            
                            for line in file:
                                # Skip lines until we find the line with RA, Dec, Beam Index data
                                if "RA" in line and "DEC" in line and "BM-Idx" in line:
                                    # Start processing after the header line
                                    processing_data = True
                                    continue  # Skip the header line itself
                                
                                # If we're processing data, process the line
                                if processing_data:
                                    # Skip lines that contain string data like Date or Time
                                    if any(word in line for word in ["Date", "Time", "GTAC"]):
                                        break  # Stop processing further lines as the data ends here

                                    # Split the line into columns (RA, Dec, Beam Index, etc.)
                                    columns = line.split()
                                    if len(columns) >= 4:
                                        try:
                                            # Extract the RA, Dec, Beam Index
                                            ra = columns[0]  # First column is RA
                                            dec = columns[1]  # Second column is Dec
                                            beam_index = columns[2]  # Third column is Beam Index
                                            beam_sub_index = columns[3]  # Fourth column is Beam Sub-Index
                                            beam_id = f"BM{beam_index}"  # Construct Beam ID (e.g., BM100)
                                            
                                            # Write the extracted data to the output file
                                            out_file.write(f"{ra}, {dec}, {beam_index}, {beam_id}\n")
                                            data_found = True
                                        except IndexError:
                                            print(f"Error: Incomplete data in line: {line}")
                                        
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
            
            if not data_found:
                print("No valid RA, Dec, Beam Index data found in the input files.")
        
        print(f"Extraction complete. Data saved to {output_file}")

    except Exception as e:
        print(f"Error: {e}")


def read_beam_data_to_array(input_dir):
    """
    Reads the extracted RA, Dec, Beam Index, and Beam ID data from the output file 
    and converts it into a NumPy array.

    :param input_dir: Directory containing the output file.
    :return: A NumPy array with four columns: RA (Rad), Dec (Rad), Beam Index, Beam ID.
    """
    # Define the file path by combining input_dir and output file name
    file_path = os.path.join(input_dir, "Extracted_RA_Dec_beam_index.txt")

    try:
        # Initialize a list to hold the data rows
        data = []

        with open(file_path, "r") as file:
            for line in file:
                # Skip header or comment lines
                if line.startswith("#") or "RA (Rad)" in line:
                    continue

                # Split the line into components
                parts = line.strip().split(", ")
                if len(parts) == 4:  # Ensure there are exactly four columns
                    ra = float(parts[0])         # Convert RA to float
                    dec = float(parts[1])        # Convert Dec to float
                    beam_index = int(parts[2])   # Convert Beam Index to int
                    beam_id = parts[3]           # Keep Beam ID as a string

                    # Append the row to the data list
                    data.append([ra, dec, beam_index, beam_id])

        # Convert the list of rows into a NumPy array
        # dtype=object is used since the last column (Beam ID) contains strings
        data_array = np.array(data, dtype=object)

        return data_array

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error while reading file: {e}")
        return None


def calculate_beam_distances(given_beam_id, beam_data_array):
    """
    Calculate the angular distance of each beam from a given beam 
    and return the Beam IDs ordered by distance.
    Handles edge cases such as invalid RA/Dec values and missing beam IDs.

    :param beam_data_array: NumPy array with columns [RA, Dec, Beam Index, Beam ID] (RA and Dec in radians).
    :param given_beam_id: Beam ID of the reference beam (e.g., 'BM001').
    :return: NumPy array of Beam IDs sorted by distance from the given beam.
    """
    try:
        # Ensure the beam data array has the correct number of columns (RA, Dec, Beam Index, Beam ID)
        if beam_data_array.shape[1] != 4:
            raise ValueError("Beam data array must have exactly 4 columns (RA, Dec, Beam Index, Beam ID).")

        # Find the row corresponding to the given beam ID
        given_beam_row = beam_data_array[beam_data_array[:, 3] == given_beam_id]
        
        if len(given_beam_row) == 0:
            raise ValueError(f"Beam ID '{given_beam_id}' not found in the data.")

        # Extract the RA and Dec of the given beam
        given_ra, given_dec = given_beam_row[0, 0], given_beam_row[0, 1]

        # Check if the RA and Dec values are within valid ranges
        if not (0 <= given_ra <= 2 * np.pi) or not (-np.pi / 2 <= given_dec <= np.pi / 2):
            raise ValueError(f"Invalid RA/Dec for given beam {given_beam_id}: RA = {given_ra}, Dec = {given_dec}.")

        # Extract all RA and Dec values
        all_ra = beam_data_array[:, 0].astype(float)
        all_dec = beam_data_array[:, 1].astype(float)

        # Check if the RA/Dec values in the entire array are within valid ranges
        if not np.all((0 <= all_ra) & (all_ra <= 2 * np.pi)):
            raise ValueError("Some RA values are out of the valid range [0, 2π].")
        if not np.all((-np.pi / 2 <= all_dec) & (all_dec <= np.pi / 2)):
            raise ValueError("Some Dec values are out of the valid range [-π/2, π/2].")

        # Calculate the angular distance using the Spherical Law of Cosines formula
        delta_ra = all_ra - given_ra
        cos_distance = np.sin(given_dec) * np.sin(all_dec) + np.cos(given_dec) * np.cos(all_dec) * np.cos(delta_ra)
        
        # Clip the cosine values to avoid numerical errors due to floating point precision (values should be between -1 and 1)
        cos_distance = np.clip(cos_distance, -1.0, 1.0)
        
        # Calculate the angular distance in radians
        distances = np.arccos(cos_distance)

        # Sort the indices of distances in ascending order
        sorted_indices = np.argsort(distances)

        # Return the Beam IDs in order of distance (including the given beam)
        sorted_beam_ids = beam_data_array[sorted_indices, 3]
        
        return sorted_beam_ids

    except Exception as e:
        print(f"Error: {e}")
        return None


def count_consecutive_unique_beams(consequtive_beam_ids, unique_beam_ids):
    """
    Counts how many beam IDs from `Unique_beam_ids` appear consecutively from 
    the start in `consequtive_beam_ids`.

    :param consequtive_beam_ids: NumPy array of beam IDs in sorted order.
    :param unique_beam_ids: NumPy array of unique beam IDs.
    :return: Integer count of consecutive matching elements from the start.
    """
    count = 0
    unique_beam_set = set(unique_beam_ids)  # Convert to set for faster lookup

    for beam_id in consequtive_beam_ids:
        if beam_id in unique_beam_set:
            count += 1
        else:
            break  # Stop as soon as we find a non-matching beam

    return count


def list_candidates(input_dir, file_name, harmonic_opt_flag):
    """
    Loads the candidates based on the harmonic_opt_flag and returns the reversed array.
    
    :param file_name: The base name for the candidates file.
    :param harmonic_opt_flag: The flag to determine which candidates file to load.
    :param input_dir: The directory where the input files are located.
    :return: Reversed candidate array or None if an error occurs.
    """
    # Determine the candidate file name based on flags
    if harmonic_opt_flag == 0.0:
        candidate_file = f"{file_name}_all_sifted_candidates.txt"
    elif harmonic_opt_flag == 1.0:
        candidate_file = f"{file_name}_all_sifted_harmonic_removed_candidates.txt"
    else:
        print("Invalid value of harmonic_opt_flag.")
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


def get_output_filename(beam_id, harmonic_opt_flag, output_dir, beam_ids, file_names):
    """
    Generates the output filename based on the beam_id and harmonic_opt_flag.

    :param beam_id: The beam ID for which the file is generated.
    :param harmonic_opt_flag: The flag to determine the type of candidate file.
    :param output_dir: The directory where the output file will be saved.
    :param beam_ids: List of beam IDs.
    :param file_names: List of corresponding file names.
    :return: The complete output file path as a string.
    """
    try:
        # Find the index of the beam_id in the beam_ids list
        if beam_id not in beam_ids:
            raise ValueError(f"Beam ID '{beam_id}' not found in the provided beam_ids list.")
        
        index = beam_ids.index(beam_id)
        base_file_name = file_names[index]

        # Determine the file suffix based on the harmonic_opt_flag
        if harmonic_opt_flag == 0.0:
            suffix = "_all_sifted_beam_sorted_candidates.txt"
        elif harmonic_opt_flag == 1.0:
            suffix = "_all_sifted_harmonic_removed_beam_sorted_candidates.txt"
        else:
            raise ValueError("Invalid harmonic_opt_flag. Must be 0.0 or 1.0.")

        # Construct the full output file path
        output_filename = os.path.join(output_dir, base_file_name + suffix)
        return output_filename
    
    except Exception as e:
        print(f"Error: {e}")
        return None


def remove_duplicate_candidates_from_beams(beam_ids, harmonic_opt_flag, output_dir, file_names):
    """
    Processes multiple beam IDs to remove duplicate candidates based on the first column (Period).
    This function generates output filenames for each beam ID and removes duplicate candidates.

    :param beam_ids: List of beam IDs to process.
    :param harmonic_opt_flag: Flag determining the type of candidate file.
    :param output_dir: Directory where the output files are saved.
    :param file_names: List of corresponding file names.
    """
    try:
        for beam_id in beam_ids:
            # Generate the output filename for the current beam_id
            output_filename = get_output_filename(beam_id, harmonic_opt_flag, output_dir, beam_ids, file_names)
            
            if output_filename:
                print(f"Processing file: {output_filename}")
                
                # Open the file, read its contents, and remove duplicates based on the first column
                with open(output_filename, 'r') as file:
                    lines = file.readlines()
                
                # Track unique candidates (using a set for the first column values)
                unique_candidates = set()
                unique_lines = []
                
                # Iterate through the lines, skipping the header
                for line in lines:
                    # Skip empty lines and header
                    if line.startswith("Period(sec)"):
                        unique_lines.append(line)
                        continue
                    
                    # Split the line by whitespace
                    columns = line.split()
                    
                    if len(columns) > 0:
                        period = columns[0]  # The first column is the Period (assumed unique identifier)
                        
                        # If this period hasn't been seen before, add it to the set and store the line
                        if period not in unique_candidates:
                            unique_candidates.add(period)
                            unique_lines.append(line)
                
                # Rewrite the file with unique candidates only
                with open(output_filename, 'w') as file:
                    file.writelines(unique_lines)
                
                print(f"Duplicate candidates removed. Unique candidates saved to {output_filename}")
            else:
                print(f"Error generating output filename for beam ID {beam_id}. Skipping...")

    except Exception as e:
        print(f"Error while processing beam IDs: {e}")




def beam_level_candidate_sifting(input_dir, output_dir, ahdr_input_dir, ahdr_output_dir, input_file_dir, log_dir, harmonic_opt_flag, period_tol_beam_sort, min_beam_cut):
    # Copies the adhr files for extrating beam information in further analysis
    copy_ahdr_files(ahdr_input_dir, ahdr_output_dir)
    
    # Extracts the beam information for further use in sorting.
    extract_ra_dec_beam(ahdr_output_dir, input_dir)

    # Process files to get the file name for processing the candidate files for sorting.
    files_to_process = [
        f for f in os.listdir(input_file_dir)
        if f.startswith("PULSELINE") and "node" in f and "gpu_id" in f and f.endswith(".txt")
    ]

    if not files_to_process:
        logging.info(f"No files found in {input_file_dir} matching the criteria. Skipping...")
        return


    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cand_len = []
    file_name_list = []
    beam_id_list = []
    for i, file in enumerate(sorted(files_to_process)):
        # Reload parameters from pulseline input file
        try:
            params = load_parameters(os.path.join(input_file_dir, file))
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
        beam_id = extract_beam_id(file_name)

        # Load candidates based on harmonic flag
        candidate_array = list_candidates(input_dir, file_name, harmonic_opt_flag)
        if candidate_array is None:
            print("No candidates to process. Exiting...")
            return

        cand_len.append(len(candidate_array[:, 0]))
        file_name_list.append(file_name)
        beam_id_list.append(beam_id)

    max_cand_len = max(cand_len)
    if max_cand_len == 0:
        print("No candidates found in any beam data. Exiting.")
        return
    
    # Initialize arrays for candidate parameters
    Period_array = np.full((len(files_to_process), max_cand_len), np.nan)
    Period_dot_array = np.full((len(files_to_process), max_cand_len), np.nan)
    DM_array = np.full((len(files_to_process), max_cand_len), np.nan)
    SNR_array = np.full((len(files_to_process), max_cand_len), np.nan)

    for i in range(len(file_name_list)):
        candidate_array = list_candidates(input_dir, file_name_list[i], harmonic_opt_flag)
        if candidate_array is None:
            print("No candidates to process. Exiting...")
            return

        cand_array_len = len(candidate_array[:, 0])

        for j in range(cand_array_len):
            Period_array[i, j] = candidate_array[j, 0]
            Period_dot_array[i, j] = candidate_array[j, 1]
            DM_array[i, j] = candidate_array[j, 2]
            SNR_array[i, j] = candidate_array[j, 3]

    # Unique period filtering
    Periods_flatten = Period_array.flatten()
    Filtered_period_list0 = [x for x in np.unique(Periods_flatten) if str(x) != 'nan']
    #print(Filtered_period_list0)
    tot_cand = len(Filtered_period_list0)
    unique_period_list0 = []
    Period_tol_array = []

    print(f"Filtered unique period list length: {len(Filtered_period_list0)}")

    while len(Filtered_period_list0) > 0:
        indices = np.where(
            Filtered_period_list0 <= Filtered_period_list0[0] + Filtered_period_list0[0] * (period_tol_beam_sort / 100.0)
        )
        unique_period_list0.append(Filtered_period_list0[0])
        Period_tol_array.append(Filtered_period_list0[0] * (period_tol_beam_sort / 100.0))
        Filtered_period_list0 = np.delete(Filtered_period_list0, indices[0])


    # Read the beam information (RA, Dec, beam_index, beam_id)
    beam_information_array = read_beam_data_to_array(input_dir)
    print(beam_information_array)

    # Initiallise all the output files by looping through the beam_information_array
    for beam_info in beam_information_array:
        beam_id = beam_info[3]  # Extract beam ID from the fourth column
        
        # Call the get_output_filename function to generate the output filename
        output_filename = get_output_filename(beam_id, harmonic_opt_flag, output_dir, beam_id_list, file_name_list)
        
        if output_filename:
            try:
                # Open the output file in write mode and create it if it doesn't exist
                with open(output_filename, 'w') as output_file:
                    # Write the header to the file
                    output_file.write("Period(sec)   Pdot(s/s)  DM(pc/cc)   SNR\n")
                    print(f"Header written to {output_filename}")
            except Exception as e:
                print(f"Error writing to file {output_filename}: {e}")

    # Now the sorting begins
    for i, uniq_period in enumerate(unique_period_list0):
        if i == 0:
            index = np.where(Period_array <= uniq_period + Period_tol_array[i] / 2)
        else:
            index = np.where((Period_array > unique_period_list0[i - 1] + Period_tol_array[i - 1] / 2) &
                             (Period_array <= uniq_period + Period_tol_array[i] / 2))

        print(index[0], index[1])
        Beam_id_index, cand_index = index[0], index[1]
        Beam_ids = np.array(beam_id_list)[Beam_id_index]
        unique_beam_id_index = np.unique(Beam_id_index)
        unique_beam_ids = np.array(beam_id_list)[unique_beam_id_index]
        # print(Beam_id_index, cand_index)
        # print(unique_beam_ids)

        for j in range(0, len(unique_beam_ids)):

            # Now calculate the consequtive beams using beam distance as parameter
            consequtive_beam_ids = calculate_beam_distances(unique_beam_ids[j], beam_information_array)

            # Now see for each beam id, in how many consequtive beams the canddiate is present
            count = count_consecutive_unique_beams(consequtive_beam_ids, unique_beam_ids)
            
            # Initialising the SNR array for further analysis
            Filtered_SNR_array = np.full(SNR_array.shape, np.nan)

            if count > min_beam_cut:
                # Find indices of beams in `consequtive_beam_ids` that exist in `Beam_ids`
                Beam_cand_indices = np.isin(Beam_ids, consequtive_beam_ids[:count])

                # Get corresponding indices
                valid_beam_indices = Beam_id_index[Beam_cand_indices]
                valid_cand_indices = cand_index[Beam_cand_indices]
                # print(valid_beam_indices)
                # print(valid_cand_indices)

                # Update the `Filtered_SNR_array` using valid indices
                Filtered_SNR_array[valid_beam_indices, valid_cand_indices] = SNR_array[valid_beam_indices, valid_cand_indices]
                
                # Find the maximum SNR candidate out of the sorted canddiates
                maxima_index = np.unravel_index(np.nanargmax(Filtered_SNR_array), Filtered_SNR_array.shape)
                print(maxima_index)
                # Get teh output file name
                output_file = get_output_filename(beam_id_list[maxima_index[0]], harmonic_opt_flag, output_dir, beam_id_list, file_name_list)

                # Write in output file
                with open(output_file, "a") as file:
                    file.write(f"{Period_array[maxima_index]:.10f}     "
                                f"{Period_dot_array[maxima_index]:.6e}     "
                                f"{DM_array[maxima_index]}     "
                                f"{SNR_array[maxima_index]:.2f}\n")
                print(f"Candidate: Period={Period_array[maxima_index]:.6f}, "
                        f"Pdot={Period_dot_array[maxima_index]:.6e}, "
                        f"SNR={SNR_array[maxima_index]:.2f}")
                        
        print("Finished processing all candidates and writing the output.")

    # Remove duplicate candidates from the beam filtered generated output files
    remove_duplicate_candidates_from_beams(beam_id_list, harmonic_opt_flag, output_dir, file_name_list)

    print("Finished removing duplicate candidates, now proceeding towrads folding.")




    

        


        

        



