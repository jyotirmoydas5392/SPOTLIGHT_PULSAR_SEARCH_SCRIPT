import os
import glob

def aa_output_rename(input_dir, output_dir, file_name, harmonic_sum_flag):
    """
    Renames or copies files in the input directory that match the pattern "*harm*dat" 
    and saves them in the output directory with a new name based on the DM value.

    :param input_path: Path to the directory containing the original files.
    :param output_path: Path to the directory where renamed/copied files will be saved.
    :param file_name: Base name for the renamed files.
    """
    # Ensure output path exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all files matching the pattern in the input directory
    if harmonic_sum_flag == 0:
        files = [f for f in glob.glob(os.path.join(input_dir, "*dat")) if "harm" not in f]
    elif harmonic_sum_flag == 1:
        files = glob.glob(os.path.join(input_dir, "*harm*dat"))
    else:
        print("Define the harmobnic_sum_flag parameter correctly.")

    if not files:
        print("No matching files found.")
        return

    # Process each file
    for file in files:
        try:
            # Extract DM value from the filename
            if harmonic_sum_flag == 0:
                DM_value = "{:.2f}".format(float(file.split("list_")[1].split(".dat")[0]))
            elif harmonic_sum_flag == 1:
                DM_value = "{:.2f}".format(float(file.split("harm_")[1].split(".dat")[0]))

            # Construct the new file name
            new_file_name = f"{file_name.replace('.fil', '')}_DM{DM_value}.dat"
            new_file_path = os.path.join(output_dir, new_file_name)

            # Copy the file with the new name to the output directory
            os.system(f"cp {file} {new_file_path}")
            print(f"Copied {file} to {new_file_path}")

        except (IndexError, ValueError) as e:
            print(f"Error processing file {file}: {e}")

