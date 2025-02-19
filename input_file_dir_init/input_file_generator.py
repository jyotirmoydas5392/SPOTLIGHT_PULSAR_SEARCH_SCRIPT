import os
import sys
import numpy as np

# Insert custom script directory to the system path
sys.path.insert(0, '/lustre_archive/spotlight/data/pulsar_search_pipeline_ver0/input_file_dir_init/scripts')

try:
    from read_input_file_dir import load_parameters
except ImportError as e:
    print("Error importing 'read_input_file_dir'. Ensure the script exists in the specified path.")
    print(e)
    sys.exit(1)

# Path to the configuration file
config_file_path = "/lustre_archive/spotlight/data/pulsar_search_pipeline_ver0/input_file_dir_init/input_dir/input_file_directory.txt"

# Check if the configuration file exists
if not os.path.exists(config_file_path):
    print(f"Configuration file not found at: {config_file_path}")
    sys.exit(1)

# Load parameters from the configuration file
try:
    params = load_parameters(config_file_path)
except Exception as e:
    print("Error loading parameters from the configuration file.")
    print(e)
    sys.exit(1)

# Extract directory paths from parameters
aa_input_file_dir = params.get('aa_input_file_dir')
aa_input_dir = params.get('aa_input_dir')
aa_output_dir = params.get('aa_output_dir')

pulseline_input_file_dir = params.get('pulseline_input_file_dir')
pulseline_input_dir = params.get('pulseline_input_dir')
pulseline_output_dir = params.get('pulseline_output_dir')

avail_gpus_file_dir = params.get('avail_gpus_file_dir')

# Get the available GPU nodes and all filterbank files that are to be processed
avail_gpu_nodes = np.loadtxt(os.path.join(avail_gpus_file_dir, 'avail_gpu_nodes.txt'), dtype=str)
fil_files = sorted(f for f in os.listdir(aa_input_dir) if f.endswith('.fil'))  # Filter only .fil files
num_fil_files = len(fil_files)
num_nodes = len(avail_gpu_nodes)

# Helper function to delete pre-generated input files
def clear_directory(directory, file_prefix):
    """
    Deletes all files in the directory starting with the given file_prefix.
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    for filename in os.listdir(directory):
        if filename.startswith(file_prefix):
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"Deleted pre-generated file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

# Helper function to generate input files
def generate_input_files(input_file_dir, master_input_file, file_prefix, file_suffix, placeholder, replacement, is_aa):
    toggle = 0
    node_index = 0

    for i, fil_file in enumerate(fil_files):
        gpu_id = toggle
        gpu_node = avail_gpu_nodes[node_index]
        abs_filterbank_path = os.path.join(aa_input_dir, fil_file)
        
        # Construct the new input file name
        new_file_name = f"{file_prefix}_{fil_file.split('.fil')[0]}_node_{gpu_node}_gpu_id_{gpu_id}{file_suffix}"
        new_file_path = os.path.join(input_file_dir, new_file_name)
        
        try:
            # Read and modify the master template
            with open(os.path.join(input_file_dir, master_input_file), 'r') as template_file:
                content = template_file.read()
            
            # Update placeholders for the filterbank path
            updated_content = content.replace(placeholder, replacement.format(abs_filterbank_path))
            
            # For AA files, update the GPU ID
            if is_aa:
                updated_content = updated_content.replace('selected_card_id 0', f'selected_card_id {gpu_id}')
            
            # Write the modified content to the new file
            with open(new_file_path, 'w') as new_file:
                new_file.write(updated_content)
            
            print(f"Generated input file at: {new_file_path}")

        except Exception as e:
            print(f"Error processing {fil_file}: {e}")

        # Toggle GPU ID and move to the next node if necessary
        toggle = 1 - toggle
        if toggle == 0:
            node_index = (node_index + 1) % num_nodes

# Clear old input files before generating new ones
clear_directory(aa_input_file_dir, 'AA_2dhs')
clear_directory(pulseline_input_file_dir, 'PULSELINE')

# Generate AA input files
generate_input_files(
    input_file_dir=aa_input_file_dir,
    master_input_file='aa_2dhs_master.txt',
    file_prefix='AA_2dhs',
    file_suffix='.txt',
    placeholder='file /test/Test.fil',
    replacement='file {}',
    is_aa=True
)

# Generate Pulseline input files
generate_input_files(
    input_file_dir=pulseline_input_file_dir,
    master_input_file='pulseline_master.txt',
    file_prefix='PULSELINE',
    file_suffix='.txt',
    placeholder='Test.fil',
    replacement='{}',
    is_aa=False
)
