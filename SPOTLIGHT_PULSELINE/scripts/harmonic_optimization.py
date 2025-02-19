import numpy as np
import os


def is_harmonically_related(a, b, tolerance):
    # Check if a and b are harmonically related (i.e., ratio is close to an integer)
    ratio = a / b
    return np.abs(ratio - round(ratio)) < tolerance

def remove_harmonics(arr, period_tol_harm):
    index = []
    harm_cand = []
    
    for i in range(len(arr)):
        harmonically_related = False
        
        harm_cand0 = []
        for j in range(i):
            
            tolerance = arr[i] * period_tol_harm / (100 * arr[j])
            
            if is_harmonically_related(arr[i], arr[j], tolerance):
                harmonically_related = True
                
                break
            
        if harmonically_related:
            index.append(i)
            harm_cand0.append(arr[j])
            harm_cand0.append(arr[i])
            harm_cand.append(harm_cand0)
    
    return harm_cand


def harmonic_optimization(input_dir, output_dir, file_name, period_tol_harm):
    """
    Perform harmonic filtering on the candidate list and save the results.

    Args:
        file_name (str): The base name of the file to process (without extension).
        period_tol_harm (float): The tolerance for identifying harmonics.
        input_dir (str): The directory where input files are stored (absolute path).
        output_dir (str): The directory where output files will be saved (absolute path).
    """
    # Construct the full path for the input file
    input_file_path = os.path.join(input_dir, file_name + "_all_sifted_candidates.txt")

    # Load the data from the file
    data = np.loadtxt(input_file_path, dtype=str)
    data = np.delete(data, 0, 0)
    
    A = np.array(data, dtype=float)
    A1 = A[::-1]
    
    # Perform harmonic filtering
    harmonic_candidates = remove_harmonics(A1[:, 0], period_tol_harm)
    
    # Sort harmonic candidates by the fundamental period
    sorted_fundamental_and_harmonic_candidates = np.array(sorted(harmonic_candidates, key=lambda x: x[0]))
    fundamental_candidates = sorted(list(set(sorted_fundamental_and_harmonic_candidates[:, 0])))
    print(sorted_fundamental_and_harmonic_candidates)
    
    harmonic_len = []
    harmonic_candidate_index = []
    for i in range(0, len(fundamental_candidates)):
        harmonic_cand_index_temp = []
        X0 = np.where(sorted_fundamental_and_harmonic_candidates[:, 0] == fundamental_candidates[i])
        
        X1 = np.where(A1[:, 0] == fundamental_candidates[i])
        harmonic_cand_index_temp.append(X1[0][0])
        
        for j in range(0, len(X0[0])):
            X2 = np.where(A1[:, 0] == sorted_fundamental_and_harmonic_candidates[:, 1][X0[0][j]])
            harmonic_cand_index_temp.append(X2[0][0])
            
        harmonic_len.append(len(harmonic_cand_index_temp))
        harmonic_candidate_index.append(harmonic_cand_index_temp)
    
    max_harmonic_len = np.max(harmonic_len)
    harmonic_index_array = np.empty((int(len(harmonic_candidate_index)), int(max_harmonic_len)))
    harmonic_index_array[:] = np.nan
    
    for i in range(int(len(harmonic_candidate_index))):
        for j in range(int(len(harmonic_candidate_index[i]))):
            harmonic_index_array[i][j] = harmonic_candidate_index[i][j]
    
    flatten_harmonic_index_array = np.array([x for x in np.unique(harmonic_index_array.flatten()) if str(x) != 'nan'], dtype=int)
    
    A2 = np.delete(A1, flatten_harmonic_index_array, axis=0)
    A2 = list(A2)
    
    for i in range(int(len(harmonic_candidate_index))):
        harmonic_SNR_array = np.empty(int(len(A1[:, 0])))
        harmonic_SNR_array[:] = np.nan
        
        for j in range(int(len(harmonic_candidate_index[i]))):
            harmonic_SNR_array[harmonic_candidate_index[i][j]] = A1[:, 3][harmonic_candidate_index[i][j]]
            
        X0 = np.where(harmonic_SNR_array == np.nanmax(harmonic_SNR_array))
        A2.append(A1[X0[0][0]])
        
    A2 = np.array(sorted(A2, key=lambda x: x[0]))
    A2 = A2[::-1]

    # Construct the output file path
    output_file_path = os.path.join(output_dir, file_name + "_all_sifted_harmonic_removed_candidates.txt")
    
    # Open the output file in write mode and write the header
    with open(output_file_path, "w") as file:
        file.write("Period(sec)   Pdot(s/s)  DM(pc/cc)   SNR\n")

        # Iterate over the A2 array and write each row of data
        for row in A2:
            # Ensure proper formatting of the values with space separation
            file.write(f"{row[0]:.10f}     {row[1]:.6e}     {row[2]:.2f}     {row[3]:.2f}\n")

    # Print confirmation after saving the results
    print(f"Results saved to {output_file_path}")