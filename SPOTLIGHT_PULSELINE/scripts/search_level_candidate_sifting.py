import numpy as np
import shutil
import math
import os

def consecutive(data, stepsize=1):
    """Identify consecutive groups in an array."""
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)

def prepare_dm_array(start_DM, end_DM, dm_step):
    """Prepare DM array based on start, end, and step."""
    N0 = int(math.floor((end_DM - start_DM) / dm_step))
    dm1 = ["{:.2f}".format(start_DM + i * dm_step) for i in range(N0)]
    DM_array = np.array(dm1, dtype=str)
    return DM_array

def search_level_sift_candidates(input_dir, output_dir, file_name, start_DM, end_DM, low_period, high_period, 
            dm_step, DM_filtering_cut_10, DM_filtering_cut_1000, SNR_cut, period_tol_init_sort):
    """
    Process the candidate data and filter based on the configuration parameters.
    
    Args:
        input_dir (str): The directory where input files are stored.
        output_dir (str): The directory where output files will be saved.
        file_name (str): The base name of the .fil file to process (without extension).
        start_DM (float): The starting DM value.
        end_DM (float): The ending DM value.
        low_period (float): The minimum period in milliseconds.
        high_period (float): The maximum period in milliseconds.
        dm_step (float): The step size for DM.
        DM_filtering_cut (int): The cutoff for DM filtering.
        SNR_cut (float): The SNR threshold for candidates.
        period_tol (float): The tolerance percentage for period filtering.
    """
    # Ensure output path exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the DM array
    DM_array = prepare_dm_array(start_DM, end_DM, dm_step)
    print(f"Generated DM array: {DM_array}")

    cand_len = []

    for dm in DM_array:
        file_dm_path0 = os.path.join(input_dir, f"{file_name}_DM{float(dm) - dm_step:.2f}.dat")
        file_dm_path1 = os.path.join(input_dir, f"{file_name}_DM{dm}.dat")

        if os.path.exists(file_dm_path1):
            with open(file_dm_path1) as f:
                lines = f.readlines()
            cand_len.append(len(lines))
            print(f"Loaded {len(lines)} candidates for DM={dm}")
        else:
            print(f"File not found for DM={dm}, attempting to copy from previous DM.")
            if os.path.exists(file_dm_path0):
                shutil.copy(file_dm_path0, file_dm_path1)
                try:
                    with open(file_dm_path1) as f:
                        lines = f.readlines()
                    cand_len.append(len(lines))
                    print(f"Loaded {len(lines)} candidates for DM={dm} after copying the previous DM file.")
                except Exception as e:
                    print(f"Failed to read the copied file for DM={dm}: {e}")
            else:
                print(f"Previous DM file {file_dm_path0} also not found. Skipping DM={dm}.")

    max_cand_len = max(cand_len)
    if max_cand_len == 0:
        print("No candidates found in any DM trial. Exiting.")
        return

    print(f"Maximum number of candidates in any DM trial: {max_cand_len}")

    # Initialize arrays for candidate parameters
    Period_dot_array = np.full((len(DM_array), max_cand_len), np.nan)
    r_bin_array = np.full((len(DM_array), max_cand_len), np.nan)
    Period_array = np.full((len(DM_array), max_cand_len), np.nan)
    SNR_array = np.full((len(DM_array), max_cand_len), np.nan)
    Power_array = np.full((len(DM_array), max_cand_len), np.nan)

    for i in range(0, len(DM_array)):
        data = np.loadtxt(os.path.join(input_dir, f"{file_name}_DM{DM_array[i]}.dat"), dtype=float)
        A0 = np.array(data)

        #  A0[:,0] is z_bin array
        #  A0[:,1] is acceleration array
        #  A0[:,2] is r_bin array
        #  A0[:,3] is frequency array
        #  A0[:,4] is Power array
        #  A0[:,5] is SNR array

        # Deletes all the candidates having zero-frequency, zero SNR and very very high SNR
        A0 = A0[A0[:, 3] != 0]
        A0 = A0[A0[:, 5] > 0]
        A0 = A0[A0[:, 5] < 10**8]
    
        c = 3*10**8
        cand_array_len = len(A0[:,0])
        
        for j in range(0, int(cand_array_len)):
            if low_period <= np.multiply(np.divide(1.0, A0[j][3]), 1000) <= high_period:
            
                Period_dot_array[i][j] = np.divide(np.multiply(A0[j][1], np.divide(1.0, A0[j][3])), c)
                r_bin_array[i][j] = A0[j][2]
                Period_array[i][j] = np.divide(1.0, A0[j][3]) # In seconds
                Power_array[i][j] = A0[j][4]
                SNR_array[i][j] = A0[j][5]

    print(f"Finished processing candidate data for {len(DM_array)} DM trials.")

    # Unique r_bin filtering
    r_bin_flatten = r_bin_array.flatten()
    filtered_r_bin_list0 = [x for x in np.unique(r_bin_flatten) if str(x) != 'nan']
    print(filtered_r_bin_list0)
    tot_cand = len(filtered_r_bin_list0)
    uniq_r_bin_list0, r_tol_array = [], []

    print(f"Filtered unique r_bin list length: {len(filtered_r_bin_list0)}")

    while len(filtered_r_bin_list0) > 0:
        indices = np.where(filtered_r_bin_list0 <= filtered_r_bin_list0[0] + filtered_r_bin_list0[0] * (period_tol_init_sort / 100.0))
        uniq_r_bin_list0.append(filtered_r_bin_list0[0])
        r_tol_array.append(filtered_r_bin_list0[0] * (period_tol_init_sort / 100.0))
        filtered_r_bin_list0 = np.delete(filtered_r_bin_list0, indices[0])

    # Final filtering and output
    output_file = os.path.join(output_dir, f"{file_name}_all_sifted_candidates.txt")
    with open(output_file, "w") as file:
        file.write("Period(sec)   Pdot(s/s)  DM(pc/cc)   SNR\n")

    print(f"Writing output to {output_file}")

    for i, uniq_r_bin in enumerate(uniq_r_bin_list0):
        if i == 0:
            index = np.where(r_bin_array <= uniq_r_bin + r_tol_array[i] / 2)
        else:
            index = np.where((r_bin_array > uniq_r_bin_list0[i - 1] + r_tol_array[i - 1] / 2) &
                             (r_bin_array <= uniq_r_bin + r_tol_array[i] / 2))

        DM_index, cand_index = index[0], index[1]
        DM_groups = consecutive(np.unique(DM_index))

        DM_slope = np.divide(np.subtract(DM_filtering_cut_1000, DM_filtering_cut_10), 0.990)
        DM_intercept = np.subtract(DM_filtering_cut_10, np.multiply(DM_slope, 0.010))

        DM_filtering_cut = int(np.divide(np.add(DM_intercept, np.multiply(DM_slope, Period_array[index[0][0]][index[1][0]])), dm_step))
        
        #print(f"DM slope is {DM_slope} pc/cc, DM intercept is {DM_intercept} pc/cc, and DM filtering cut is {DM_filtering_cut}")
        print(f"For Period {Period_array[index[0][0]][index[1][0]]} sec, DM tolerance is {np.multiply(DM_filtering_cut, dm_step)} pc/cc")

        for group in DM_groups:
            if len(group) >= DM_filtering_cut:
                Filtered_SNR_array = np.full(SNR_array.shape, np.nan)
                for dm_idx in group:
                    DM_cand_indices = np.where(DM_index == dm_idx)
                    Filtered_SNR_array[dm_idx, cand_index[DM_cand_indices]] = SNR_array[dm_idx, cand_index[DM_cand_indices]]

                if np.all(np.isnan(Filtered_SNR_array)):
                    continue

                maxima_index = np.unravel_index(np.nanargmax(Filtered_SNR_array), Filtered_SNR_array.shape)
                if SNR_array[maxima_index] >= SNR_cut:
                    with open(output_file, "a") as file:
                        file.write(f"{Period_array[maxima_index]:.10f}     "
                                   f"{Period_dot_array[maxima_index]:.6e}     "
                                   f"{DM_array[maxima_index[0]]}     "
                                   f"{SNR_array[maxima_index]:.2f}\n")
                    print(f"Candidate: Period={Period_array[maxima_index]:.6f}, "
                          f"Pdot={Period_dot_array[maxima_index]:.6e}, "
                          f"SNR={SNR_array[maxima_index]:.2f}")
                    
    print("Finished processing all candidates and writing the output.")
