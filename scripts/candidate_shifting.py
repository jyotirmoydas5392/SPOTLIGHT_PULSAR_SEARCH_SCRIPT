from multiprocessing import Process, Value, Array
from multiprocessing import Pool
from functools import partial
import concurrent.futures
import subprocess
import itertools
import numpy as np
import math
import sys
import os

file_name = fil_file.replace(".fil", "")
N0 = int(math.floor((end_DM - start_DM)/dm_step))

dm = []
for i in range(0, N0):
    dm.append("{:.2f}".format(start_DM + i*dm_step))
    DM_array = np.array(dm, dtype = str)


def consecutive(data, stepsize = 1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def sorting(file_name, DM_array, period_tol, DM_filtering_cut, low_period, high_period):
    
    #Stores the candidate number per DM trial.
    cand_len = []
    
    for i in range(0, len(DM_array)):
        with open(file_name + "_DM" + DM_array[i] + ".dat") as f:
            lines = f.readlines()
        cand_len.append(int(len(lines)))
    #print("Total number of candidates before filtering: ", sum(cand_len))
    
    os.system("sleep 2")
    max_cand_len = int(max(cand_len))
    print(max_cand_len)

    
    #Defining the arrays for storing the candidate parameters
    Period_dot_array = np.empty((int(len(DM_array)), max_cand_len))
    Period_dot_array[:] = np.nan
    
    r_bin_array = np.empty((int(len(DM_array)), max_cand_len))
    r_bin_array[:] = np.nan
    
    Period_array = np.empty((int(len(DM_array)), max_cand_len))
    Period_array[:] = np.nan
    
    SNR_array = np.empty((int(len(DM_array)), max_cand_len))
    SNR_array[:] = np.nan
    
    Power_array = np.empty((int(len(DM_array)), max_cand_len))
    Power_array[:] = np.nan
    
    
    for i in range(0, len(DM_array)):
        data = np.loadtxt(file_name + "_DM" + DM_array[i] + ".dat", dtype = float)
        A0 = np.array(data)

        #  A0[:,0] is z_bin array
        #  A0[:,1] is acceleration array
        #  A0[:,2] is r_bin array
        #  A0[:,3] is frequency array
        #  A0[:,4] is power array
        #  A0[:,5] is SNR array

        # Deletes all the candidates having zero-frequency and very very high SNR
        A0 = A0[A0[:, 3] != 0]
        A0 = A0[A0[:, 5] < 10**8]
    
        c = 3*10**8
        cand_array_len = len(A0[:,0])
        
        for j in range(0, int(cand_array_len)):
            if low_period <= np.multiply(np.divide(1.0, A0[j][3]), 1000) <= high_period:
            
                Period_dot_array[i][j] = np.divide(np.multiply(A0[j][0], np.divide(1.0, A0[j][3])), c)
                r_bin_array[i][j] = A0[j][2]
                Period_array[i][j] = np.divide(1.0, A0[j][3]) #In second
                Power_array[i][j] = A0[j][4]
                SNR_array[i][j] = A0[j][5]
        
        
    r_bin_flatten = r_bin_array.flatten()
    filtered_r_bin_list0 = [x for x in np.unique(r_bin_flatten) if str(x) != 'nan']
    
    tot_cand = int(len(filtered_r_bin_list0))
    uniq_r_bin_list0 = []
    r_tol_array = []
    
#     print(filtered_r_bin_list0)
    print(len(filtered_r_bin_list0))
    
    for i in range(0, tot_cand):
        if (len(filtered_r_bin_list0) > 0):

            indices = np.where(filtered_r_bin_list0 <= (filtered_r_bin_list0[0] + int(filtered_r_bin_list0[0] * (period_tol/100))))
            uniq_r_bin_list0.append(filtered_r_bin_list0[0])
            r_tol_array.append(int(filtered_r_bin_list0[0] * (period_tol/100)))
            filtered_r_bin_list0 = np.delete(filtered_r_bin_list0, indices[0])

    
#     print(uniq_r_bin_list0) 
#     print(len(r_tol_array))
#     print(r_tol_array)
#     print(uniq_r_bin_list0)

    Filtered_SNR_array = np.empty((int(len(DM_array)), max_cand_len))
    
    file = open(file_name + "_all_shifted_candidates.txt", "w")
    file.write("Period(sec)   Pdot(s/s)  DM(pc/cc)   SNR \n")
    file.close()
    file = open(file_name + "_all_shifted_candidates.txt", "a")
    
    for i in range(0, int(len(uniq_r_bin_list0))):
        
        if i == 0:
            index = np.where((r_bin_array <= uniq_r_bin_list0[i] + r_tol_array[i]/2))
            
        else:
            index = np.where((r_bin_array > uniq_r_bin_list0[i-1] + r_tol_array[i]/2) & (r_bin_array <= uniq_r_bin_list0[i] + r_tol_array[i]/2))
        
        DM_index = index[0]
        cand_index = index[1]
        DM_index0 = list(set(DM_index))

        Consecutive_DM_array = consecutive(DM_index0)
        

        for j in range(0, int(len(Consecutive_DM_array))):
            if len(Consecutive_DM_array[j]) >= DM_filtering_cut:
                Filtered_SNR_array[:] = np.nan

                for k in range(0, int(len(Consecutive_DM_array[j]))):
                    cand_index_pos = np.where(DM_index == Consecutive_DM_array[j][k])
                    for l in range(0, int(len(cand_index_pos))):
                        Filtered_SNR_array[Consecutive_DM_array[j][k]][cand_index[cand_index_pos[l]]] = SNR_array[Consecutive_DM_array[j][k]][cand_index[cand_index_pos[l]]]

                maxima_index = np.where(Filtered_SNR_array == np.nanmax(Filtered_SNR_array))

                if SNR_array[maxima_index[0][0]][maxima_index[1][0]] >= SNR_cut:
                    file.write(str(Period_array[maxima_index[0][0]][maxima_index[1][0]]))
                    file.write("     " + str(Period_dot_array[maxima_index[0][0]][maxima_index[1][0]]))
                    file.write("     " + str(DM_array[maxima_index[0][0]]))
                    file.write("     " + str(SNR_array[maxima_index[0][0]][maxima_index[1][0]]) + "\n")
                    
                    print(Period_array[maxima_index[0][0]][maxima_index[1][0]])
                    print(Period_dot_array[maxima_index[0][0]][maxima_index[1][0]])
                    print(SNR_array[maxima_index[0][0]][maxima_index[1][0]])