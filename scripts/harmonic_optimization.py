import numpy as np


def is_harmonically_related(a, b, tolerance):
    # Check if a and b are harmonically related (i.e., ratio is close to an integer)
    ratio = a / b
    return np.abs(ratio - round(ratio)) < tolerance

def remove_harmonics(arr, period_tol):
    index = []
    harm_cand = []
    
    for i in range(len(arr)):
        harmonically_related = False
        
        harm_cand0 = []
        for j in range(i):
            
            tolerance = arr[i]*period_tol/(100*arr[j])
#             tolerance = period_tol/100
            
            if is_harmonically_related(arr[i], arr[j], tolerance):
                harmonically_related = True
                
                break
            
        if harmonically_related:
            index.append(i)
            harm_cand0.append(arr[j])
            harm_cand0.append(arr[i])
            harm_cand.append(harm_cand0)
    
    return harm_cand


def harmonic_optimization(file_name, period_tol):
    
    data = np.loadtxt(file_name+ "_all_shifted_candidates.txt", dtype = str)
    data = np.delete(data, 0, 0)
    
    A = np.array(data, dtype = float)
    A1 = A[::-1]
    #  A0[:,0] is period array
    #  A0[:,1] is period dot array
    #  A0[:,2] is DM array
    #  A0[:,3] is SNR array
    
    harmonic_candidates = remove_harmonics(A1[:,0], period_tol)
    
    sorted_fundamental_and_harmonic_candidates = np.array(sorted(harmonic_candidates, key=lambda x: x[0]))
    fundamental_candidates = sorted(list(set(sorted_fundamental_and_harmonic_candidates[:,0])))
#     print(fundamental_candidates)
    print(sorted_fundamental_and_harmonic_candidates)
    
    harmonic_len = []
    harmonic_candidate_index = []
    for i in range(0, len(fundamental_candidates)):
        harmonic_cand_index_temp = []
        X0 = np.where(sorted_fundamental_and_harmonic_candidates[:,0] == fundamental_candidates[i])
        
        X1 = np.where(A1[:,0] == fundamental_candidates[i])
        harmonic_cand_index_temp.append(X1[0][0])
#         print(X1)
        
        for j in range(0, len(X0[0])):
            X2 = np.where(A1[:,0] == sorted_fundamental_and_harmonic_candidates[:,1][X0[0][j]])
            harmonic_cand_index_temp.append(X2[0][0])
#             print(X2)
            
        harmonic_len.append(len(harmonic_cand_index_temp))
        harmonic_candidate_index.append(harmonic_cand_index_temp)
    
#     print(harmonic_candidate_index)
    
    
    max_harmonic_len = np.max(harmonic_len)
    harmonic_index_array = np.empty((int(len(harmonic_candidate_index)), int(max_harmonic_len)))
    harmonic_index_array[:] = np.nan
    
    for i in range(int(len(harmonic_candidate_index))):
        for j in range(int(len(harmonic_candidate_index[i]))):
            harmonic_index_array[i][j] = harmonic_candidate_index[i][j]
    
    flatten_harmonic_index_array = np.array([x for x in np.unique(harmonic_index_array.flatten()) if str(x) != 'nan'], dtype = int)
    
    A2 = np.delete(A1, flatten_harmonic_index_array, axis = 0)
#     print(A2[:,0])
    A2 = list(A2)
    
    for i in range(int(len(harmonic_candidate_index))):
        harmonic_SNR_array = np.empty(int(len(A1[:,0])))
        harmonic_SNR_array[:] = np.nan
        
        for j in range(int(len(harmonic_candidate_index[i]))):
            harmonic_SNR_array[harmonic_candidate_index[i][j]] = A1[:,3][harmonic_candidate_index[i][j]]
            
        X0 = np.where(harmonic_SNR_array == np.nanmax(harmonic_SNR_array))
        A2.append(A1[X0[0][0]])
        
    A2 = np.array(sorted(A2, key=lambda x: x[0]))
    A2 = A2[::-1]
#     print(A2)

    file = open(file_name + "_harmonic_removed_candidates.txt", "w")
    file.write("Period(sec)   Pdot(s/s)  DM(pc/cc)   SNR \n")
    file.close()
    file = open(file_name + "_harmonic_removed_candidates.txt", "a")
    
    for i in range(0, len(A2)):
        file.write(str(A2[i][0]))
        file.write("     " + str(A2[i][1]))
        file.write("     " + str(A2[i][2]))
        file.write("     " + str(A2[i][3]) + "\n")