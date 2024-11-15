import numpy as np

fil_file = "GMRT_band4_real_noise_4k_81.92us.fil"
file_name = fil_file.replace(".fil", "")


def folding(string):
    os.system(string)

def candidates(file_name):
    
    if harmonic_flag == 0.0:
        data = np.loadtxt(file_name+ "_all_shifted_candidates.txt", dtype = str)
        data = np.delete(data, 0, 0)
    
        A = np.array(data, dtype = float)
        A1 = A[::-1]
        return A1
        
    elif harmonic_flag == 1.0:
        data = np.loadtxt(file_name+ "_harmonic_removed_candidates.txt", dtype = str)
        data = np.delete(data, 0, 0)
    
        A = np.array(data, dtype = float)
        A1 = A[::-1]
        return A1
    
    else:
        print("Select the correct harmonic_optimization flag.")
        
    
def folding_operation(fil_file, file_name, workers, harmonic_flag):
    
    candidate_array = candidates(file_name)
    
    Folding_period = candidate_array[:,0]
    Folding_period_dot = candidate_array[:,1]
    Folding_DM = candidate_array[:,2]
    
    if fold_soft == 0.0:
        
        dat_folding_strings = []
        fil_folding_strings = []
        
        for i in range(0, len(Folding_period)):
            dat_folding_strings.append("prepfold -p -pd -dm " + str("{:.2f}".format(Folding_DM[i])) + " -noxwin -zerodm -o " + file_name + "_DM" + str(Folding_DM[i]) + "_DAT" + " " + file_name + "_DM" + str(Folding_DM[i]) + ".dat")
            fil_folding_strings.append("prepfold -p -pd -dm " + str("{:.2f}".format(Folding_DM[i])) + " -noxwin -zerodm -o " + file_name + "_DM" + str(Folding_DM[i]) + "_FIL" + " " + fil_file)
            
        
        def main():
            with Pool(workers) as pool:
                if (fold_type == 0.0):
                    print("Total number of candidates after filtering:", len(dat_folding_strings))
                    pool.map(folding, dat_folding_strings)
                elif (fold_type == 1.0):
                    print("Total number of candidates after filtering:", len(fil_folding_strings))
                    pool.map(folding, fil_folding_strings)
                else:
                    print("Total number of candidates after filtering:", len(dat_folding_strings))
                    pool.map(folding, dat_folding_strings)
                    pool.map(folding, fil_folding_strings)

        if __name__ == '__main__':
            main()
        
    elif fold_soft == 1.0:
        print("Hello")
            
    else:
        print("Select the correct folding software.")