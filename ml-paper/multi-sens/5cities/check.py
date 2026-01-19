import os
import numpy as np

# Define the parameters (same as in your main script)
f_values = np.linspace(1000, 50000, 101)
num_tenths = 10

# Base paths
full_base_path = "/scratch/almo2783/scratch/ml-paper/multi-sens/5cities/results/"
sub_base_path  = "/scratch/almo2783/scratch/ml-paper/multi-sens/5cities/results/one-tenth/"

# Containers for missing files
missing_full = []                     # list of frequencies with missing full 1s file
missing_sub  = {}                      # dict: frequency -> list of missing segments (1-10)

print("Checking file existence...\n")

all_exist = True

for i, f in enumerate(f_values):
    int_f = int(f)  # as used in filenames
    
    # Check full 1-second file
    full_path = os.path.join(full_base_path, f"f-{int_f}.npz")
    if not os.path.isfile(full_path):
        missing_full.append(int_f)
        all_exist = False
    
    # Check 10 sub-segment files
    missing_segs = []
    for seg in range(1, num_tenths + 1):
        sub_path = os.path.join(sub_base_path, f"f-{int_f}-{seg}.npz")
        if not os.path.isfile(sub_path):
            missing_segs.append(seg)
            all_exist = False
    
    if missing_segs:
        missing_sub[int_f] = missing_segs

# Summary output
if all_exist:
    print("All required files exist! (101 full files + 101×10 = 1010 sub-segment files)")
else:
    print("Some files are missing.\n")
    
    if missing_full:
        print(f"Missing full 1-second files ({len(missing_full)} frequencies):")
        print(sorted(missing_full))
        print()
    
    if missing_sub:
        print(f"Missing sub-segment files (for {len(missing_sub)} frequencies):")
        for freq in sorted(missing_sub):
            segs = sorted(missing_sub[freq])
            print(f"  Frequency {freq}: missing segments {segs}  ({len(segs)} out of 10)")