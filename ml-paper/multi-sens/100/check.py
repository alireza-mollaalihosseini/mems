import glob
import os
import numpy as np



# read all .npz files in a directory and check which ones are missing
f_values = np.linspace(1000, 50000, 101)
path = "/scratch/almo2783/scratch/ml-paper/multi-sens/100/results"
all_files = set(os.path.basename(f) for f in glob.glob(os.path.join(path, "f-*.npz")))
missing_files = []
for f in f_values:
    filename = f"f-{int(f)}.npz"
    if filename not in all_files:
        missing_files.append(filename)


#  save missing files to a text file
with open("missing_files.txt", "w") as f:
    for filename in missing_files:
        f.write(f"{filename}\n")

#  print missinf frequencies
print("Missing frequencies:")
for filename in missing_files:
    f_value = int(filename.split("-")[1].split(".")[0])
    print(f_value)

