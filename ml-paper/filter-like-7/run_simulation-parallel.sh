#!/bin/bash

a_values=(-2.0  -1.96 -1.92 -1.88 -1.84 -1.8  -1.76 -1.72 -1.68
          -1.64 -1.6  -1.56 -1.52 -1.48 -1.44 -1.4  -1.36 -1.32
          -1.28 -1.24 -1.2  -1.16 -1.12 -1.08 -1.04 -0.96 -0.92
          -0.88 -0.84 -0.76 -0.72 -0.68 -0.64 -0.6  -0.56 -0.52
          -0.48 -0.44 -0.36 -0.32 -0.28 -0.24 -0.16 -0.12 -0.08
          -0.04  0.04  0.08  0.12  0.16  0.24  0.28  0.32  0.36
           0.4   0.44  0.48  0.52  0.56  0.64  0.68  0.72  0.76
           0.8   0.84  0.88  0.92  0.96  1.04  1.08  1.12  1.16
           1.2   1.24  1.28  1.32  1.36  1.4   1.44  1.48  1.52
           1.56  1.6   1.64  1.68  1.72  1.76  1.8   1.84  1.88
           1.92  1.96  2.0)

# Ensure output directory exists
mkdir -p /scratch/almo2783/scratch/ml-paper/filter-like-7/err-out-up

job_count=0
max_jobs=10

for a in "${a_values[@]}"; do
    bsub -q BatchXL \
        -o "/scratch/almo2783/scratch/ml-paper/filter-like-7/err-out-up/output_${a}.txt" \
        -e "/scratch/almo2783/scratch/ml-paper/filter-like-7/err-out-up/error_${a}.txt" \
        -J "simulation_${a}" \
        python3 simulation-parallel.py "${a}" \
        || echo "Failed to submit job for a=${a}" >> /scratch/almo2783/scratch/ml-paper/filter-like-7/err-out-up/failed_jobs.log

    ((job_count++))

    if [[ $job_count -eq $max_jobs ]]; then
        echo "Submitted $max_jobs jobs. Waiting for 3 hours before continuing..."
        sleep 5000   # 2.5 hours = 9000 seconds
        job_count=0
    fi
done

echo "All jobs have been scheduled."
