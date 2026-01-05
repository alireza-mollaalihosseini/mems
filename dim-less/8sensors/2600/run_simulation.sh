#!/bin/bash

u_dc_values=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
a_values=(-1.0   -0.98 -0.96 -0.94 -0.92 -0.9  -0.88 -0.86 -0.84
          -0.82 -0.8  -0.78 -0.76 -0.74 -0.72 -0.7  -0.68 -0.66
          -0.64 -0.62 -0.6  -0.58 -0.56 -0.54 -0.52 -0.5  -0.48
          -0.46 -0.44 -0.42 -0.4  -0.38 -0.36 -0.34 -0.32 -0.3 
          -0.28 -0.26 -0.24 -0.22 -0.2  -0.18 -0.16 -0.14 -0.12
          -0.1  -0.08 -0.06 -0.04 -0.02  0.0   0.02  0.04  0.06
           0.08  0.1   0.12  0.14  0.16  0.18  0.2   0.22  0.24
           0.26  0.28  0.3   0.32  0.34  0.36  0.38  0.4   0.42
           0.44  0.46  0.48  0.5   0.52  0.54  0.56  0.58  0.6 
           0.62  0.64  0.66  0.68  0.7   0.72  0.74  0.76  0.78
           0.8   0.82  0.84  0.86  0.88  0.9   0.92  0.94  0.96
           0.98  1.0)

mu=1.0

# Ensure output directory exists
mkdir -p /scratch/almo2783/scratch/dim-less/8sensors/2600/err-out

job_count=0
max_jobs=120

for a in "${a_values[@]}"; do
    for u_dc in "${u_dc_values[@]}"; do
        bsub -q BatchXL \
            -o "/scratch/almo2783/scratch/dim-less/8sensors/2600/err-out/output_${a}_${u_dc}_${mu}.txt" \
            -e "/scratch/almo2783/scratch/dim-less/8sensors/2600/err-out/error_${a}_${u_dc}_${mu}.txt" \
            -J "simulation_${a}_${u_dc}_${mu}" \
            python3 simulation.py "${a}" "${u_dc}" "${mu}" \
            || echo "Failed to submit job for a=${a}, u_dc=${u_dc}, mu=${mu}" >> /scratch/almo2783/scratch/dim-less/8sensors/2600/err-out/failed_jobs.log

        ((job_count++))

        if [[ $job_count -eq $max_jobs ]]; then
            echo "Submitted $max_jobs jobs. Waiting for 1 hours before continuing..."
            sleep 900
            job_count=0
        fi
    done
done

echo "All jobs have been scheduled."