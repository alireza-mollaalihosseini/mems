#!/bin/bash

u_dc_values=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
a_values=(-2.0 -1.94 -1.88 -1.82 -1.76 -1.7 -1.64 -1.58 -1.52
          -1.46 -1.4 -1.34 -1.28 -1.22 -1.16 -1.1 -1.04 -0.98
          -0.92 -0.86 -0.8 -0.74 -0.68 -0.62 -0.56 -0.5 -0.44
          -0.38 -0.32 -0.26 -0.2 -0.14 -0.08 -0.02 0.04 0.1 
           0.16 0.22 0.28 0.34 0.4 0.46 0.52 0.58 0.64
           0.7 0.76 0.82 0.88 0.94 1.0 1.06 1.12 1.18
           1.24 1.3 1.36 1.42 1.48 1.54 1.6 1.66 1.72
           1.78 1.84 1.9 1.96 2.02 2.08 2.14 2.2 2.26
           2.32 2.38 2.44 2.5 2.56 2.62 2.68 2.74 2.8 
           2.86 2.92 2.98 3.04 3.1 3.16 3.22 3.28 3.34
           3.4 3.46 3.52 3.58 3.64 3.7 3.76 3.82 3.88
           3.94 4.0)

mu=1.0

# Ensure output directory exists
mkdir -p /scratch/almo2783/scratch/dim-less/barcelona/err-out

job_count=0
max_jobs=120

for a in "${a_values[@]}"; do
    for u_dc in "${u_dc_values[@]}"; do
        bsub -q BatchXL \
            -o "/scratch/almo2783/scratch/dim-less/barcelona/err-out/output_${a}_${u_dc}_${mu}.txt" \
            -e "/scratch/almo2783/scratch/dim-less/barcelona/err-out/error_${a}_${u_dc}_${mu}.txt" \
            -J "simulation_${a}_${u_dc}_${mu}" \
            python3 simulation.py "${a}" "${u_dc}" "${mu}" \
            || echo "Failed to submit job for a=${a}, u_dc=${u_dc}, mu=${mu}" >> /scratch/almo2783/scratch/dim-less/barcelona/err-out/failed_jobs.log

        ((job_count++))

        if [[ $job_count -eq $max_jobs ]]; then
            echo "Submitted $max_jobs jobs. Waiting for 1 hours before continuing..."
            sleep 900
            job_count=0
        fi
    done
done

echo "All jobs have been scheduled."