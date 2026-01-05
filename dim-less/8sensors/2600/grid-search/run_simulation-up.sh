#!/bin/bash

u_dc_values=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Ensure output directory exists
mkdir -p /scratch/almo2783/scratch/dim-less/8sensors/2600/grid-search/err-out-up

job_count=0
max_jobs=120

for u_dc in "${u_dc_values[@]}"; do
    bsub -q BatchXL \
        -o "/scratch/almo2783/scratch/dim-less/8sensors/2600/grid-search/err-out-up/output_${u_dc}.txt" \
        -e "/scratch/almo2783/scratch/dim-less/8sensors/2600/grid-search/err-out-up/error_${u_dc}.txt" \
        -J "simulation_${u_dc}" \
        python3 sweep-up-faster.py "${u_dc}" \
        || echo "Failed to submit job for u_dc=${u_dc}" >> /scratch/almo2783/scratch/dim-less/8sensors/2600/grid-search/err-out-up/failed_jobs.log

    ((job_count++))

    if [[ $job_count -eq $max_jobs ]]; then
        echo "Submitted $max_jobs jobs. Waiting for 3 hours before continuing..."
        sleep 10800   # 3 hours = 10800 seconds
        job_count=0
    fi
done

echo "All jobs have been scheduled."
