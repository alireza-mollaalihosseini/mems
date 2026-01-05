#!/bin/bash

c_f_values=(-50.0 -48.0 -46.0 -44.0 -42.0 -40.0 -38.0 -36.0 -34.0 -32.0 -30.0
            -28.0 -26.0 -24.0 -22.0 -20.0 -18.0 -16.0 -14.0 -12.0 -10.0  -8.0
             -6.0  -4.0  -2.0   0.0   2.0   4.0   6.0   8.0  10.0  12.0  14.0
             16.0  18.0  20.0  22.0  24.0  26.0  28.0  30.0  32.0  34.0  36.0
             38.0  40.0  42.0  44.0  46.0  48.0  50.0)

# Ensure output directory exists
mkdir -p /scratch/almo2783/scratch/dim-less/coupled/err-out-up

job_count=0
max_jobs=120

for c_f in "${c_f_values[@]}"; do
    bsub -q BatchXL \
        -o "/scratch/almo2783/scratch/dim-less/coupled/err-out-up/output_${c_f}.txt" \
        -e "/scratch/almo2783/scratch/dim-less/coupled/err-out-up/error_${c_f}.txt" \
        -J "simulation_${c_f}" \
        python3 sweep-up-faster.py "${c_f}" \
        || echo "Failed to submit job for c_f=${c_f}" >> /scratch/almo2783/scratch/dim-less/coupled/err-out-up/failed_jobs.log

    ((job_count++))

    if [[ $job_count -eq $max_jobs ]]; then
        echo "Submitted $max_jobs jobs. Waiting for 3 hours before continuing..."
        sleep 10800   # 3 hours = 10800 seconds
        job_count=0
    fi
done

echo "All jobs have been scheduled."
