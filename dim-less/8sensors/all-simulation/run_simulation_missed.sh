#!/bin/bash

mu=1.0
out_dir="/scratch/almo2783/scratch/dim-less/8sensors/all-simulation/err-out"
mkdir -p "$out_dir"

job_count=0
max_jobs=20

# --- Missing parameter sets ---

# Sensor 1161 missing (a, u_dc) pairs
declare -a sensor1161_params=(
    "-0.9 0.4"
    "-0.88 0.4"
    "-0.88 0.5"
    "-0.88 0.6"
    "-0.74 0.5"
    "-0.7 0.2"
    "-0.7 0.8"
)

# Sensor 2600 missing (a, u_dc) pairs
declare -a sensor2600_params=(
    "-0.88 0.1"
)

# --- Submit missing jobs ---

for params in "${sensor1161_params[@]}"; do
    set -- $params
    a=$1
    u_dc=$2
    sensor=1161

    bsub -q BatchXL \
        -o "${out_dir}/output_${sensor}_${a}_${u_dc}_${mu}.txt" \
        -e "${out_dir}/error_${sensor}_${a}_${u_dc}_${mu}.txt" \
        -J "sim_${sensor}_${a}_${u_dc}_${mu}" \
        python3 /scratch/almo2783/scratch/dim-less/8sensors/${sensor}/simulation.py "${a}" "${u_dc}" "${mu}" \
        || echo "Failed for sensor=${sensor}, a=${a}, u_dc=${u_dc}, mu=${mu}" >> "${out_dir}/failed_jobs.log"

    ((job_count++))
    if [[ $job_count -eq $max_jobs ]]; then
        echo "Submitted $max_jobs jobs. Waiting for 20 minutes before continuing..."
        sleep 1200
        job_count=0
    fi
done

for params in "${sensor2600_params[@]}"; do
    set -- $params
    a=$1
    u_dc=$2
    sensor=2600

    bsub -q BatchXL \
        -o "${out_dir}/output_${sensor}_${a}_${u_dc}_${mu}.txt" \
        -e "${out_dir}/error_${sensor}_${a}_${u_dc}_${mu}.txt" \
        -J "sim_${sensor}_${a}_${u_dc}_${mu}" \
        python3 /scratch/almo2783/scratch/dim-less/8sensors/${sensor}/simulation.py "${a}" "${u_dc}" "${mu}" \
        || echo "Failed for sensor=${sensor}, a=${a}, u_dc=${u_dc}, mu=${mu}" >> "${out_dir}/failed_jobs.log"

    ((job_count++))
    if [[ $job_count -eq $max_jobs ]]; then
        echo "Submitted $max_jobs jobs. Waiting for 20 minutes before continuing..."
        sleep 1200
        job_count=0
    fi
done

echo "All missing jobs have been scheduled."
