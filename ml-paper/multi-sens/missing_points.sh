#!/bin/bash

# ----------------------------------
# Missing (a, u_dc) pairs
# ----------------------------------
missing_points=(
    "0.40 0.30"
    "0.40 0.60"
    "0.40 0.70"
    "0.40 0.80"
    "0.40 0.90"
    "0.40 1.00"
    "0.50 0.10"
    "0.50 0.20"
    "0.50 0.30"
    "0.50 0.40"
    "0.50 0.50"
    "0.50 0.60"
    "0.50 0.70"
    "0.50 0.90"
    "0.50 1.00"
    "0.60 0.10"
    "0.60 0.20"
    "0.60 0.30"
    "0.60 0.40"
    "0.60 0.50"
    "0.60 0.60"
    "0.60 0.70"
    "0.60 0.90"
    "0.60 1.00"
    "0.70 0.20"
    "0.70 0.60"
    "0.70 0.80"
    "0.70 1.00"
    "0.80 0.20"
    "0.80 0.30"
    "0.80 0.40"
    "0.80 0.60"
    "0.80 0.70"
    "0.80 0.80"
    "0.80 0.90"
    "0.90 0.80"
)

# ----------------------------------
# Throttling parameters
# ----------------------------------
MAX_JOBS=12
SLEEP_SECONDS=60   # 60 seconds

job_counter=0

# ----------------------------------
# Submit jobs
# ----------------------------------
for pair in "${missing_points[@]}"; do
    read -r a u_dc <<< "$pair"

    a_fmt=$(printf "%.2f" "$a")
    u_dc_fmt=$(printf "%.2f" "$u_dc")

    results_dir="/scratch/almo2783/scratch/ml-paper/multi-sens/results/a-${a_fmt}-u_dc-${u_dc_fmt}"
    mkdir -p "${results_dir}/err-out"

    failed_log="${results_dir}/err-out/failed_jobs.log"

    echo "Submitting missing job: a=$a_fmt, u_dc=$u_dc_fmt"

    bsub -q BatchXL \
        -n 64 \
        -J "ridge_missing" \
        -o "${results_dir}/err-out/output_ridge.txt" \
        -e "${results_dir}/err-out/error_ridge.txt" \
        python3 ridge.py \
        --a "$a" \
        --u_dc "$u_dc" \
        && echo "Submitted OK" \
        || echo "FAILED submission" >> "$failed_log"

    ((job_counter++))

    # ----------------------------------
    # Throttle submissions
    # ----------------------------------
    if (( job_counter % MAX_JOBS == 0 )); then
        echo "Submitted $job_counter jobs — sleeping for 60 seconds..."
        sleep "$SLEEP_SECONDS"
    fi

done
