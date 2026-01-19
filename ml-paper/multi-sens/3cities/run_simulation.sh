#!/bin/bash

# ----------------------------------
# Parameter grids
# ----------------------------------


# f grid = linspace(8000, 30000, 201)
# f_values=($(seq 1000 50 50000))
f_values=($(seq 1000 490 50000))
# f_values=($(seq 8000 110 30000))
# f_values=($(seq 8000 1000 12000))

mu=1.0  # Hardcoded, not looped
u_dc=1.0  # Hardcoded, not looped
a=0.9

# ----------------------------------
# Output dirs
# ----------------------------------
results_dir="/scratch/almo2783/scratch/ml-paper/multi-sens/3cities/results"
mkdir -p "${results_dir}/err-out"

# ----------------------------------
# Submission control
# ----------------------------------
max_concurrent_jobs=16
sleep_interval=90

failed_log="${results_dir}/err-out/failed_jobs.log"

count_user_jobs() {
    bjobs -u almo2783 -q BatchXL | grep -E "(PEND|RUN)" | wc -l
}

# ----------------------------------
# Total jobs
# ----------------------------------
total_jobs=$(( ${#f_values[@]} ))
job_count=0

echo "Starting grid search:"
echo "  a values  : ${a}"
echo "  u_dc values  : ${u_dc}"
echo "  TOTAL JOBS: $total_jobs"
echo "Target concurrency: $max_concurrent_jobs"
echo ""

# ----------------------------------
# Grid loop
# ----------------------------------
for f in "${f_values[@]}"; do

    current_jobs=$(count_user_jobs)
    while [[ $current_jobs -ge $max_concurrent_jobs ]]; do
        echo "Queue full ($current_jobs jobs). Sleeping $sleep_interval sec..."
        sleep "$sleep_interval"
        current_jobs=$(count_user_jobs)
    done

    job_name="ridge_f_${f}"

    out_file="${results_dir}/err-out/output_f_${f}.txt"
    err_file="${results_dir}/err-out/error_f_${f}.txt"

    bsub -q BatchXL \
            -n 64 \
            -J "$job_name" \
            -o "$out_file" \
            -e "$err_file" \
            python3 simulation.py \
            --f  "$f" \
            && echo "Submitted: f=$f" \
            || echo "FAILED: f=$f" >> "$failed_log"

    ((job_count++))
    echo "Progress: $job_count / $total_jobs"

    sleep 5
done

echo ""
echo "All grid jobs submitted: $job_count / $total_jobs"
echo "Monitor with: bjobs -u almo2783 -q BatchXL"