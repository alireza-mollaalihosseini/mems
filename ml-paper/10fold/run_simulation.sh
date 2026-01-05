#!/bin/bash

# ----------------------------------
# Parameter grids
# ----------------------------------


# a grid = linspace(-1,1,41)
a_values=($(seq -1 0.05 1))

mu=1.0  # Hardcoded, not looped
u_dc=0.4  # Hardcoded, not looped

# ----------------------------------
# Output dirs
# ----------------------------------
results_dir="/scratch/almo2783/scratch/ml-paper/10fold/results/5cities"
mkdir -p "${results_dir}/err-out"

# ----------------------------------
# Submission control
# ----------------------------------
max_concurrent_jobs=4
sleep_interval=300

failed_log="${results_dir}/err-out/failed_jobs.log"

count_user_jobs() {
    bjobs -u almo2783 -q highmem | grep -E "(PEND|RUN)" | wc -l
}

# ----------------------------------
# Total jobs
# ----------------------------------
total_jobs=$(( ${#a_values[@]} ))
job_count=0

echo "Starting grid search:"
echo "  a values  : ${#a_values[@]}"
echo "  TOTAL JOBS: $total_jobs"
echo "Target concurrency: $max_concurrent_jobs"
echo ""

# ----------------------------------
# Grid loop
# ----------------------------------
for a in "${a_values[@]}"; do

    current_jobs=$(count_user_jobs)
    while [[ $current_jobs -ge $max_concurrent_jobs ]]; do
        echo "Queue full ($current_jobs jobs). Sleeping $sleep_interval sec..."
        sleep "$sleep_interval"
        current_jobs=$(count_user_jobs)
    done

    job_name="ridge_a_${a}"

    out_file="${results_dir}/err-out/output_a_${a}.txt"
    err_file="${results_dir}/err-out/error_a_${a}.txt"

    bsub -q highmem \
            -n 64 \
            -J "$job_name" \
            -o "$out_file" \
            -e "$err_file" \
            python3 simulation.py \
            --a  "$a" \
            && echo "Submitted: a=$a" \
            || echo "FAILED: a=$a" >> "$failed_log"

    ((job_count++))
    echo "Progress: $job_count / $total_jobs"

    sleep 5
done

echo ""
echo "All grid jobs submitted: $job_count / $total_jobs"
echo "Monitor with: bjobs -u almo2783 -q highmem"
