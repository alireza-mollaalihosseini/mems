#!/bin/bash

# ----------------------------------
# Parameter grids
# ----------------------------------

# Mu grid (log-space)
mu_values=(1e-18 1e-17 1e-16 1e-15 1e-14 1e-13 1e-12 1e-11 1e-10 1e-9 \
           1e-8  1e-7  1e-6  1e-5  1e-4  1e-3  1e-2  1e-1  1     10 \
           1e2   1e3   1e4   1e5   1e6)

# a grid = linspace(-1,1,51)
a_values=($(seq -1 0.04 1))

u_dc=0.4

# ----------------------------------
# Output dirs
# ----------------------------------
results_dir="/scratch/almo2783/scratch/ml-paper/mu-grid/results"
mkdir -p "${results_dir}/mu-a/err-out"

# ----------------------------------
# Submission control
# ----------------------------------
max_concurrent_jobs=10
sleep_interval=300

failed_log="${results_dir}/err-out/failed_jobs.log"

count_user_jobs() {
    bjobs -u almo2783 -q BatchXL | grep -E "(PEND|RUN)" | wc -l
}

# ----------------------------------
# Total jobs
# ----------------------------------
total_jobs=$(( ${#mu_values[@]} * ${#a_values[@]} ))
job_count=0

echo "Starting grid search:"
echo "  mu values : ${#mu_values[@]}"
echo "  a values  : ${#a_values[@]}"
echo "  TOTAL JOBS: $total_jobs"
echo "Target concurrency: $max_concurrent_jobs"
echo ""

# ----------------------------------
# Grid loop
# ----------------------------------
for mu in "${mu_values[@]}"; do
    for a in "${a_values[@]}"; do

        current_jobs=$(count_user_jobs)
        while [[ $current_jobs -ge $max_concurrent_jobs ]]; do
            echo "Queue full ($current_jobs jobs). Sleeping $sleep_interval sec..."
            sleep "$sleep_interval"
            current_jobs=$(count_user_jobs)
        done

        job_name="ridge_mu_${mu}_a_${a}"

        out_file="${results_dir}/err-out/output_mu_${mu}_a_${a}.txt"
        err_file="${results_dir}/err-out/error_mu_${mu}_a_${a}.txt"

        bsub -q BatchXL \
             -n 64 \
             -J "$job_name" \
             -o "$out_file" \
             -e "$err_file" \
             python3 simulation2.py \
                --mu "$mu" \
                --a  "$a" \
             && echo "Submitted: mu=$mu | a=$a" \
             || echo "FAILED: mu=$mu | a=$a" >> "$failed_log"

        ((job_count++))
        echo "Progress: $job_count / $total_jobs"

        sleep 5
    done
done

echo ""
echo "All grid jobs submitted: $job_count / $total_jobs"
echo "Monitor with: bjobs -u almo2783 -q BatchXL"
