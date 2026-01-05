#!/bin/bash

# Define parameter array
mu_values=(1e-2  2e-2    5e-2    1e-1  2.2e-1  4.6e-1
           1     2.15    4.64    10    21.5    46.5  
           1e2   2.15e2  4.64e2  1e3   2.15e3  4.64e3
           1e4   2.15e4  4.64e4  1e5   2.15e5  6.64e5
           1e6)

lam=0.1
alpha=-0.01
beta=0.3
# Ensure output directory exists (adapted to your results dir)
results_dir="/scratch/almo2783/scratch/ml-paper/forced-hopf/mu-grid/results"
mkdir -p "${results_dir}/err-out"

# Balanced submission settings for this simulation
# Per-job: ~44 min runtime, 64 threads/cores, ~66GB mem (request 70GB, 1hr walltime)
# Total: 101 jobs; limit to 10 concurrent to avoid clutter (e.g., 10 nodes)
max_concurrent_jobs=10
sleep_interval=300      # 5 minutes between checks if queue full
max_total_jobs=25      # Total expected

job_count=0
submitted_today=0
failed_log="${results_dir}/err-out/failed_jobs.log"

# Function to count running/pending jobs for this user in the queue (BatchXL)
count_user_jobs() {
    bjobs -u almo2783 -q BatchXL | grep -E "(PEND|RUN)" | wc -l
}

echo "Starting balanced job submission for ${#mu_values[@]} mu_values (lam=${lam}, alpha=${alpha}, beta=${beta}). Total: $max_total_jobs jobs."
echo "Target concurrent jobs: <= $max_concurrent_jobs in 'BatchXL' queue."
echo "Per-job resources: 64 cores"
echo "Monitoring your jobs in queue 'BatchXL'."

for mu in "${mu_values[@]}"; do
    # Check current queue load before submitting
    current_jobs=$(count_user_jobs)
    while [[ $current_jobs -ge $max_concurrent_jobs ]]; do
        echo "Queue has $current_jobs jobs (PEND/RUN). Waiting $sleep_interval seconds..."
        sleep $sleep_interval
        current_jobs=$(count_user_jobs)
    done

    # Submit the job with resources tuned for your sim (64 cores, high mem, 1hr)
    bsub -q BatchXL \
        -n 64 \
        -o "${results_dir}/err-out/output_mu_${mu}.txt" \
        -e "${results_dir}/err-out/error_mu_${mu}.txt" \
        -J "ridge_mu_${mu}" \
        python3 simulation.py --mu "${mu}" \
        && echo "Submitted job for mu=${mu} (total submitted: $((++submitted_today)))" \
        || echo "Failed to submit job for mu=${mu}" >> "$failed_log"

    ((job_count++))
    echo "Progress: $job_count / $max_total_jobs"

    # Small pause after each submission to avoid burst
    sleep 5
done

echo "All $max_total_jobs jobs scheduled. Check $failed_log for any failures."
echo "To monitor: bjobs -u almo2783 -q BatchXL"
echo "Results will accumulate in $results_dir"