#!/bin/bash

# Define parameter array (101 values from np.linspace(-1,1,101))
lam_values=(-1.0 -0.98 -0.96 -0.94 -0.92 -0.9 -0.88 -0.86 -0.84
            -0.82 -0.8 -0.78 -0.76 -0.74 -0.72 -0.7 -0.68 -0.66 
            -0.64 -0.62 -0.6 -0.58 -0.56 -0.54 -0.52 -0.5 -0.48 
            -0.46 -0.44 -0.42 -0.4 -0.38 -0.36 -0.34 -0.32 -0.3 
            -0.28 -0.26 -0.24 -0.22 -0.2 -0.18 -0.16 -0.14 -0.12 
            -0.1 -0.08 -0.06 -0.04 -0.02 0.0 0.02 0.04 0.06 0.08 
            0.1 0.12 0.14 0.16 0.18 0.2 0.22 0.24 0.26 0.28 0.3 
            0.32 0.34 0.36 0.38 0.4 0.42 0.44 0.46 0.48 0.5 0.52 
            0.54 0.56 0.58 0.6 0.62 0.64 0.66 0.68 0.7 0.72 0.74 
            0.76 0.78 0.8 0.82 0.84 0.86 0.88 0.9 0.92 0.94 0.96 
            0.98 1.0)

mu=1e4  # Hardcoded, not looped
alpha=-0.01
beta=0.3

# Ensure output directory exists (adapted to your results dir)
results_dir="/scratch/almo2783/scratch/ml-paper/forced-hopf/10fold/results"
mkdir -p "${results_dir}/err-out"

# Balanced submission settings for this simulation
# Per-job: ~44 min runtime, 64 threads/cores, ~66GB mem (request 70GB, 1hr walltime)
# Total: 101 jobs; limit to 10 concurrent to avoid clutter (e.g., 10 nodes)
max_concurrent_jobs=10
sleep_interval=300      # 5 minutes between checks if queue full
max_total_jobs=101      # Total expected

job_count=0
submitted_today=0
failed_log="${results_dir}/err-out/failed_jobs.log"

# Function to count running/pending jobs for this user in the queue (BatchXL)
count_user_jobs() {
    bjobs -u almo2783 -q BatchXL | grep -E "(PEND|RUN)" | wc -l
}

echo "Starting balanced job submission for ${#lam_values[@]} lam_values (alpha=${alpha}, beta=${beta}, mu=${mu}). Total: $max_total_jobs jobs."
echo "Target concurrent jobs: <= $max_concurrent_jobs in 'BatchXL' queue."
echo "Per-job resources: 64 cores."
echo "Monitoring your jobs in queue 'BatchXL'."

for lam in "${lam_values[@]}"; do
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
        -o "${results_dir}/err-out/output_lam_${lam}.txt" \
        -e "${results_dir}/err-out/error_lam_${lam}.txt" \
        -J "ridge_lam_${lam}" \
        python3 simulation-10fold.py --lam "${lam}" \
        && echo "Submitted job for lam=${lam} (total submitted: $((++submitted_today)))" \
        || echo "Failed to submit job for lam=${lam}" >> "$failed_log"

    ((job_count++))
    echo "Progress: $job_count / $max_total_jobs"

    # Small pause after each submission to avoid burst
    sleep 5
done

echo "All $max_total_jobs jobs scheduled. Check $failed_log for any failures."
echo "To monitor: bjobs -u almo2783 -q BatchXL"
echo "Results will accumulate in $results_dir"