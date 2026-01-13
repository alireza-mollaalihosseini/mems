#!/bin/bash

# ----------------------------------
# Missing (a, u_dc) pairs
# ----------------------------------
missing_points=(
    # "0.40 0.30"
    # "0.40 0.60"
    # "0.40 0.70"
    # "0.40 0.80"
    # "0.40 0.90"
    # "0.40 1.00"
    # "0.50 0.10"
    # "0.50 0.20"
    # "0.50 0.30"
    # "0.50 0.40"
    # "0.50 0.50"
    # "0.50 0.60"
    # "0.50 0.70"
    # "0.50 0.90"
    # "0.50 1.00"
    # "0.60 0.10"
    # "0.60 0.20"
    # "0.60 0.30"
    # "0.60 0.40"
    # "0.60 0.50"
    # "0.60 0.60"
    # "0.60 0.70"
    # "0.60 0.90"
    # "0.60 1.00"
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
# Frequencies
# ----------------------------------
f_values=($(seq 1000 490 50000))

# ----------------------------------
# Throttling parameters
# ----------------------------------
MAX_JOBS=15
SLEEP_SECONDS=90

count_user_jobs() {
    bjobs -u almo2783 -q BatchXL | grep -E "(PEND|RUN)" | wc -l
}

job_counter=0
total_jobs=$(( ${#missing_points[@]} * ${#f_values[@]} ))

echo "Total jobs to submit: $total_jobs"
echo "Submitting in batches of $MAX_JOBS with ${SLEEP_SECONDS}s sleep"
echo ""

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

    for f in "${f_values[@]}"; do
        # -------------------------
        # WAIT until queue has room
        # -------------------------
        while true; do
            active_jobs=$(count_user_jobs)
            if (( active_jobs < MAX_JOBS )); then
                break
            fi
            echo "Queue full ($active_jobs jobs). Sleeping ${SLEEP_SECONDS}s..."
            sleep "$SLEEP_SECONDS"
        done

        echo "Submitting: a=$a_fmt, u_dc=$u_dc_fmt, f=$f"

        bsub -q BatchXL \
            -n 64 \
            -J "ridge_missing" \
            -o "${results_dir}/err-out/output_ridge.txt" \
            -e "${results_dir}/err-out/error_ridge.txt" \
            python3 simulation.py \
            --a "$a" \
            --u_dc "$u_dc" \
            --f "$f" \
            && echo "Submitted OK" \
            || echo "FAILED: f=$f" >> "$failed_log"

        ((job_counter++))
        echo "Progress: $job_counter / $total_jobs"
    done
done

echo "All jobs submitted."
