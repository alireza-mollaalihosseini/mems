#!/bin/bash

# ----------------------------------
# Parameter grids
# ----------------------------------

# grids
a_values=($(seq -2 0.1 2))
# a_values=(-2 2)
u_dc_values=($(seq 0.1 0.1 1.0))
# u_dc_values=(0.1 0.5)
# f_values=($(seq 1000 50 50000))  # 981 frequency bins
# f_values=($(seq 1000 245 50000))  # 201 frequency bins
f_values=($(seq 1000 490 50000))  # 101 frequency bins
# f_values=(2000 5000)

mu=1.0  # Hardcoded, not looped

# ----------------------------------
# Submission control
# ----------------------------------
max_concurrent_jobs=15
sleep_interval=60

count_user_jobs() {
    bjobs -u almo2783 -q Batch72 | grep -E "(PEND|RUN)" | wc -l
}

# ----------------------------------
# Total jobs
# ----------------------------------
total_jobs=$(( ${#a_values[@]} * ${#u_dc_values[@]} * ${#f_values[@]} ))
job_count=0

echo "Starting grid search:"
echo "  a values  : ${#f_values[@]}"
echo "  TOTAL JOBS: $total_jobs"
echo "Target concurrency: $max_concurrent_jobs"
echo ""

for a in "${a_values[@]}"; do
    for u_dc in "${u_dc_values[@]}";do

        # ----------------------------------
        # Format parameters (2 decimal digits)
        # ----------------------------------
        a_fmt=$(printf "%.2f" "$a")
        u_dc_fmt=$(printf "%.2f" "$u_dc")

        # ----------------------------------
        # Output dirs
        # ----------------------------------
        results_dir="/scratch/almo2783/scratch/ml-paper/multi-sens/results/a-${a_fmt}-u_dc-${u_dc_fmt}"
        mkdir -p "${results_dir}/err-out"

        failed_log="${results_dir}/err-out/failed_jobs.log"

        job_group="sim_a${a}_u${u_dc}"
        echo "Submitting simulations for a=$a, u_dc=$u_dc"

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

            # job_name="ridge_f_${f}"
            job_name="sim_a${a}_u${u_dc}_f${f}"

            out_file="${results_dir}/err-out/output_f_${f}.txt"
            err_file="${results_dir}/err-out/error_f_${f}.txt"

            bsub -q Batch72 \
                    -n 64 \
                    -J "$job_name" \
                    -g "/$job_group" \
                    -o "$out_file" \
                    -e "$err_file" \
                    python3 simulation.py \
                    --a  "$a" \
                    --u_dc  "$u_dc" \
                    --f  "$f" \
                    && echo "Submitted: f=$f" \
                    || echo "FAILED: f=$f" >> "$failed_log"

            ((job_count++))
            echo "Progress: $job_count / $total_jobs"

            sleep 5
        done

        echo ""
        echo "All grid jobs for for a=$a, u_dc=$u_dc submitted: $job_count / $total_jobs"
        echo "Monitor with: bjobs -u almo2783 -q Batch72"

        # ----------------------------------
        # Ridge Regression
        # ----------------------------------

        # wait for 1.5 mins
        sleep 90

        bsub -q Batch72 \
            -n 64 \
            -J "ridge" \
            -o "${results_dir}/err-out/output_ridge.txt" \
            -e "${results_dir}/err-out/error_ridge.txt" \
            python3 ridge.py \
            --a  "$a" \
            --u_dc  "$u_dc" \
            && echo "Submitted: ridge regression" \
            || echo "FAILED: ridge regression" >> "$failed_log"

    done
done