#!/bin/bash

# ----------------------------------
# Parameter grids
# ----------------------------------

# grids
# a_values=($(seq -2 0.1 2))
a_values=(-2 2)
# u_dc_values=($(seq 0.1 0.1 1.0))
u_dc_values=(0.1 0.5)
# f_values=($(seq 1000 50 50000))
f_values=(2000 5000)

mu=1.0  # Hardcoded, not looped

# ----------------------------------
# Submission control
# ----------------------------------
max_concurrent_jobs=15
sleep_interval=300

count_user_jobs() {
    bjobs -u almo2783 -q BatchXL | grep -E "(PEND|RUN)" | wc -l
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
        # Output dirs
        # ----------------------------------
        results_dir="/scratch/almo2783/scratch/ml-paper/multi-sens/results/a-${a}-u_dc-${u_dc}"
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

            bsub -q BatchXL \
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
        echo "All grid jobs submitted: $job_count / $total_jobs"
        echo "Monitor with: bjobs -u almo2783 -q BatchXL"

        # # ----------------------------------
        # # Ridge Regression
        # # ----------------------------------

        # # wait for 3 mins
        # sleep 180

        # bsub -q BatchXL \
        #     -n 64 \
        #     -J "ridge" \
        #     -o "${results_dir}/err-out/output_ridge.txt" \
        #     -e "${results_dir}/err-out/error_ridge.txt" \
        #     python3 ridge.py \
        #     --a  "$a" \
        #     --u_dc  "$u_dc" \
        #     && echo "Submitted: ridge regression" \
        #     || echo "FAILED: ridge regression" >> "$failed_log"

    done
done