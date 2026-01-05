# #!/bin/bash

# # Define your parameters
# a_values=(-2.0 -1.9 -1.8 -1.7 -1.6 -1.5 -1.4 -1.3 -1.2 -1.1 -1.0
#        -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1  0.0 )
# u_dc_values=(0.4 0.9)
# mu= (1e-7 5e-7 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1 5 10 50)


# # Initialize job count
# job_count=0
# max_jobs=121

# # Loop through parameter combinations
# for a in "${a_values[@]}"; do
#     for u_dc in "${u_dc_values[@]}"; do
#         # Schedule the job
#         at now <<EOF
# #!/bin/bash
#         bsub -q BatchXL \
#              -o /scratch/almo2783/scratch/amplitude/design2/err-out/output_${a}_${u_dc}.txt \
#              -e /scratch/almo2783/scratch/amplitude/design2/err-out/error_${a}_${u_dc}.txt \
#              -J simulation ./simulation $a $u_dc $mu || echo "Failed to submit job for a=${a}, u_dc=${u_dc}" >> /scratch/almo2783/scratch/amplitude/design2/err-out/failed_jobs.log
# EOF

#         # Increment the job count
#         ((job_count++))

#         # If we've submitted 253 jobs, wait for 8.3 hours before continuing
#         if [[ $job_count -eq $max_jobs ]]; then
#             echo "Submitted $max_jobs jobs. Waiting for 8.5 hours before continuing..."
#             sleep 30600  # 30600 seconds = 8.5 hours
#             job_count=0  # Reset the job count
#         fi
#     done
# done

# echo "All jobs have been scheduled."


#!/bin/bash

# # Define your parameters
# a_values=(-2.0 -1.96 -1.92 -1.88 -1.84 -1.8 -1.76 -1.72 -1.68 -1.64 -1.6 -1.56 -1.52 -1.48 -1.44 -1.4 -1.36 -1.32
#           -1.28 -1.24 -1.2 -1.16 -1.12 -1.08 -1.04 -1.0 -0.96 -0.92 -0.88 -0.84 -0.8 -0.76 -0.72 -0.68 -0.64 -0.6
#           -0.56 -0.52 -0.48 -0.44 -0.4 -0.36 -0.32 -0.28 -0.24 -0.2 -0.16 -0.12 -0.08 -0.04 0.0 0.04 0.08 0.12
#            0.16 0.2 0.24 0.28 0.32 0.36 0.4 0.44 0.48 0.52 0.56 0.6 0.64 0.68 0.72 0.76 0.8 0.84 0.88 0.92 0.96
#            1.0 1.04 1.08 1.12 1.16 1.2 1.24 1.28 1.32 1.36 1.4 1.44 1.48 1.52 1.56 1.6 1.64 1.68 1.72 1.76 1.8
#            1.84 1.88 1.92 1.96 2.0)
# u_dc_values=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
# mu=1.7e-5

# # Initialize job count
# job_count=0
# max_jobs=510  # Adjusted to match your conditional check

# # Loop through parameter combinations
# for a in "${a_values[@]}"; do
#     for u_dc in "${u_dc_values[@]}"; do
#         # Schedule the job
#         at now <<EOF
# #!/bin/bash
#         bsub -q BatchXL \
#              -o /scratch/almo2783/scratch/barcelona/design1/err-out/output_${a}_${u_dc}.txt \
#              -e /scratch/almo2783/scratch/barcelona/design1/err-out/error_${a}_${u_dc}.txt \
#              -J simulation ./simulation $a $u_dc $mu || echo "Failed to submit job for a=${a}, u_dc=${u_dc}" >> /scratch/almo2783/scratch/barcelona/design1/err-out/failed_jobs.log
# EOF

#             # Increment the job count
#             ((job_count++))

#             # If we've submitted 253 jobs, wait for 8.3 hours before continuing
#             if [[ $job_count -eq $max_jobs ]]; then
#                 echo "Submitted $max_jobs jobs. Waiting for 8.5 hours before continuing..."
#                 sleep 13000  # 31000 seconds = 8.5 hours
#                 job_count=0  # Reset the job count
#             fi
#         done
#     done
# done

# echo "All jobs have been scheduled."


parameter_combinations=(
    "-1.96 0.8"
    "-1.68 0.2"
    "-1.68 0.7"
    "-1.36 0.2"
    "-1.36 0.4"
    "-1.28 0.4"
    "-1.28 1.0"
    "-1.24 0.5"
    "-1.2 0.3"
    "-1.2 0.4"
    "-1.2 0.5"
    "-1.16 0.3"
    "-1.16 0.7"
    "-1.16 0.9"
    "-1.12 0.2"
    "-1.12 0.6"
    "-1.08 0.4"
    "-1.04 0.1"
    "-1.04 0.7"
    "-1.04 1.0"
    "-1.0 0.1"
    "-1.0 0.2"
    "-1.0 0.6"
    "-1.0 0.7"
    "-0.96 0.2"
    "-0.96 0.6"
    "-0.92 0.2"
    "-0.92 0.5"
    "-0.92 0.9"
    "-0.88 0.2"
    "-0.88 0.3"
    "-0.88 0.9"
    "-0.84 0.3"
    "-0.84 0.5"
    "-0.84 1.0"
    "-0.8 0.6"
    "-0.8 1.0"
    "-0.72 0.2"
    "-0.68 0.7"
    "-0.68 1.0"
    "-0.64 0.4"
    "-0.6 0.4"
    "-0.56 0.5"
    "0.32 0.7"
    "0.44 0.5"
    "0.76 0.8"
    "0.76 0.9"
    "1.04 0.7"
    "1.24 1.0"
    "1.32 0.8"
    "1.48 0.3"
    "1.52 0.9"
    "1.72 0.3"
    "2.0 0.2"
)

mu=1.7e-5

# Initialize job count
job_count=0
max_jobs=510

# Loop through the specific parameter combinations
for params in "${parameter_combinations[@]}"; do
    # Extract `a` and `u_dc` values
    a=$(echo $params | awk '{print $1}')
    u_dc=$(echo $params | awk '{print $2}')
    
    # Schedule the job
    at now <<EOF
#!/bin/bash
        bsub -q highmem \
             -o /scratch/almo2783/scratch/barcelona/design1/err-out/output_${a}_${u_dc}.txt \
             -e /scratch/almo2783/scratch/barcelona/design1/err-out/error_${a}_${u_dc}.txt \
             -J simulation ./simulation $a $u_dc $mu || echo "Failed to submit job for a=${a}, u_dc=${u_dc}" >> /scratch/almo2783/scratch/barcelona/design1/err-out/failed_jobs.log
EOF

    # Increment the job count
    ((job_count++))

    # If we've submitted `max_jobs`, wait for 3 hours before continuing
    if [[ $job_count -eq $max_jobs ]]; then
        echo "Submitted $max_jobs jobs. Waiting for 3 hours before continuing..."
        sleep 31000  # 10800 seconds = 3 hours
        job_count=0  # Reset the job count
    fi
done

echo "All jobs have been scheduled."