#!/bin/bash -xe

# Ensure the slurm_out directory exists
mkdir -p slurm_out

# Array to store jobs to check later
declare -a job_ids

for log in $(find ../../benchmarks/default/ -name '*_klog'); do
    # Prepare the filename for the SLURM output
    job_name="plot_$(basename $(dirname $log))"
    slurm_file="slurm_out/${job_name}.out"

    output_file=`python3 fault_parsing.py $log`
    if [ ! -e $output_file ]; then
        # Submitting the job to SLURM with the correct output file and partition
        job_id=$(sbatch --parsable --job-name=$job_name --output=$slurm_file --partition=hsw --time=1:00:00 --mem=40G --cpus-per-task=4 --wrap="python3 fault_plot.py $log")
        job_ids+=($job_id)
    else
        echo "skipping $output_file because it already exists."
    fi
done

# Wait for all jobs to complete
all_done=0
while [ $all_done -eq 0 ]; do
    all_done=1
    for job_id in "${job_ids[@]}"; do
        status=$(sacct -j $job_id --format=State --noheader | head -n 1 | awk '{print $1}')
        if [[ $status == "RUNNING" ]] || [[ $status == "PENDING" ]]; then
            all_done=0
            break
        fi
    done
    sleep 10  # Check every 10 seconds
done

# Check job status and re-submit failed jobs with exclusive resources
for job_id in "${job_ids[@]}"; do
    status=$(sacct -j $job_id --format=State --noheader | head -n 1 | awk '{print $1}')
    if [[ $status != "COMPLETED" ]]; then
        log=$(sacct -j $job_id --format=JobName%256 --noheader | sed 's/plot_//g' | awk '{print "../../benchmarks/default/" $1 "_klog"}')
        job_name="retry_plot_$(basename $log)"
        slurm_file="slurm_out/${job_name}.out"

        # Resubmit the failed job with exclusive resources
        sbatch --job-name=$job_name --output=$slurm_file --partition=hsw --time=1:00:00  --exclusive --wrap="python3 fault_plot.py $log"
    fi
done

