#!/bin/bash

# Base directory containing subdirectories with klog files
base_dir="../../benchmarks/default/"
log_dir="slurm_out/"

# Ensure the log directory exists, create it if it does not
mkdir -p "$log_dir"

# Loop through each subdirectory under the base directory
for subdir in "$base_dir"*/; do
    # Ensure the subdir path is valid and check for the presence of klog files
    klog_files=$(find "$subdir" -name '*_klog')
    if [ -z "$klog_files" ]; then
        continue
    fi

    # Initialize variables to store file paths, job names, and slurm output paths
    pf_files=""
    nopf_files=""
    pf_sizes=()
    nopf_sizes=()
    benchmark=$(basename "$subdir")

    # Categorize files into with and without "nopf" and extract sizes
    for file in $klog_files; do
        if [[ "$file" =~ .*_nopf_([0-9]+)_.* ]]; then
            nopf_files+="$file "
            nopf_sizes+=("${BASH_REMATCH[1]}")
        elif [[ "$file" =~ .*_([0-9]+)_.* && "$file" != *"nopf"* ]]; then
            pf_files+="$file "
            pf_sizes+=("${BASH_REMATCH[1]}")
        fi
    done

    # Sort and create a dash-separated string of problem sizes
    IFS=$'\n' sorted_pf_sizes=($(sort -nu <<<"${pf_sizes[*]}"))
    IFS=$'\n' sorted_nopf_sizes=($(sort -nu <<<"${nopf_sizes[*]}"))
    pf_sizes_string=$(IFS=-; echo "${sorted_pf_sizes[*]}")
    nopf_sizes_string=$(IFS=-; echo "${sorted_nopf_sizes[*]}")
    job_name_pf="plot_scaling_${benchmark}_pf_${pf_sizes_string}"
    job_name_nopf="plot_scaling_${benchmark}_nopf_${nopf_sizes_string}"

    slurm_file_pf="${log_dir}output_${benchmark}_pf_${pf_sizes_string}.txt"
    slurm_file_nopf="${log_dir}output_${benchmark}_nopf_${nopf_sizes_string}.txt"

    # Trim the trailing space
    pf_files=$(echo "$pf_files" | sed 's/ *$//')
    nopf_files=$(echo "$nopf_files" | sed 's/ *$//')

    # Check if the number of problem sizes is less than 5
    if [[ ${#sorted_pf_sizes[@]} -lt 5 || ${#sorted_nopf_sizes[@]} -lt 5 ]]; then
        echo "Skipping $subdir due to insufficient problem sizes."
        continue
    fi

    # Echo the results for the current subdirectory
    echo "Regular files in $benchmark: $pf_files"
    echo "Nopf files in $benchmark: $nopf_files"
    echo "Job name for PF: $job_name_pf"
    echo "Job name for NOPF: $job_name_nopf"

    # Submit the jobs
    sbatch --parsable -t 12:00:00 --job-name="$job_name_pf" --output="$slurm_file_pf" --partition=hsw --exclusive --wrap="python3 fault_scaling_plot.py $pf_files"
    sbatch --parsable -t 12:00:00 --job-name="$job_name_nopf" --output="$slurm_file_nopf" --partition=hsw --exclusive --wrap="python3 fault_scaling_plot.py $nopf_files"
done

