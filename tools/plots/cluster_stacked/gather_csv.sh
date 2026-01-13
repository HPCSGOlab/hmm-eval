#!/bin/bash


# Output CSV file
csv_file="benchmark_phases.csv"

# Write CSV header
echo "benchmark,group,phase,value" > "$csv_file"

# Mapping of tags to human-readable phases
declare -A phase_map=(
    [a]="UVM alloc"
    [b]="UVM finalize/map"
    [s]="HMM setup"
    [f]="UVM alloc"
    [p]="HMM pages"
    [m]="UVM finalize/map"
    [v]="HMM finalize"
)

input_dir=~/hmm-eval/experiments/migr_timings/detailed_timings/post_data

for file in "$input_dir"/*.txt; do
	echo $(basename "$file")
	filename=$(basename "$file") 

	if [[ "$filename" == *sgemm* ]]; then
		echo "SGEMM"
		benchmark=SGEMM
	else
		echo "something else"
	fi

	if [[ "$filename" == uvm* ]]; then
		echo "UVM"
		group=UVM
	else
		echo "HMM"
		group="Native HMM"
	fi

# Read file line by line, skipping the header
	grep -E "^[[:space:]]+[a-z]:" "$file" | while read -r line; do
		tag=$(echo "$line" | awk '{print $1}' | tr -d ':')
		value=$(echo "$line" | awk '{print $2}' | cut -d'.' -f1) # remove decimal part
		phase=${phase_map[$tag]}
		echo "${benchmark},${group},${phase},${value}" >> "$csv_file"
	done
done
