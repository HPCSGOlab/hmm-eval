#!/bin/bash -x

# Create the perf directory if it doesn't exist
mkdir -p perf

# Initialize perf.csv
echo "Problem Size,Allocator Mode,Runtime" > perf/perf.csv
echo "Problem Size,Allocator Mode,Runtime" > perf/perf-noprefetch.csv

# Array of olmalloc modes
modes=(0 1 2 3 4 5)

# Function to run the benchmark
run_benchmarks () {
  # Run the benchmark for various problem sizes and olmalloc modes
  for size in $(seq 204800000 204800000 1024000000); do
    for mode in "${modes[@]}"; do
      # Run the benchmark and store the output runtime
      runtime=$(./stream $size $mode 2>&1 >/dev/null)
      # Append the result to the CSV file
      echo "$size,$mode,$runtime" >> $1
    done
  done
}

# Run the benchmarks with normal settings
run_benchmarks perf/perf.csv

# Disable prefetching and run the benchmarks
sudo systemctl stop nvidia-persistenced
sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm uvm_perf_prefetch_enable=0
sudo systemctl start nvidia-persistenced

run_benchmarks perf/perf-noprefetch.csv

# Unload the nvidia-uvm driver to reset the system state
sudo rmmod nvidia-uvm

