#!/bin/bash -ex
module load cuda
make
echo "File Size (MB),Access Pattern,CPU Time (s),GPU Time (s)" > results.csv
# Loop over various problem sizes
for size in 1 10 100 1000 10000; do
    # Check if the files exist, and if not, generate them and move to inputs/
    if [[ ! -f "inputs/cpu_mmap_${size}" ]] || [[ ! -f "inputs/gpu_mmap_${size}" ]]; then
        ./filegen -s "${size}"
        mv "cpu_mmap_${size}" inputs/
        mv "gpu_mmap_${size}" inputs/
    fi
    echo "Pre-checking cpu file correctness:"
    python3 checker.py "inputs/cpu_mmap_${size}" 0
    echo "Pre-checking gpu file correctness:"
    python3 checker.py "inputs/gpu_mmap_${size}" 0

    val=1
    for access_mode in "linear" "random"; do
        sync
        echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
        ./benchmark --size=${size} --access=${access_mode} >> results.csv
        echo "Checking cpu correctness:"
        python3 checker.py "inputs/cpu_mmap_${size}" ${val}
        echo "Checking gpu correctness:"
        python3 checker.py "inputs/gpu_mmap_${size}" ${val}
        val=$(( $val + 1 ))
    done
done

python3 plot_results.py results.csv
