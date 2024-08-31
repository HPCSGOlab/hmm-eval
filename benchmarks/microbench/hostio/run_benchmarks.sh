#!/bin/bash -ex

make
echo "File Size (MB),Access Pattern,Read Time (s),MMap Time (s)" > results.csv
# Loop over various problem sizes
for size in 1 10 100 1000 10000; do
    # Check if the files exist, and if not, generate them and move to inputs/
    if [[ ! -f "inputs/filename_mmap_${size}" ]] || [[ ! -f "inputs/filename_read_${size}" ]]; then
        ./filegen -s "${size}"
        mv "filename_mmap_${size}" inputs/
        mv "filename_read_${size}" inputs/
    fi

    for access_mode in "linear" "random"; do
        sync
        echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
        ./benchmark --size=${size} --access=${access_mode} >> results.csv
    done
done

python3 plot_results.py results.csv
