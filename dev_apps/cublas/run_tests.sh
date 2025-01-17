#!/bin/bash -x

# Initialize CSV files with headers
echo "Type,Size,Time,GFLOPs" > cublas_results.csv
echo "Type,Size,Time,GFLOPs" > cblas_results.csv

# Loop through problem sizes
for N in 8192 16384 32768 65536; do
        # Run with CUBLAS and append results to CSV
        ./sgemm -n $N >> cublas_results.csv

        # Run with CBLAS and append results to CSV
        ./sgemm -n $N -c >> cblas_results.csv
done

# Run the Python script to generate the plot
python plot_results.py

