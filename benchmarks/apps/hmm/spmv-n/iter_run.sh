#!/bin/bash

# Define the application and output file
APP="./basic-spmv data/$1.mtx"  # Replace with the actual application path
OUTPUT_FILE="resultsdir/$1tlb.txt"

# Run the application 10 times and append results
for i in {1..10}; do
    echo "Run #$i:" >> "$OUTPUT_FILE"
    $APP >> "$OUTPUT_FILE" 2>&1
    echo "--------------------------------------" >> "$OUTPUT_FILE"
done

echo "Execution completed. Results stored in $OUTPUT_FILE"
