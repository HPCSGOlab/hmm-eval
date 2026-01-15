#!/bin/bash

# Define the application and output file
APP="./sgemm -n $1"  # Replace with the actual application path
OUTPUT_FILE="resultsdir/$1overlap.txt"

# Run the application 10 times and append results
for i in {1..10}; do
    echo "Run #$i:" >> "$OUTPUT_FILE"
    $APP >> "$OUTPUT_FILE" 2>&1
    echo "--------------------------------------" >> "$OUTPUT_FILE"
done

echo "Execution completed. Results stored in $OUTPUT_FILE"
