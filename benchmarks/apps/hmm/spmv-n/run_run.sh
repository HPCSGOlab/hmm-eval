#!/bin/bash

DATA_DIR="data"

for file in "$DATA_DIR"/*.mtx; do
    name=$(basename "$file" .mtx)
    echo "Running $name"
    bash iter_run.sh "$name"
done

