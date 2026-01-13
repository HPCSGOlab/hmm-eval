#!/bin/bash

mem=(hmm uvm)
app=sgemm
size=8192
exec=$app

mkdir -p data
mkdir -p post_data

for m in ${mem[@]}; do
       out=${m}_${app}_${size}.txt

       cd ~/hmm-eval/benchmarks/apps/$m/$app
       sudo dmesg -C
       ./$exec -n $size

       cd ~/hmm-eval/experiments/migr_timings/detailed_timings
       dmesg > data/$out

       python3 detailed_timings.py data/$out > post_data/$out
done
