#!/bin/bash -xe

module load cuda

types=("base" "uvm" "hmm")

cd ../../benchmarks/apps

for N in ${types[@]}; do
	cd $N/sgemm
	make

	echo ---------------------$N---------------------------

	perf stat -e cache-misses,L1-dcache-misses,faults ./sgemm -n 65536

	cd ../../
done
