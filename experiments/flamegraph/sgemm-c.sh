#!/bin/bash -xe

module load cuda

types=("sgemm" "sgemm-c" "sgemm-c-g")

mkdir -p ../../figs/flamegraph/`hostname`/

for N in ${types[@]}; do
	cd ../../benchmarks/apps/hmm/$N
	make

	echo $N

	cd ../../../../experiments/flamegraph/FlameGraph

	perf record -F 99 -a -g ./../../../benchmarks/apps/hmm/$N/sgemm -n 65536
	perf script | ./stackcollapse-perf.pl > out.perf-folded
	./flamegraph.pl out.perf-folded > sgemm_$N.svg
	mv sgemm_$N.svg ../../../figs/flamegraph/`hostname`/

	cd ../
done
