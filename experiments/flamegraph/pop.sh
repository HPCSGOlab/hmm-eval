#!/bin/bash -xe

module load cuda

types=("uvm" "hmm")

mkdir -p ../../figs/flamegraph/`hostname`/

for N in ${types[@]}; do
	cd ../../benchmarks/apps/$N/pop
	make

	cd ../../../../experiments/flamegraph/FlameGraph

	numgig=10
	gig=$((268435456 * numgig))
	perf record -F 99 -a -g ./../../../benchmarks/apps/$N/pop/pop -n $gig 
	perf script | ./stackcollapse-perf.pl > out.perf-folded
	./flamegraph.pl out.perf-folded > pop_$N.svg
	mv pop_$N.svg ../../../figs/flamegraph/`hostname`/

	cd ../
done
