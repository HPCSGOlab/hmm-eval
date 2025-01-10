#!/bin/bash -xe

module load cuda

types=("base" "uvm" "hmm")

#rm -rf ../../figs/flamegraph/`hostname`/

mkdir -p ../../figs/flamegraph/`hostname`/


for N in ${types[@]}; do
	cd ../../benchmarks/apps/$N/spmv
	make

	cd ../../../../experiments/flamegraph/FlameGraph

	perf record -F 99 -a -g ./../../../benchmarks/apps/$N/spmv/spmv ../../../benchmarks/apps/$N/spmv/data/pwtk.mtx
	perf script | ./stackcollapse-perf.pl > out.perf-folded
	./flamegraph.pl out.perf-folded > spmv_$N.svg
	mv spmv_$N.svg ../../../figs/flamegraph/`hostname`/

	cd ../
done
