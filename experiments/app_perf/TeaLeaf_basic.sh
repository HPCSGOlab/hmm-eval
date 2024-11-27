#!/bin/bash -xe

# User-configurable flags to enable/disable specific benchmarks
ENABLE_BASE=true
ENABLE_UVM=true
ENABLE_HMM=true

#               196KB                      382KB                   768KB            1536KB           3072KB
benchmarks=(Benchmarks/smallest.in Benchmarks/smaller.in Benchmarks/small.in Benchmarks/med.in Benchmarks/big.in)
ITERS=100

module load cuda

function setup_environment {
	# Parameters
	local eval_type=$1 # This should be either 'uvm' or 'hmm'

	# Finds root directory of the git repo; 
	ROOTDIR=`echo "${PWD%hmm-eval*}hmm-eval"`
	FILENAME=cublas_tealeaf.csv

	BENCHMARK_DIR=$ROOTDIR/benchmarks/apps/$eval_type/TeaLeaf-master/
	BENCHMARK_EXE=$BENCHMARK_DIR/build/cuda-tealeaf

	PLOT_SCRIPT=$ROOTDIR/tools/plot/tealeaf_perf.py

	DATA_DIR=$ROOTDIR/data/`hostname`/app_perf/$eval_type/tealeaf/

	OUT_PATH=$DATA_DIR/$FILENAME

	FIGS_DIR=$ROOTDIR/figs/`hostname`/app_perf/$eval_type/tealeaf/

	CSV_HEADER="Type,KB,s"

	rm -rf $DATA_DIR $FIGS_DIR

	mkdir -p $DATA_DIR
	mkdir -p $FIGS_DIR
	
	make -C $BENCHMARK_DIR

	echo $CSV_HEADER > $OUT_PATH
}

function run_benchmark {
	local ALL_CPUS=`lscpu | grep "On-line CPU" | awk '{print $4}'`

	cd $BENCHMARK_DIR
	bash build.sh

	# changing array sizes for iterations
	for N in "${benchmarks[@]}"; do
	    cp $N  tea.in

	    numactl --physcpubind=$ALL_CPUS build/cuda-tealeaf >> $OUT_PATH
	done

	cd $ROOTDIR/experiments/app_perf
}

if [ "$ENABLE_UVM" = true ]; then
    setup_environment "uvm"
    run_benchmark
fi

if [ "$ENABLE_HMM" = true ]; then
    setup_environment "hmm"
    run_benchmark
fi

if [ "$ENABLE_BASE" = true ]; then
    setup_environment "base"
    run_benchmark
fi

# python3 $PLOT_SCRIPT -o $FIGS_DIR
