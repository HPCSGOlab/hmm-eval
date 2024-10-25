#!/bin/bash -xe

# User-configurable flags to enable/disable specific benchmarks
ENABLE_BASE=true
ENABLE_UVM=true
ENABLE_HMM=true

PSIZES=(268435456 536870912 1073741824)
ITERS=100

module load cuda

function setup_environment {
	# Parameters
	local eval_type=$1 # This should be either 'uvm' or 'hmm'

	# Finds root directory of the git repo; 
	ROOTDIR=`echo "${PWD%hmm-eval*}hmm-eval"`
	FILENAME=cublas_stream.csv

	BENCHMARK_DIR=$ROOTDIR/benchmarks/apps/$eval_type/stream/
	BENCHMARK_EXE=$BENCHMARK_DIR/cuda-stream

	PLOT_SCRIPT=$ROOTDIR/tools/plot/stream_perf.py

	DATA_DIR=$ROOTDIR/data/`hostname`/app_perf/$eval_type/stream/

	OUT_PATH=$DATA_DIR/$FILENAME

	FIGS_DIR=$ROOTDIR/figs/`hostname`/app_perf/$eval_type/stream/

	CSV_HEADER="Type,Size,AverTime,GBytes/s"

	rm -rf $DATA_DIR $FIGS_DIR

	mkdir -p $DATA_DIR
	mkdir -p $FIGS_DIR
	
	make -C $BENCHMARK_DIR

	echo $CSV_HEADER > $OUT_PATH
}

function run_benchmark {
	# MEMMODE=0,1
	# changing array sizes for iterations
	for N in "${PSIZES[@]}"; do
	    make -C $BENCHMARK_DIR clean

	    # compile and run stream with new array size
	    make -C $BENCHMARK_DIR STREAM_ARRAY_SIZE=$N ITERS=$ITERS
	    $BENCHMARK_EXE -n $ITERS >> $OUT_PATH
	    # $BENCHMARK_EXE -n 1 -c 2 >> $OUT_PATH
	done
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
