#!/bin/bash -xe

# User-configurable flags to enable/disable specific benchmarks
ENABLE_BASE=true
ENABLE_UVM=true
ENABLE_HMM=true

datasets=("bcsstk34" "bcsstk38" "bcsstk37" "ct20stif" "pwtk" "Cube_Coup_dt6" "GAP-twitter")

module load cuda

function setup_environment {
	# Parameters
	local eval_type=$1 # This should be either 'uvm' or 'hmm'

	# Finds root directory of the git repo; 
	ROOTDIR=`echo "${PWD%hmm-eval*}hmm-eval"`
	FILENAME=cublas_spmv.csv

	BENCHMARK_DIR=$ROOTDIR/benchmarks/apps/$eval_type/spmv/
	BENCHMARK_EXE=$BENCHMARK_DIR/spmv

	PLOT_SCRIPT=$ROOTDIR/tools/plot/spmv_perf.py

	DATA_DIR=$ROOTDIR/data/`hostname`/app_perf/$eval_type/spmv/

	OUT_PATH=$DATA_DIR/$FILENAME

	FIGS_DIR=$ROOTDIR/figs/`hostname`/app_perf/$eval_type/spmv/

	CSV_HEADER="Type,Size,Time,GFlops"

	rm -rf $DATA_DIR $FIGS_DIR

	mkdir -p $DATA_DIR
	#mkdir -p $FIGS_DIR
	
	#make -C $BENCHMARK_DIR

	echo $CSV_HEADER > $OUT_PATH
}

function run_benchmark {
	local ALL_CPUS=`lscpu | grep "On-line CPU" | awk '{print $4}'`

	cd $BENCHMARK_DIR
	echo $BENCHMARK_DIR

	#bash get_datasets.sh
	make

	# changing array sizes for iterations
	for N in "${datasets[@]}"; do
	    numactl --physcpubind=$ALL_CPUS ./spmv data/$N.mtx >> $OUT_PATH
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
