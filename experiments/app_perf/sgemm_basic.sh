#!/bin/bash -xe

# User-configurable flags to enable/disable specific benchmarks
ENABLE_BASE=true
ENABLE_UVM=true
ENABLE_HMM=true

module load cuda

PSIZES=( 4096 8192 16384 32768 65536)

function setup_environment {
    # Parameters
    local eval_type=$1  # This should be either 'uvm' or 'hmm'

    # Finds the root directory of the git repo; 
    ROOTDIR=`echo "${PWD%hmm-eval*}hmm-eval"`
    FILENAME=cublas_results.csv

    BENCHMARK_DIR=$ROOTDIR/benchmarks/apps/$eval_type/sgemm/
    BENCHMARK_EXE=$BENCHMARK_DIR/sgemm

    # post-processing
    PLOT_SCRIPT=$ROOTDIR/tools/plot/sgemm_perf.py

    # directory for data
    DATA_DIR=$ROOTDIR/data/`hostname`/app_perf/$eval_type/sgemm/

    # data output
    OUT_PATH=$DATA_DIR/$FILENAME

    # figure output directory
    FIGS_DIR=$ROOTDIR/figs/`hostname`/app_perf/$eval_type/sgemm/

    CSV_HEADER="Type,Size,Time,GFLOPs"

    # Prepare environment
    # Remove old CSV files if they exist
    rm -rf $DATA_DIR $FIGS_DIR

    # make data dirs
    mkdir -p $DATA_DIR
#    mkdir -p $FIGS_DIR

    # build benchmark if it hasn't been compiled yet
    make -C $BENCHMARK_DIR

    # Initialize CSV files with headers
    echo $CSV_HEADER > $OUT_PATH
}

function run_benchmark {
    local ALL_CPUS=`lscpu | grep "On-line CPU" | awk '{print $4}'`

    for N in ${PSIZES[@]}; do
        numactl --physcpubind=$ALL_CPUS $BENCHMARK_EXE -n $N -i 1 >> $OUT_PATH
    done

#    python3  $PLOT_SCRIPT $OUT_PATH -o $FIGS_DIR
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
