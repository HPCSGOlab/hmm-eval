#!/bin/bash -xe
module load cuda

ROOTDIR=`echo "${PWD%gracehopper_power*}gracehopper_power"`
FILENAME=cublas_stream

POWER_FILENAME=cublas_power
POWER_EXT=.csv

BENCHMARK_DIR=$ROOTDIR/benchmarks/stream/
BENCHMARK_EXE=$BENCHMARK_DIR/stream_openmp.exe

PLOT_SCRIPT=$ROOTDIR/tools/plot/stream_perf.py

DATA_DIR=$ROOTDIR/data/app_perf/stream/

OUT_PATH=$DATA_DIR/$FILENAME
POWER_PATH=$DATA_DIR/$POWER_FILENAME

FIGS_DIR=$ROOTDIR/figs/app_perf/stream/

POWER_SCRIPT=$ROOTDIR/tools/power/Power2.py

rm -rf $DATA_DIR $FIGS_DIR

mkdir -p $DATA_DIR
mkdir -p $FIGS_DIR

# need echo for CSV header for stream data > $OUT_PATH
array_sizes=(268435456 536870912 1073741824)
ITERS=100
# MEMMODE=0,1
# changing array sizes for iterations
for N in "${array_sizes[@]}"; do
    #start power coll in background
    python3 $POWER_SCRIPT ${POWER_PATH}_${N}$POWER_EXT &> /dev/null &
    # catch process id to kill after
    pid=$!

    make -C $BENCHMARK_DIR clean

    # compile and run stream with new array size
    make -C $BENCHMARK_DIR STREAM_ARRAY_SIZE=$N ITERS=$ITERS
    $BENCHMARK_EXE -n 1 -c 2> $OUT_PATH${N}.csv

    # end power collection
    kill $pid
done

python3 $PLOT_SCRIPT -o $FIGS_DIR
