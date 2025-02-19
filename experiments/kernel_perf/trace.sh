#!/bin/bash -x

module load cuda

# check if an argument is provided
if [ $# -eq 0 ]; then
	echo "Usage: $0 <hmm, uvm, or base>"
	exit 1
fi

TYPE=$1

ROOT_DIR=`echo "${PWD%hmm-eval*}hmm-eval"`
cd $ROOT_DIR/driver_apps/linear
PROGRAM=./$1

ARGS="" 

sudo echo > /sys/kernel/debug/tracing/trace

sudo echo function_graph > /sys/kernel/debug/tracing/current_tracer

$PROGRAM $ARGS &
PROGRAM_PID=$!

# Wait for the target process to start
TARGET_PROCESS="UVM GPU1 BH"
echo "Waiting for process '$TARGET_PROCESS' to start..."
while true; do
    UVM_PID=$(pgrep -x "$TARGET_PROCESS")
    if [ -n "$UVM_PID" ]; then
        echo "Process '$TARGET_PROCESS' started with PID $UVM_PID."
        break
    fi
done

sudo echo $UVM_PID > /sys/kernel/debug/tracing/set_ftrace_pid

sudo echo 1 > /sys/kernel/debug/tracing/tracing_on

wait $PROGRAM_PID

sudo echo 0 > /sys/kernel/debug/tracing/tracing_on

sudo cat /sys/kernel/debug/tracing/trace > $ROOT_DIR/experiments/kernel_perf/trace_$1.txt
