#!/bin/bash -x

module load cuda

# check if an argument is provided
if [ $# -eq 0 ]; then
	echo "Usage: $0 <hmm, uvm, or base>"
	exit 1
fi

TYPE=$1

ROOT_DIR=`echo "${PWD%hmm-eval*}hmm-eval"`
#cd $ROOT_DIR/benchmarks/apps/$TYPE/TeaLeaf-master
#PROGRAM="./build/cuda-tealeaf"
cd $ROOT_DIR/driver_apps/linear
PROGRAM=./$1

#65536
ARGS="" 

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

BPFTRACE_SCRIPT="
kprobe:zap_pte_range /pid == $UVM_PID/ {
	printf(\"start: %lu, end: %lu\\n\", (arg3), (arg4));
}
"

echo "Tracing native_flush_tlb_multi for PID $UVM_PID..."
sudo bpftrace -I /usr/src/linux-hwe-6.8-headers-6.8.0-49/arch/x86/include -e "$BPFTRACE_SCRIPT" &> $ROOT_DIR/experiments/kernel_perf/bpf_addr/linear_$TYPE

#wait $PROGRAM_ID
