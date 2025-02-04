#!/bin/bash -x

module load cuda

# check if an argument is provided
if [ $# -eq 0 ]; then
	TYPE="hmm"
else
	TYPE=$1
fi

ROOT_DIR=`echo "${PWD%hmm-eval*}hmm-eval"`
cd $ROOT_DIR/benchmarks/apps/$TYPE/sgemm
PROGRAM="./sgemm"

#65536
ARGS="-n $(expr 4096 \* 16)" 

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
#include <linux/migrate.h>
kprobe:migrate_vma_setup /pid == $UVM_PID/ {
    printf(\"start: %lu, end: %lu\\n\", ((struct migrate_vma *)arg0)->start, ((struct migrate_vma *)arg0)->end);
}
"

echo "Tracing migrate_vma_setup_locked for PID $UVM_PID..."

#IF YOU ARE GETTING STRUCT NOT FOUND ERROR IT IS FROM HERE
# THERE HAS GOT TO BE A BETTER WAY OF DOING THIS

sudo bpftrace -I /usr/src/linux-hwe-6.8-headers-6.8.0-49/arch/x86/include -e "$BPFTRACE_SCRIPT" &> $ROOT_DIR/experiments/kernel_perf/bpf_addr/migrtrace_$TYPE

#wait $PROGRAM_ID
