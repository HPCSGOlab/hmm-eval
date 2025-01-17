#!/bin/bash -xe

#for i in "vanilla" "faults" "faults-new"; do
for i in  "faults-new"; do
    make -j -C ~/dev/uvm-eviction/drivers/x86_64-555.42.02/${i}/kernel-open &> ${i}-kbuild.log
    sudo rmmod -f nvidia-uvm || true
    sudo insmod ~/dev/uvm-eviction/drivers/x86_64-555.42.02/${i}/kernel-open/nvidia-uvm.ko uvm_perf_prefetch_enable=0 
    
    if [ "$i" == "faults-new" ]; then
        ../../tools/sysloggerv2/log "junk.txt" &
        pid=$!
    fi

    ./matrixMul2 -wA=32768 -hA=32768 -wB=32768 -hB=32768 |& tee ${i}.log


    if [ "$i" == "faults-new" ]; then
        kill $pid
    fi
done

sudo rmmod nvidia-uvm
