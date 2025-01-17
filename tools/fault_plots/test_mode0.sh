#!/bin/bash -xe

MODULE_DIR=../../drivers/x86_64-555.42.02/faults-new/kernel-open
MODULE_PATH=${MODULE_DIR}/nvidia-uvm.ko
make -C $MODULE_DIR -j

sudo rmmod -f nvidia-uvm || true
sudo insmod $MODULE_PATH hpcs_log_short=0

../sysloggerv2/log "mode0.txt" &
pid=$!
../fault_gen/pager 4096
sleep 1
kill $pid

python3 parser.py mode0.txt

sudo rmmod nvidia-uvm
