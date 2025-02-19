#!/bin/bash -xe

module load cuda

ROOTDIR=`echo "${PWD%hmm-eval*}hmm-eval"`

cd $ROOTDIR/drivers/x86_64-560.35.05/exp/kernel-open
make

sudo rmmod nvidia-uvm
sudo insmod nvidia-uvm.ko

echo "--------------NICK-----------"

cd $ROOTDIR/benchmarks/apps/hmm/sgemm
make

./sgemm -n 65536

sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm

echo "---------------HMM-------------"

./sgemm -n 65536
