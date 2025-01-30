#!/bin/bash -xe

ROOTDIR=`echo "${PWD%hmm-eval*}hmm-eval"`

cd $ROOTDIR/drivers/x86_64-560.35.05/exp/kernel-open
make

sudo rmmod nvidia-uvm
sudo insmod nvidia-uvm.ko

types=("base" "uvm" "hmm")

for N in ${types[@]}; do
	cd $ROOTDIR/benchmarks/apps/$N/sgemm
	make

	./sgemm -n 65536
done

cd $ROOTDIR/drivers/x86_64-560.35.05/exp/kernel-open
sudo rmmod nvidia-uvm
sudo insmod nvidia-uvm.ko uvm_perf_prefetch_enable=0

for N in ${types[@]}; do
	cd $ROOTDIR/benchmarks/apps/$N/sgemm
	make

	./sgemm -n 65536
done
