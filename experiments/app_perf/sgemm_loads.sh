#!/bin/bash -xe

module load cuda

ROOTDIR=`echo "${PWD%hmm-eval*}hmm-eval"`

cd $ROOTDIR/drivers/x86_64-560.35.05/exp/kernel-open
make

sudo rmmod nvidia-uvm
sudo insmod nvidia-uvm.ko uvm_perf_prefetch_enable=0

#removed base for now
types=("uvm" "hmm")

cd $ROOTDIR/benchmarks/apps


echo ------No Prefetching---------


for N in ${types[@]}; do
	cd $N/sgemm
	make

	echo ---------------------$N---------------------------
	
	perf stat -e ls_dispatch.ld_st_dispatch,ls_tlb_flush.all,dTLB-loads $ROOTDIR/benchmarks/apps/$N/sgemm/sgemm -n 65536	
	
	cd ../../

done

cd $ROOTDIR/drivers/x86_64-560.35.05/exp/kernel-open
sudo rmmod nvidia-uvm
sudo insmod nvidia-uvm.ko #uvm_perf_prefetch_enable=0

cd $ROOTDIR/benchmarks/apps

echo -----------With Prefetching-----------

for N in ${types[@]}; do
	cd $N/sgemm
	make

	echo ---------------------$N---------------------------
	
	perf stat -e ls_dispatch.ld_st_dispatch,ls_tlb_flush.all,dTLB-loads $ROOTDIR/benchmarks/apps/$N/sgemm/sgemm -n 65536	
	
	cd ../../

done
