#!/bin/bash -xe

module load cuda

types=("base" "uvm" "hmm")

cd ../../benchmarks/apps

for N in ${types[@]}; do
	cd $N/sgemm
	make

	echo ---------------------$N---------------------------
	
	perf stat -e ls_dispatch.ld_st_dispatch,ls_misal_loads.ma4k,ls_tlb_flush.all,dTLB-loads,dTLB-load-misses ~/hmm-eval/benchmarks/apps/$N/sgemm/sgemm -n 65536	
	
	cd ../../

done
