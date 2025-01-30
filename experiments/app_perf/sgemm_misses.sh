#!/bin/bash -xe

module load cuda

#removed base for now
types=("uvm" "hmm")

cd ../../benchmarks/apps

for N in ${types[@]}; do
	cd $N/sgemm
	make

	echo ---------------------$N---------------------------

	#perf stat -e ls_tlb_flush.all,dTLB-loads,dTLB-load-misses,ls_l1_d_tlb_miss.all,ign_rd_wr_mmio_1ff8h ./sgemm -n 65536
	perf stat -e xen:xen_mmu_flush_tlb_multi ./sgemm -n 65536

	cd ../../
done
