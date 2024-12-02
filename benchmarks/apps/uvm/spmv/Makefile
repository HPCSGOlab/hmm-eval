CC = nvcc
objects = mmio.o main.o

spmv: $(objects)
	$(CC) -arch=sm_90 $(objects) -o spmv

mmio.o: src/mmio.c
	$(CC) -x cu -arch=sm_90 -Iinclude -c src/mmio.c

#src/spmv_csr_scalar.cu src/spmv_csr_vector.cu src/spmv_csr_adaptive.cu src/spmv_pcsr.cu
main.o:  src/spmv_light.cu src/spmv_cpu.cu src/utils.cu src/main.cu
	$(CC) -arch=sm_90 -Iinclude -I. -dc src/main.cu

clean:
	rm *.o
	rm spmv
