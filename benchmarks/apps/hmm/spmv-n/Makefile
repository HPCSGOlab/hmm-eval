CC = g++
NVCC = nvcc
CFLAGS = -Wall -Wextra -O3 -g -fopenmp 
LDFLAGS = -lopenblas
NVFLAGS = -lcublas -Xcompiler "${CFLAGS}" -arch=sm_86

all: spmv basic

spmv: spmv.cu
	$(NVCC) -o spmv spmv.cu $(LDFLAGS) $(NVFLAGS)

basic: basic-spmv.cu
	$(NVCC) -o basic-spmv basic-spmv.cu $(LDFLAGS) $(NVFLAGS)


clean:
	rm -f spmv 
