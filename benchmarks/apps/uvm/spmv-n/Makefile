CC = g++
NVCC = nvcc
CFLAGS = -Wall -Wextra -O3 -g -fopenmp 
LDFLAGS = -lopenblas
NVFLAGS = -lcublas -Xcompiler "${CFLAGS}" -arch=sm_86

all: spmv 

spmv: spmv.cu
	$(NVCC) -o spmv spmv.cu $(LDFLAGS) $(NVFLAGS)

clean:
	rm -f spmv 
