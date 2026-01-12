CC = g++
NVCC = nvcc
CFLAGS = -Wall -Wextra -O3 -g -fopenmp 
LDFLAGS = -lopenblas
NVFLAGS = -lcublas -Xcompiler "${CFLAGS}" -arch=sm_86

all: sgemm basic

sgemm: sgemm.cu
	$(NVCC) -o sgemm sgemm.cu $(LDFLAGS) $(NVFLAGS)

basic: basic_sgemm.cu
	$(NVCC) -o basic_sgemm basic_sgemm.cu $(LDFLAGS) $(NVFLAGS)

clean:
	rm -f sgemm
