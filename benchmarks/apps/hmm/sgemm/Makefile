CC = g++
NVCC = nvcc
CFLAGS = -Wall -Wextra -O3 -g -fopenmp 
LDFLAGS = -lopenblas
NVFLAGS = -lcublas -Xcompiler "${CFLAGS}" -arch=sm_86

all: sgemm huge 

sgemm: sgemm.cu
	$(NVCC) -o sgemm sgemm.cu $(LDFLAGS) $(NVFLAGS)

huge: sgemm_huge.cu
	$(NVCC) -o sgemm_huge sgemm_huge.cu $(LDFLAGS) $(NVFLAGS)

clean:
	rm -f sgemm
	rm -f sgemm_huge 
