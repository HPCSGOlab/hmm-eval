CC = g++
NVCC = nvcc
CFLAGS = -Wall -Wextra -O3 -g #-fopenmp 
LDFLAGS = -lopenblas
NVFLAGS = -lcublas -Xcompiler "${CFLAGS}" -arch=sm_90

all: sgemm

sgemm: sgemm.cu
	$(NVCC) -o sgemm sgemm.cu $(LDFLAGS) $(NVFLAGS)

clean:
	rm -f sgemm
