#!/bin/bash -xe

CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc
CUDA_ARCH=sm_90

cmake -Bbuild -H. -DMODEL=CUDA -DENABLE_MPI=OFF -DCMAKE_CUDA_COMPILER=$CUDA_COMPILER -DCUDA_ARCH=$CUDA_ARCH 

cmake --build build

#RUN EXPERIMENT WITH: 
#	./build/cuda-tealeaf
#	To get input correct
#		cp Benchmarks/<.in file> tea.in

