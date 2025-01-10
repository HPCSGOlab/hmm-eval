#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "mmio.h"

#define BlockDim 1024
#define MAX_NUM_THREADS_PER_BLOCK 1024
#define ITER 3

template <typename T>
__global__ void spmv_csr_scalar_kernel(T * d_val, T * d_vector, int * d_cols, int * d_ptr, int N, T * d_out) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
		T t = 0;
		int start = d_ptr[i];
		int end = d_ptr[i + 1];
		for (int j = start; j < end; j++) {
			int col = d_cols[j];
			t += d_val[j] * d_vector[col];
		}
		d_out[i] = t;
	}
}

template <typename T, int THREADS_PER_VECTOR, int MAX_NUM_VECTORS_PER_BLOCK>
__global__ void spmv_light_kernel(int* cudaRowCounter, int* d_ptr, int* d_cols,T* d_val, T* d_vector, T* d_out,int N) {
	int i;
	T sum;
	int row;
	int rowStart, rowEnd;
	int laneId = threadIdx.x % THREADS_PER_VECTOR; //lane index in the vector
	int vectorId = threadIdx.x / THREADS_PER_VECTOR; //vector index in the thread block
	int warpLaneId = threadIdx.x & 31;	//lane index in the warp
	int warpVectorId = warpLaneId / THREADS_PER_VECTOR;	//vector index in the warp

	__shared__ volatile int space[MAX_NUM_VECTORS_PER_BLOCK][2];

	// Get the row index
	if (warpLaneId == 0) {
		row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
	}
	// Broadcast the value to other threads in the same warp and compute the row index of each vector
		row = __shfl_sync(0xffffffff,row, 0) + warpVectorId;
	
	while (row < N) {

		// Use two threads to fetch the row offset
		if (laneId < 2) {
			space[vectorId][laneId] = d_ptr[row + laneId];
		}
		rowStart = space[vectorId][0];
		rowEnd = space[vectorId][1];

		sum = 0;
		// Compute dot product
		if (THREADS_PER_VECTOR == 32) {

			// Ensure aligned memory access
			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

			// Process the unaligned part
			if (i >= rowStart && i < rowEnd) {
				sum += d_val[i] * d_vector[d_cols[i]];
			}

				// Process the aligned part
			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += d_val[i] * d_vector[d_cols[i]];
			}
		} else {
			for (i = rowStart + laneId; i < rowEnd; i +=
					THREADS_PER_VECTOR) {
				sum += d_val[i] * d_vector[d_cols[i]];
			}
		}
		// Intra-vector reduction
		for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
				sum += __shfl_down_sync(0xffffffff,sum, i);
		}

		// Save the results
		if (laneId == 0) {
			d_out[row] = sum;
		}

		// Get a new row index
		if(warpLaneId == 0){
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
		}
		// Broadcast the row index to the other threads in the same warp and compute the row index of each vector
			row = __shfl_sync(0xffffffff,row, 0) + warpVectorId;

	}
}


template <typename T>
void spmv_light(MatrixInfo<T> * mat,T *vector,T *out)
{
    	T *d_vector,*d_val, *d_out;
    	int *d_cols, *d_ptr;
    	float time_taken;
    	double gflop = 2 * (double) mat->nz / 1e9;
    	int meanElementsPerRow = mat->nz/mat->M;
    	int *cudaRowCounter;

    	// Allocate memory on device
    	cudaMalloc(&d_vector,mat->N*sizeof(T));
    	cudaMalloc(&d_val,mat->nz*sizeof(T));
    	cudaMalloc(&d_out,mat->M*sizeof(T));
    	cudaMalloc(&d_cols,mat->nz*sizeof(int));
    	cudaMalloc(&d_ptr,(mat->M+1)*sizeof(int));
    	cudaMalloc(&cudaRowCounter, sizeof(int));

	// Copy from host memory to device memory
    	cudaMemcpy(d_vector,vector,mat->N*sizeof(T),cudaMemcpyHostToDevice);
    	cudaMemcpy(d_val,mat->val,mat->nz*sizeof(T),cudaMemcpyHostToDevice);
    	cudaMemcpy(d_cols,mat->cIndex,mat->nz*sizeof(int),cudaMemcpyHostToDevice);
    	cudaMemcpy(d_ptr,mat->rIndex,(mat->M+1)*sizeof(int),cudaMemcpyHostToDevice);
    	cudaMemset(d_out, 0, mat->M*sizeof(T));
    	cudaMemset(cudaRowCounter, 0, sizeof(int));

	// Choose the vector size depending on the NNZ/Row, run the kernel and time it
	if (0) {
		printf("HERE\n");
		if (meanElementsPerRow <= 2) {
			for (int i = 0; i < ITER; i++) {
				spmv_light_kernel<T, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<ceil(mat->M/(float)BlockDim), BlockDim>>>(
					cudaRowCounter, d_ptr, d_cols,d_val,d_vector,d_out,mat->M);
				cudaMemset(cudaRowCounter, 0, sizeof(int));
			}
		} else if (meanElementsPerRow <= 4) {
			for (int i = 0; i < ITER; i++) {
				spmv_light_kernel<T, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<ceil(mat->M/(float)BlockDim), BlockDim>>>(
					cudaRowCounter, d_ptr, d_cols,d_val, d_vector, d_out,mat->M);
				cudaMemset(cudaRowCounter, 0, sizeof(int));
			}
		} else if(meanElementsPerRow <= 64) {
			for (int i = 0; i < ITER; i++) {
				spmv_light_kernel<T, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<ceil(mat->M/(float)BlockDim), BlockDim>>>(
					cudaRowCounter,d_ptr,d_cols,d_val, d_vector, d_out,mat->M);
				cudaMemset(cudaRowCounter, 0, sizeof(int));
			}
		} else {
			for (int i = 0; i < ITER; i++){
				spmv_light_kernel<T, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<ceil(mat->M/(float)BlockDim), BlockDim>>>(
					cudaRowCounter, d_ptr, d_cols,d_val, d_vector, d_out,mat->M);
				cudaMemset(cudaRowCounter, 0, sizeof(int));
			}
		}
	}
	else {
		for (int i = 0; i < ITER; i++)
			spmv_csr_scalar_kernel<T><<<ceil(mat->M/(float)BlockDim), BlockDim>>>(d_val, d_vector, d_cols, d_ptr,mat->M,d_out);
	}

    		// Copy from device memory to host memory
    	cudaMemcpy(out, d_out, mat->M*sizeof(T), cudaMemcpyDeviceToHost);
    	
	// Free device memory	
		// Calculate and print out GFLOPs and GB/s
	
}
