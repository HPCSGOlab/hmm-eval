#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define PAGE_SIZE 4096
#define VA_BLOCK_SIZE 512 * PAGE_SIZE
#define NUM_PAGES 1024
#define ARRAY_SIZE (PAGE_SIZE / sizeof(float)) * NUM_PAGES

#define THREADS 64
#define BLOCKS (1 + (NUM_PAGES / THREADS)) 

extern "C"
__global__ void faults(float* arr) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	idx = idx * ((VA_BLOCK_SIZE / 2) / sizeof(float));

	if (idx < ARRAY_SIZE)
		arr[idx] += (float)idx;
}

extern "C"
__global__ void stupid() {
	return;
}

int main() {
	float *array;
	//array = (float *) malloc(sizeof(float) * ARRAY_SIZE);
	cudaMallocManaged(&array, ARRAY_SIZE * sizeof(float));

	for (size_t i = 0; i < ARRAY_SIZE; i++) {
		array[i] = 0.0;
	}

	stupid<<<1,1>>>();
	cudaDeviceSynchronize();

	sleep(5);

	cudaEvent_t start;
        cudaEventCreate(&start);

        cudaEvent_t stop;
        cudaEventCreate(&stop);

        cudaEventRecord(start, NULL);

	faults<<<BLOCKS, THREADS>>>(array);	
	cudaDeviceSynchronize();

        cudaEventRecord(stop, NULL);

	cudaEventSynchronize(stop);
        cudaDeviceSynchronize();

	float msecTotal = 0.0f;
	cudaEventElapsedTime(&msecTotal, start, stop);

        // should be pages / sec
        printf("perf,%lf\n", (BLOCKS * THREADS) / (msecTotal/1000.0));

	cudaFree(array);
}
