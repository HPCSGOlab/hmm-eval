#include <iostream>
#include <stdlib.h>
#include <stdio.h>

__global__ void inc(int* a)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    a[i] += 1;
}

#define BLOCKS 1024
#define TPB 256
#define NUM_THREADS (BLOCKS * TPB)
int main(void)
{
    int* a = (int*) malloc(sizeof(int) * NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; ++i)
    {
        a[i] = 3;
    }
    inc<<<BLOCKS, TPB>>>(a);
    cudaDeviceSynchronize();
    for (int i = 0; i < 20; ++i)
    {
        printf("a[%d] = %d\n", i, a[i]);
    }
}
