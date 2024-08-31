#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <chrono>

template <typename T> int calculateOptimalBlocks(T kernel, int threadsPerBlock);

// olmalloc_mode determines which allocator we use
// 0 is malloc (hmm only!) (default)
// 1 is cudaMallocManaged
static int olmalloc_mode = 0;
void* olmalloc(size_t bytes)
{
    void* foo = nullptr;
    if (!olmalloc_mode)
    {
        foo = malloc(bytes);
    }
    else if (olmalloc_mode == 1)
    {
        cudaMallocManaged(&foo, bytes);
    }
    else
    {
        fprintf(stderr, "invalid olmalloc mode %d\n", olmalloc_mode);
        exit(1);
    }
    if (foo == nullptr)
    {
        fprintf(stderr, "olmalloc failed\n");
        exit(1);
    }
    return foo;
}

__global__ void warmup()
{
    return;
}

__global__ void inc(int* a, size_t n)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    for (; i < n; i += blockDim.x * gridDim.x)
    {
        a[i] += 1;
    }
}

#define TPB 256
int main(int argc, char* argv[]) {
    size_t N;
    if (argc != 2 && argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <size_t>\n";
        return 1;
    }
    try
    {
        N = std::stoul(argv[1]);
        if (argc == 3)
        {
            olmalloc_mode = std::stoi(argv[2]);
        }
    }
    catch (const std::invalid_argument& e) 
    {
        std::cerr << "Invalid argument: the input is not an unsigned integer.\n";
        return 2;
    }
    catch (const std::out_of_range& e) 
    {
        std::cerr << "Invalid argument: the input is out of range for a size_t.\n";
        return 3;
    }

    const int blocks = calculateOptimalBlocks(inc, TPB);
    int* a = (int*) olmalloc(sizeof(int) * N);

    printf("Allocating %lu bytes\n", sizeof(int) * N);
    printf("Allocating %lf gigabytes\n", sizeof(int) * N / 1e9);
    printf("Kernel Config: %d, %d\n", blocks, TPB);

    for (size_t i = 0; i < N; ++i)
    {
        a[i] = 3;
    }

    warmup<<<1, 1>>>();
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();

    inc<<<blocks, TPB>>>(a, N);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cerr << duration.count()/1e9 << std::endl;
    std::cout << "runtime: " << duration.count()/1e9 << " seconds" << std::endl;


    for (int i = 0; i < 10; ++i)
    {
        printf("a[%d] = %d\n", i, a[i]);
    }
    for (size_t i = 0; i < N; ++i)
    {
        assert(a[i] == 4);
    }
}


// thanks chatgpt
template <typename T>
int calculateOptimalBlocks(T kernel, int threadsPerBlock) {
    int device;
    cudaDeviceProp props;

    // Get the device
    cudaGetDevice(&device);

    // Get the device properties
    cudaGetDeviceProperties(&props, device);

    // Get kernel attributes
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel);

    // The maximum number of blocks is determined by the device compute capability
    int maxBlocksPerSM;
    switch (props.major) 
    {
        case 1:  // Compute capability 1.x
            maxBlocksPerSM = 8;
            break;
        case 2:  // Compute capability 2.x
            maxBlocksPerSM = 8;
            break;
        case 3:  // Compute capability 3.x
            maxBlocksPerSM = 16;
            break;
        case 5:  // Compute capability 5.x
            maxBlocksPerSM = 32;
            break;
        case 6:  // Compute capability 6.x
            if (props.minor == 0) maxBlocksPerSM = 40;
            else maxBlocksPerSM = 32;
            break;
        case 7:  // Compute capability 7.x
            maxBlocksPerSM = 16;
            break;
        default:  // Compute capability 8.x or above
            maxBlocksPerSM = 16;
            break;
    }

    // Adjust for the maximum number of threads per multiprocessor
    int maxBlocksByThreads = props.maxThreadsPerMultiProcessor / threadsPerBlock;
    maxBlocksPerSM = min(maxBlocksPerSM, maxBlocksByThreads);

    // Adjust for the amount of shared memory used by the kernel
    if (attr.sharedSizeBytes > 0) {
        int maxBlocksBySharedMem = props.sharedMemPerMultiprocessor / attr.sharedSizeBytes;
        maxBlocksPerSM = min(maxBlocksPerSM, maxBlocksBySharedMem);
    }

    // Adjust for the number of registers used by the kernel
    if (attr.numRegs > 0) {
        int maxBlocksByRegs = props.regsPerMultiprocessor / (attr.numRegs * threadsPerBlock);
        maxBlocksPerSM = min(maxBlocksPerSM, maxBlocksByRegs);
    }

    // Calculate the total number of blocks that can be resident on the GPU simultaneously
    int totalBlocks = maxBlocksPerSM * props.multiProcessorCount;

    return totalBlocks;
}
