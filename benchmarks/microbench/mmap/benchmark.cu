#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <getopt.h>
#include <string>
#include <cuda_runtime.h>
#include <chrono>

#define TPB 256
template <typename T>
int calculateOptimalBlocks(T kernel, int threadsPerBlock);

void create_random_indices(std::vector<size_t>& indices, size_t array_size) {
    for (size_t i = 0; i < array_size; ++i) {
        indices[i] = i;
    }
    std::random_shuffle(indices.begin(), indices.end());
}

__global__  void warmup()
{
    return;
}

__global__ void inc(int* a, size_t* indices, size_t n)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    for (; i < n; i += blockDim.x * gridDim.x)
    {
        a[indices[i]] += 1;
    }
}

// 0 cpu 1 gpu
double benchmark_mmap(const char *filename, size_t array_size, std::vector<size_t>& indices, int cpugpu) {
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    int fd = open(filename, O_RDWR);
    if (fd == -1) {
        perror("File open error");
        exit(1);
    }
    int *map = (int*)mmap(NULL, array_size * sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("mmap error");
        exit(1);
    }
    if (!cpugpu)
    {
        start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < array_size; ++i) {
            map[indices[i]]++;
        }
        end = std::chrono::steady_clock::now();
    }
    else
    {
        const int blocks = calculateOptimalBlocks(inc, TPB);
        start = std::chrono::steady_clock::now();
        inc<<<blocks, TPB>>>(map, &indices[0], array_size);
        cudaDeviceSynchronize();
        end = std::chrono::steady_clock::now();
    }
    munmap(map, array_size * sizeof(int));
    close(fd);
    return std::chrono::duration<double>(end - start).count();
}

int main(int argc, char *argv[]) {
    int c;
    std::string access_mode = "linear";
    size_t file_size_mb = 0;

    while (1) {
        static struct option long_options[] = {
            {"size", required_argument, 0, 's'},
            {"access", required_argument, 0, 'a'},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        c = getopt_long(argc, argv, "s:a:", long_options, &option_index);
        if (c == -1) break;
        switch (c) {
            case 's':
                file_size_mb = std::stoul(optarg);
                break;
            case 'a':
                access_mode = optarg;
                break;
            default:
                return 1;
        }
    }

    size_t array_size = (file_size_mb * 1024 * 1024) / sizeof(int);
    std::vector<size_t> indices(array_size);
    if (access_mode == "random") {
        create_random_indices(indices, array_size);
    } else {
        for (size_t i = 0; i < array_size; ++i) {
            indices[i] = i;
        }
    }

    // Generate filenames with the file size in MB
    std::string cpu_mmap = "inputs/cpu_mmap_" + std::to_string(file_size_mb);
    std::string gpu_mmap = "inputs/gpu_mmap_" + std::to_string(file_size_mb);

    warmup<<<1,1>>>();
    cudaDeviceSynchronize();

    double cpu_time = benchmark_mmap(cpu_mmap.c_str(), array_size, indices, 0);
    double gpu_time = benchmark_mmap(gpu_mmap.c_str(), array_size, indices, 1);

    printf("%zu,%s,%f,%f\n", file_size_mb, access_mode.c_str(), cpu_time, gpu_time);

    return 0;
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

