#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

__global__ void increment(int* foo, int size)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < size)
    {
	foo[gid] += 1;
    }	
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Open the file
    int fd = open(argv[1], O_RDWR);
    if (fd == -1) {
        perror("Error opening file");
        return EXIT_FAILURE;
    }

    // Get the file's size
    struct stat fileInfo;
    if (fstat(fd, &fileInfo) == -1) {
        perror("Error getting the file size");
        close(fd);
        return EXIT_FAILURE;
    }

    if (fileInfo.st_size % sizeof(int) != 0) {
        fprintf(stderr, "File size isn't a multiple of int size\n");
        close(fd);
        return EXIT_FAILURE;
    }

    // Memory map the file
    int *data = (int*) mmap(NULL, fileInfo.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
        perror("Error mapping file");
        close(fd);
        return EXIT_FAILURE;
    }

    // Process the integers
    int numInts = fileInfo.st_size / sizeof(int);
    
    increment<<<numInts / 1024 + 1, 1024>>>(data, numInts);
    cudaDeviceSynchronize();
    /*
    for (int i = 0; i < numInts; i++) {
        data[i]++;  // Increment each integer
    }*/

    // Unmap and close
    if (munmap(data, fileInfo.st_size) == -1) {
        perror("Error unmapping the file");
    }

    close(fd);
    return EXIT_SUCCESS;
}

