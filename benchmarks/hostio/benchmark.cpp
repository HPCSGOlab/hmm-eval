#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include <getopt.h>
#include <string>


void create_random_indices(std::vector<size_t>& indices, size_t array_size) {
    for (size_t i = 0; i < array_size; ++i) {
        indices[i] = i;
    }
    std::random_shuffle(indices.begin(), indices.end());
}

double benchmark_read(const char *filename, size_t array_size, std::vector<size_t>& indices) {
    int fd = open(filename, O_RDWR);
    if (fd == -1) {
        perror("File open error");
        exit(1);
    }

    int *buffer = (int*)malloc(array_size * sizeof(int));
    if (buffer == nullptr) {
        perror("Memory allocation error");
        exit(1);
    }

    size_t total_bytes = array_size * sizeof(int);
    size_t bytes_read = 0;
    char *buffer_ptr = reinterpret_cast<char*>(buffer);

    clock_t start = clock();

    // Read entire file into buffer
    while (bytes_read < total_bytes) {
        ssize_t result = read(fd, buffer_ptr + bytes_read, total_bytes - bytes_read);
        if (result == -1) {
            perror("File read error");
            exit(1);
        } else if (result == 0) {
            // EOF reached before expected
            break;
        }
        bytes_read += result;
    }

    // Modify the buffer (increment each element)
    for (size_t i = 0; i < array_size; ++i) {
        buffer[i]++;
    }

    // Write the modified buffer back to the file
    lseek(fd, 0, SEEK_SET); // Reset file pointer to the beginning of the file
    size_t bytes_written = 0;
    while (bytes_written < total_bytes) {
        ssize_t written = write(fd, buffer_ptr + bytes_written, total_bytes - bytes_written);
        if (written == -1) {
            perror("File write error");
            exit(1);
        }
        bytes_written += written;
    }

    clock_t end = clock();
    close(fd);
    free(buffer);
    return static_cast<double>(end - start) / CLOCKS_PER_SEC;
}



double benchmark_mmap(const char *filename, size_t array_size, std::vector<size_t>& indices) {
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
    clock_t start = clock();
    for (size_t i = 0; i < array_size; ++i) {
        map[indices[i]]++;
    }
    clock_t end = clock();
    munmap(map, array_size * sizeof(int));
    close(fd);
    return (double)(end - start) / CLOCKS_PER_SEC;
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
    std::string filename_read = "inputs/filename_read_" + std::to_string(file_size_mb);
    std::string filename_mmap = "inputs/filename_mmap_" + std::to_string(file_size_mb);

    double read_time = benchmark_read(filename_read.c_str(), array_size, indices);
    double mmap_time = benchmark_mmap(filename_mmap.c_str(), array_size, indices);

    printf("%zu,%s,Read,%f\n", file_size_mb, access_mode.c_str(), read_time);
    printf("%zu,%s,Mmap,%f\n", file_size_mb, access_mode.c_str(), mmap_time);

    return 0;
}

