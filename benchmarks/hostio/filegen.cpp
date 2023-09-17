#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <sys/stat.h>


void generate_file(const char *filename, size_t array_size, size_t file_size_mb) {
    // Create a buffer to store the new filename
    char new_filename[100];
    size_t bytes_needed = snprintf(new_filename, sizeof(new_filename), "%s_%lu", filename, file_size_mb);

    // Check if the filename fits within the buffer
    if (bytes_needed >= sizeof(new_filename)) {
        fprintf(stderr, "Filename is too long. Aborting.\n");
        exit(1);
    }

    int fd = open(new_filename, O_RDWR | O_CREAT, 0666);
    if (fd == -1) {
        perror("File open error");
        exit(1);
    }
    int *data = new int[array_size];
    for (size_t i = 0; i < array_size; ++i) {
        data[i] = i;
    }
    // Initialize variables to keep track of the data.
    size_t total_bytes = array_size * sizeof(int);
    size_t bytes_written = 0;
    char *data_ptr = reinterpret_cast<char*>(data);

    while (bytes_written < total_bytes) {
        ssize_t written = write(fd, data_ptr + bytes_written, total_bytes - bytes_written);
        if (written == -1) {
            perror("File write error");
            exit(1);
        }
        bytes_written += written;
    }

    close(fd);
    delete[] data;
}


int main(int argc, char *argv[]) {
    int c;
    size_t file_size_mb = 0;

    while (1) {
        static struct option long_options[] = {
            {"size", required_argument, 0, 's'},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        c = getopt_long(argc, argv, "s:", long_options, &option_index);
        if (c == -1) break;
        switch (c) {
            case 's':
                file_size_mb = std::stoul(optarg);
                break;
            default:
                return 1;
        }
    }
    if (file_size_mb == 0)
    {
        fprintf(stderr, "Usage: %s -s <file size in mb>\n", argv[0]);
        exit(1);
    }

    size_t array_size = (file_size_mb * 1024 * 1024) / sizeof(int);

    generate_file("filename_read", array_size, file_size_mb);
    generate_file("filename_mmap", array_size, file_size_mb);

    return 0;
}
