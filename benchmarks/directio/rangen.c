#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <filename> <number_of_integers>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Convert second argument to integer count
    int count = atoi(argv[2]);
    if (count <= 0) {
        fprintf(stderr, "Please specify a positive number of integers.\n");
        return EXIT_FAILURE;
    }

    // Seed random number generator
    srand((unsigned) time(NULL));

    // Open file for binary write
    FILE *file = fopen(argv[1], "wb");
    if (!file) {
        perror("Error opening file for writing");
        return EXIT_FAILURE;
    }

    // Generate and write random integers to file
    for (int i = 0; i < count; i++) {
        int random_int = rand();
        fwrite(&random_int, sizeof(int), 1, file);
    }

    fclose(file);
    printf("Generated %d random integers to %s\n", count, argv[1]);

    return EXIT_SUCCESS;
}

