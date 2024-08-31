#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        return EXIT_FAILURE;
    }

    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        perror("Error opening file");
        return EXIT_FAILURE;
    }

    int num;
    while (fread(&num, sizeof(int), 1, file) == 1) {
        printf("%d\n", num);
    }

    fclose(file);
    return EXIT_SUCCESS;
}

