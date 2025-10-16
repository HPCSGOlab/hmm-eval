#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

typedef struct {
    int nrows;
    int ncols;
    int nnz;
    int *row_ptr;
    int *col_idx;
    double *values;
} CSRMatrix;

void *allocate_aligned(size_t size) {
	void *ptr;
	int result = posix_memalign(&ptr, 2 * 1024 * 1024, size);
	if (result != 0) {
		fprintf(stderr, "posix_memalign failed %d\n", result);
		exit(1);
	}

	if (madvise(ptr, size, MADV_HUGEPAGE) != 0) {
		perror("madvise");
	}

	return ptr;
}

void read_mtx_to_csr(const char *filename, CSRMatrix *csr) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    char line[1024];
    do {
        if (!fgets(line, sizeof(line), f)) {
            fprintf(stderr, "Error: invalid MTX file.\n");
            exit(EXIT_FAILURE);
        }
    } while (line[0] == '%');

    int M, N, NNZ;
    if (sscanf(line, "%d %d %d", &M, &N, &NNZ) != 3) {
        fprintf(stderr, "Error reading matrix size.\n");
        exit(EXIT_FAILURE);
    }

    csr->nrows = M;
    csr->ncols = N;
    csr->nnz   = NNZ;
    
    const size_t hugepagesize = 2097152;
    size_t size = ((M + 1) * sizeof(int));
    int rem = size % hugepagesize;
    int numpages = size / hugepagesize;
    if (rem)
	    numpages++;

    csr->row_ptr = (int *) allocate_aligned(numpages * hugepagesize);
    memset(csr->row_ptr, 0, numpages * hugepagesize);

    size = NNZ * sizeof(int);
    rem = size % hugepagesize;
    numpages = size / hugepagesize;
    if (rem)
	    numpages++;

    csr->col_idx = (int *) allocate_aligned(numpages * hugepagesize);

    size = NNZ * sizeof(double);
    rem = size % hugepagesize;
    numpages = size / hugepagesize;
    if (rem)
	    numpages++;

    csr->values  = (double *) allocate_aligned(numpages * hugepagesize);

    if (!csr->row_ptr || !csr->col_idx || !csr->values) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    int *rows = (int *) malloc(NNZ * sizeof(int));
    int *cols = (int *) malloc(NNZ * sizeof(int));
    double *vals = (double *) malloc(NNZ * sizeof(double));
    if (!rows || !cols || !vals) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    // Read entries (1-based in MTX)
    for (int i = 0; i < NNZ; i++) {
        if (fscanf(f, "%d %d %lf", &rows[i], &cols[i], &vals[i]) != 3) {
            fprintf(stderr, "Error reading entry %d.\n", i);
            exit(EXIT_FAILURE);
        }
        rows[i]--;  // convert to 0-based
        cols[i]--;
        csr->row_ptr[rows[i] + 1]++;
    }
    fclose(f);

    for (int i = 0; i < M; i++) {
        csr->row_ptr[i + 1] += csr->row_ptr[i];
    }

    int *row_offset = (int *) calloc(M, sizeof(int));
    for (int i = 0; i < NNZ; i++) {
        int r = rows[i];
        int dest = csr->row_ptr[r] + row_offset[r];
        csr->col_idx[dest] = cols[i];
        csr->values[dest]  = vals[i];
        row_offset[r]++;
    }

    getchar();

    free(rows);
    free(cols);
    free(vals);
    free(row_offset);
}

void free_csr(CSRMatrix *csr) {
    free(csr->row_ptr);
    free(csr->col_idx);
    free(csr->values);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s matrix.mtx\n", argv[0]);
        return EXIT_FAILURE;
    }

    CSRMatrix A;
    read_mtx_to_csr(argv[1], &A);

    printf("Matrix loaded: %d x %d with %d nonzeros\n", A.nrows, A.ncols, A.nnz);

    for (int i = 0; i < A.nnz && i < 5; i++) {
        printf("val[%d] = %.7f at col %d at row_ptr %d\n", i, A.values[i], A.col_idx[i], A.row_ptr[i]);
    }

    free_csr(&A);
    return EXIT_SUCCESS;
}

