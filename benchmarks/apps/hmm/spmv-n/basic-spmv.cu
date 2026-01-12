#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

typedef struct {
    int nrows;
    int ncols;
    int nnz;
    int *row_ptr;
    int *col_idx;
    float *values;
} CSRMatrix;

__global__ void spmv_csr_kernel(int num_rows, const int *rowPtr, const int *colIdx, const float* val,
				const float *x, float *y) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num_rows) {
		float dot = 0.0f;
		for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
			dot += val[j] * x[colIdx[j]];
		}

		y[i] = dot;
	}
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
    
    size_t size = ((M + 1) * sizeof(int));

    csr->row_ptr = (int *) (size);
    memset(csr->row_ptr, 0, size);

    size = NNZ * sizeof(int);

    csr->col_idx = (int *) malloc(size);

    size = NNZ * sizeof(float);

    csr->values  = (float *) malloc(size);

    if (!csr->row_ptr || !csr->col_idx || !csr->values) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    int *rows = (int *) malloc(NNZ * sizeof(int));
    int *cols = (int *) malloc(NNZ * sizeof(int));
    float *vals = (float *) malloc(NNZ * sizeof(float));
    if (!rows || !cols || !vals) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    // Read entries (1-based in MTX)
    for (int i = 0; i < NNZ; i++) {
        if (fscanf(f, "%d %d %f", &rows[i], &cols[i], &vals[i]) != 3) {
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

    //getchar();

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

    size_t size = A.nrows * sizeof(float);

    float* x = (float *) malloc(size);
    float* y = (float *) malloc(size);

    for (int i = 0; i < A.nrows; i++) {
	x[i] = i % 7;
	y[i] = 0; 
    }

    int t = 256;
    int b = (A.nrows + t - 1) / t;

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaDeviceSynchronize();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < 1; i++)
	    spmv_csr_kernel<<<b, t>>>(A.nrows, A.row_ptr, A.col_idx, A.values, x, y);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
	printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    float data_size = (A.nrows * sizeof(float)) + (A.nrows * sizeof(float))
		    + ((A.nrows + 1) * sizeof(int)) + (A.nnz * sizeof(int)) + (A.nnz * sizeof(float))/* sizeof(y) + sizeof(x) + sizeof(A) */;
    float bandwidth = data_size / elapsedTime;
    printf("GPU,%d,%f,%f\n", A.nrows, elapsedTime / 1000.0, bandwidth);
    
    return 0;
}

