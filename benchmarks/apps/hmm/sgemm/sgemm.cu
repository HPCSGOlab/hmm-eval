#include <iostream>
#include <chrono>
#include <cublas_v2.h>
#include <cblas.h>
#include <getopt.h>
#include <unistd.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>

using namespace std::chrono;

void cpu_multiply(float *A, float *B, float *C, size_t N, size_t iterations) {
    for (size_t i = 0; i < iterations; ++i) {
	    //ROW TO COL FOR CHECKING
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
    }
}

void gpu_multiply(float *A, float *B, float *C, size_t N, size_t iterations) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle;
    cudaEvent_t start, stop;
    float elapsedTime;
    
    cublasCreate(&handle);

    cudaDeviceSynchronize();
    
   
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (size_t i = 0; i < iterations; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    float gflops = iterations * (2.0f * N * N * N - N * N) / (elapsedTime / 1000.0f) / 1e9;
    printf("GPU,%zu,%f,%f\n", N, elapsedTime / 1000.0, gflops);

//    cublasGetMatrix(N, N, sizeof(float), d_C, N, C, N);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
}

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

int main(int argc, char **argv) {
    size_t N = 0;
    size_t iterations = 1;
    bool use_cpu = false;

    int opt;
    while ((opt = getopt(argc, argv, "n:ci:")) != -1) {
        switch (opt) {
            case 'n':
                N = std::stoull(optarg);
                break;
            case 'c':
                use_cpu = true;
                break;
            case 'i':
                iterations = std::stoull(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -n N [-c] [-i iterations]" << std::endl;
                return 1;
        }
    }

    if (!N) {
        std::cerr << "Usage: " << argv[0] << " -n N [-c] [-i iterations]" << std::endl;
        return 1;
    }

    size_t size = sizeof(float) * N * N;

    /*
    float *A = (float*) mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    float *B = (float*) mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    float *C = (float*) mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    madvise(A, size, MADV_HUGEPAGE);
    madvise(B, size, MADV_HUGEPAGE);
    madvise(C, size, MADV_HUGEPAGE);
    */

    size_t hugepagesize = 2097152;
    int rem = size % hugepagesize;
    int numpages = size / hugepagesize;

    if (rem)
	    numpages++;

    float *A = (float *)allocate_aligned(numpages * hugepagesize);
    float *B = (float *)allocate_aligned(numpages * hugepagesize);
    float *C = (float *)allocate_aligned(numpages * hugepagesize);

    for (size_t i = 0; i < N * N; ++i) {
        A[i] = i % 7;
        B[i] = (2 * i) % 7;
	C[i] = 0;
    }
    
    //getchar();

    if (use_cpu) {
        int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
        fprintf(stderr, "Detected %d cores.\n", num_cores);
        //openblas_set_num_threads(num_cores);

        //int num_threads = openblas_get_num_threads();

        //fprintf(stderr, "Number of threads OpenBLAS is using: %d\n", num_threads);


        high_resolution_clock::time_point start = high_resolution_clock::now();
        cpu_multiply(A, B, C, N, iterations);
        high_resolution_clock::time_point end = high_resolution_clock::now();
        duration<float> elapsed_time = duration_cast<duration<float>>(end - start);
        float gflops = iterations * (2.0f * N * N * N - N * N) / elapsed_time.count() / 1e9;
        printf("CPU,%zu,%f,%f\n", N, elapsed_time.count(), gflops);
    } else {
        gpu_multiply(A, B, C, N, iterations);

	/*
	float *CPU_C = new float[N * N];
	cpu_multiply(A, B, CPU_C, N, iterations);

#pragma omp parallel
	for (size_t i = 0; i < N * N; i++) {
		if (C[i] - CPU_C[i] > 0.01f) {
			printf("C: %f != CPU_C: %f\n", C[i], CPU_C[i]);
		}
	}
	*/
    }

    /*
    free(A);
    free(B);
    free(C);
    */

    return 0;
}

