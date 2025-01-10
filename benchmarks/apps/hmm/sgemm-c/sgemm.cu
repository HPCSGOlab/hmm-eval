#include <iostream>
#include <chrono>
#include <cublas_v2.h>
#include <cblas.h>
#include <getopt.h>
#include <unistd.h>

using namespace std::chrono;

void cpu_multiply(float *A, float *B, float *C, size_t N, size_t iterations) {
    for (size_t i = 0; i < iterations; ++i) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
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

    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];

    unsigned int seed = 243;
#pragma omp parallel for private(seed)
    for (size_t i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(rand_r(&seed)) / RAND_MAX;
        B[i] = static_cast<float>(rand_r(&seed)) / RAND_MAX;
	C[i] = static_cast<float>(0);
    }

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
    }

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

