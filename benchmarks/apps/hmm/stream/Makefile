CXXFLAGS=-O3
CUDA_CXX=nvcc

cuda-stream: main.cu CUDAStream.cu
	$(CUDA_CXX) -arch=sm_86 -std=c++11 $(CXXFLAGS) -DCUDA $^ $(EXTRA_FLAGS) -Xptxas "-v" -o $@

.PHONY: clean
clean:
	rm -f cuda-stream

