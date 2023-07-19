all: stream

stream: stream.cu Makefile
	nvcc -arch=sm_86 -Xcompiler "-Wall -Wextra" -Xptxas -v stream.cu -o stream
