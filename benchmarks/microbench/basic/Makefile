all: basic

basic: basic.cu Makefile
	nvcc -arch=sm_86 -Xcompiler "-Wall -Wextra" -Xptxas -v basic.cu -o basic
