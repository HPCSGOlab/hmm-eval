CC = gcc
CFLAGS = -Wall -Wextra -g -O3

NVCC = nvcc
NVFLAGS = -arch=sm_86 -Xptxas -v -Xcompiler "$(CFLAGS)"  # You can adjust this to your GPU architecture

TARGETS = rangen dio bcat diocpu

all: $(TARGETS)

rangen: rangen.c
	$(CC) $(CFLAGS) rangen.c -o rangen

dio: dio.cu
	$(NVCC) $(NVFLAGS) dio.cu -o dio

diocpu: dio.c
	$(CC) $(CFLAGS) dio.c -o diocpu

bcat: bcat.c
	$(CC) $(CFLAGS) bcat.c -o bcat

clean:
	rm -f $(TARGETS)

