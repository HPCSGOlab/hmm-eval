CC = g++
CFLAGS = -Wall -Wextra -O3 -g
TARGET = benchmark

all: $(TARGET) filegen

$(TARGET): benchmark.cu
	nvcc -arch=sm_90 -Xptxas -v -Xcompiler "$(CFLAGS)" benchmark.cu -o $(TARGET)

filegen: filegen.cpp
	$(CC) $(CFLAGS) filegen.cpp -o filegen

clean:
	rm -f $(TARGET)

