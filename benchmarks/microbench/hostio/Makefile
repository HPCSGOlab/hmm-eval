CC = g++
CFLAGS = -Wall -Wextra -O3 -g -lrt
TARGET = benchmark

all: $(TARGET) filegen

$(TARGET): benchmark.cpp
	$(CC) $(CFLAGS) benchmark.cpp -o $(TARGET)

filegen: filegen.cpp
	$(CC) $(CFLAGS) filegen.cpp -o filegen

clean:
	rm -f $(TARGET)

