import matplotlib.pyplot as plt
import csv

def read_csv(filename):
    sizes, times, gflops = [], [], []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            _, size, time, gflop = row
            sizes.append(int(size))
            times.append(float(time))
            gflops.append(float(gflop))
    return sizes, times, gflops

def plot_time(cpu_sizes, cpu_times, gpu_sizes, gpu_times):
    plt.figure(figsize=(10, 6))
    plt.plot(cpu_sizes, cpu_times, marker='o', label='CBLAS (CPU)')
    plt.plot(gpu_sizes, gpu_times, marker='x', label='CUBLAS (GPU)')
    plt.xlabel('Problem Size (N)')
    plt.ylabel('Time (s)')
    plt.title('Time Comparison: CBLAS vs CUBLAS')
    plt.legend()
    plt.grid(True)
    plt.savefig('time_comparison.pdf')
    plt.savefig('time_comparison.png')

def plot_gflops(cpu_sizes, cpu_gflops, gpu_sizes, gpu_gflops):
    plt.figure(figsize=(10, 6))
    plt.plot(cpu_sizes, cpu_gflops, marker='o', label='CBLAS (CPU)')
    plt.plot(gpu_sizes, gpu_gflops, marker='x', label='CUBLAS (GPU)')
    plt.xlabel('Problem Size (N)')
    plt.ylabel('GFLOPs')
    plt.title('Performance Comparison: CBLAS vs CUBLAS')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_comparison.pdf')
    plt.savefig('performance_comparison.png')

def main():
    cpu_sizes, cpu_times, cpu_gflops = read_csv("cblas_results.csv")
    gpu_sizes, gpu_times, gpu_gflops = read_csv("cublas_results.csv")
    
    plot_time(cpu_sizes, cpu_times, gpu_sizes, gpu_times)
    plot_gflops(cpu_sizes, cpu_gflops, gpu_sizes, gpu_gflops)

if __name__ == "__main__":
    main()

