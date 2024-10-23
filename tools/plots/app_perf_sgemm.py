import argparse
import csv

import matplotlib.pyplot as plt

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

def plot_time(cpu_sizes, cpu_times, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(cpu_sizes, cpu_times, marker='o', label='CBLAS (CPU)')
    plt.xlabel('Problem Size (N)')
    plt.ylabel('Time (s)')
    plt.title('Time: CBLAS')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_path}/time.pdf')
    plt.savefig(f'{output_path}/time.png')

def plot_gflops(cpu_sizes, cpu_gflops, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(cpu_sizes, cpu_gflops, marker='o', label='CBLAS (CPU)')
    plt.xlabel('Problem Size (N)')
    plt.ylabel('GFLOPs')
    plt.title('Performance: CBLAS')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_path}/performance.pdf')
    plt.savefig(f'{output_path}/performance.png')


def main():
    # Set up argparse to handle command line arguments
    parser = argparse.ArgumentParser(description="Process a CSV file.")
    parser.add_argument("filepath", help="File path of the CSV file to process", type=str)
    parser.add_argument("-o", "--output_path", help="Output directory to save the results", type=str, required=True)
    args = parser.parse_args()

    # Use the filepath provided as a command line argument
    cpu_sizes, cpu_times, cpu_gflops = read_csv(args.filepath)
    plot_time(cpu_sizes, cpu_times, args.output_path)
    plot_gflops(cpu_sizes, cpu_gflops, args.output_path)

if __name__ == "__main__":
    main()
