#!/usr/bin/python
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot performance benchmark data.')
    parser.add_argument('dir', type=str, help='Directory path of performance data.')
    return parser.parse_args()

def plot_data(file_path, prefetching_status):
    print(f"Parsing file: {file_path}")
    # Read the CSV file
    data = pd.read_csv(file_path)
    print(f"File entries: \n{data}")

    # Set plot size
    plt.figure(figsize=(10, 6))

    # Plot each allocator mode
    for mode in allocator_modes.keys():
        subset = data[data['Allocator Mode'] == mode]
        plt.plot(subset['Problem Size'], subset['Runtime'], marker='o', label=allocator_modes[mode])

    # Set plot title, labels, and legend
    plt.title(f'{benchmark_name} Benchmark Performance {prefetching_status}')
    plt.xlabel('Problem Size')
    plt.ylabel('Runtime (s)')
    plt.legend()

    # Ensure the output directory exists
    os.makedirs(f'../figures/alloc-perf', exist_ok=True)

    # Save the figure
    plt.savefig(f'../figures/alloc-perf/{benchmark_name}_{prefetching_status.replace(" ", "_")}.png')

def main():
    args = parse_arguments()

    # Set your directory path here
    directory_path = args.dir # e.g., "../benchmarks/stream/perf"

    # Parse the benchmark name from the directory path
    global benchmark_name
    benchmark_name = directory_path.split('/')[-2]

    # Define allocator mode mapping
    global allocator_modes
    allocator_modes = {0: "malloc", 1: "cudaMallocManaged", 2: "malloc + RDMA", 3: "cudaMallocManaged + RDMA", 4: "malloc + RDMA + premap", 5: "cudaMallocManaged + RDMA + premap"}

    # For each file in the directory, if it's a CSV file, plot the data
    for file in os.listdir(directory_path):
        if file.endswith('.csv'):
            prefetching_status = "with_Prefetch" if "noprefetch" not in file else "without_Prefetch"
            plot_data(os.path.join(directory_path, file), prefetching_status)

if __name__ == "__main__":
    main()

