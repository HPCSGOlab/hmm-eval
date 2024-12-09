#!/usr/bin/python
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot performance benchmark data.')
    parser.add_argument('dir', type=str, help='Directory path of performance data.')
    parser.add_argument('benchmark', type=str, help='Benchmark we are interested in.')
    return parser.parse_args()

def plot_data(file_paths, machine):
    data_for_all = []
    for file_path in file_paths:
        print(f"Parsing file: {file_path}")
        # Read the CSV file
        data = pd.read_csv(file_path)
        #print(f"File entries: \n{data}")
        data_for_all.append(data)

    # Set plot size
    plt.figure(figsize=(10, 6))
    markers = ['o', 'x', 'v']

    # Plot each memory management mode
    for item in data_for_all[0].columns[2:]:
        for idx, mode in enumerate(data_for_all):
            plt.plot(mode['Size'], mode[item], marker=markers[idx], label=file_paths[idx].split('/')[-3])

        # Set plot title, labels, and legend
        plt.title(f'{benchmark_name} Benchmark Performance')
        plt.xlabel('Problem Size')
        plt.ylabel(item)
        plt.ylim(0)
        plt.legend()


        # Ensure the output directory exists
        os.makedirs(f'../../figs/alloc-perf/{machine}', exist_ok=True)

        # Save the figure
        plt.savefig(f'../../figs/alloc-perf/{machine}/{benchmark_name}_{item}.png')

def main():
    args = parse_arguments()

    # Set your directory path here
    directory_path = args.dir # e.g., "../data/cci-hopper/app_perf/"

    # THIS IS LIKE VERY SPECIFIC TO THE PROJECT SO IF THERE ARE PROBS IT MIGHT BE THIS TODO
    machine = directory_path.split("/")[-3]
    print("IF THIS IS NOT YOUR MACHINE IM SORRY PLEASE FIX: " + machine)

    global benchmark_name
    benchmark_name  = args.benchmark # e.g., "stream'
    
    print(os.listdir(directory_path))
    csv_list = []
    for management_type in os.listdir(directory_path):
        csv_path = directory_path + "/" + management_type + "/" + benchmark_name
        file = os.listdir(csv_path)[0] #TODO assuming one csv in dir
        if file.endswith('.csv'):
            #prefetching_status = "with_Prefetch" if "noprefetch" not in file else "without_Prefetch"
            csv_list.append(os.path.join(csv_path, file))

    print(csv_list)
    plot_data(csv_list, machine)


if __name__ == "__main__":
    main()
