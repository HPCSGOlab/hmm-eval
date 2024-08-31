import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Plot File Access Time vs File Size.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file.')
    return parser.parse_args()

def read_and_transform_csv(file_path):
    """Read the CSV file and transform it into a suitable format."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    df = pd.read_csv(file_path)
    df['File Size (GB)'] = df['File Size (MB)'] / 1024.0
    return df

def plot_data(ax, df):
    """Plot the data on a given axis."""
    for access_pattern in df['Access Pattern'].unique():
        for read_method in df['Read Time (s)'].unique():
            subset = df[(df['Access Pattern'] == access_pattern) & (df['Read Time (s)'] == read_method)]
            subset = subset.sort_values('File Size (GB)')
            ax.plot(subset['File Size (GB)'], subset['MMap Time (s)'], label=f'{access_pattern} {read_method}')

def set_plot_properties(ax):
    """Set labels, title, scale and legend for the plot."""
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('File Size (GB)')
    ax.set_ylabel('Time (s)')
    ax.set_title('File Access Time vs File Size')
    ax.legend()

def save_plot():
    """Save the plot as PDF and PNG."""
    plt.savefig('file_access_time_vs_size.pdf')
    plt.savefig('file_access_time_vs_size.png')

def main():
    """Main function to read data, plot and save the figure."""
    args = parse_arguments()
    file_path = args.file_path
    df = read_and_transform_csv(file_path)
    
    fig, ax = plt.subplots()
    plot_data(ax, df)
    set_plot_properties(ax)
    save_plot()

if __name__ == "__main__":
    main()

