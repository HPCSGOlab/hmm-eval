import pandas as pd
import matplotlib.pyplot as plt
import sys

def parse_data(file_path):
    df = pd.read_csv(file_path)
    return df

def generate_plot(df):
    plt.figure(figsize=(10, 6))

    markers = ['o', 's', 'v', '^']
    linestyles = ['-', '--', '-.', ':']

    index = 0
    for access_pattern in ['linear', 'random']:
        df_filtered = df[df['Access Pattern'] == access_pattern]
        problem_sizes_gb = df_filtered['File Size (MB)'] / 1024

        plt.plot(problem_sizes_gb, df_filtered['CPU Time (s)'], marker=markers[index], linestyle=linestyles[index], label=f'CPU-{access_pattern.capitalize()}')
        plt.plot(problem_sizes_gb, df_filtered['GPU Time (s)'], marker=markers[index+1], linestyle=linestyles[index+1], label=f'GPU-{access_pattern.capitalize()}')
    
        index += 2


    plt.xlabel('Problem Size (GB)', fontsize=22)
    plt.ylabel('Time (s)', fontsize=22)
    plt.title('Map-Increment Task: GPU vs. CPU', fontsize=24)
    plt.legend(loc='upper left', fontsize=22)
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.savefig('performance_comparison.png', format='png', dpi=300)
    plt.savefig('performance_comparison.pdf', format='pdf')

def main():
    if len(sys.argv) != 2:
        print("Error: Missing input file. Usage: python script.py <input_file.csv>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    df = parse_data(file_path)
    generate_plot(df)

if __name__ == '__main__':
    main()

