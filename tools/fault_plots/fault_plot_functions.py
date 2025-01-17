import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

from functools import wraps
from matplotlib.ticker import FixedLocator

matplotlib.use('Agg')
#matplotlib.use('pdf')
mplstyle.use('fast')


def plotter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        plt.figure(figsize=(10, 6))
        result = func(*args, **kwargs)
        plt.clf()
        return result
    return wrapper

@plotter
def plot_faults_relative_order(df_faults, df_prefetch, df_evictions, df_address_ranges, output_state, plot_end):
    
    print("configuring plot")
    fontsize=16
    legend_marker_size=50
    output_path = output_state.get_output_path("faults_relative")
    plt.scatter([], [], color='green', alpha=1, label='Faults', marker='o', s=legend_marker_size, linewidths=0)
    plt.scatter(df_faults.index, df_faults['adjusted_faddr'], color='green', alpha=0.5, label=None, marker='o', s=1, linewidths=0)

    if not df_prefetch.empty:
        plt.scatter([],[], color='blue', alpha=1, label='Prefetches', marker='o', s=legend_marker_size, linewidths=0)
        plt.scatter(df_prefetch.index, df_prefetch['adjusted_faddr'], color='blue', alpha=0.5, label=None, marker='o', s=1, linewidths=0)
    if not df_evictions.empty:
        plt.scatter([],[], color='purple', alpha=1, label='Evictions', marker='o', s=legend_marker_size, linewidths=0)
        plt.scatter(df_evictions.index, df_evictions['adjusted_faddr'], color='purple', alpha=0.5, label=None, marker='o', s=1, linewidths=0)
 

    xmin = 0
    xmax = max(df_faults.index.max(), len(df_faults) + len(df_address_ranges))

    df_address_ranges = df_address_ranges.sort_values('adjusted_base', ascending=False).reset_index()
    for idx, row in df_address_ranges.iterrows():
        print(f"plotting allocation relative range starting at {hex(int(row['adjusted_base']))}")
        plt.hlines(y=row['adjusted_base'], xmin=xmin, xmax=xmax, color='black', label='Address Range Start' if idx == 0 else "")
        if plot_end:
            plt.hlines(y=row['adjusted_end'], xmin=xmin, xmax=xmax, color='red', label='Address Range End' if idx == 0 else "")
        if idx < len(df_address_ranges) - 1:
            midpoint = (row['adjusted_base'] + df_address_ranges['adjusted_base'][idx + 1]) / 2
            plt.text(xmin, midpoint, chr(65 + idx), ha='center', va='center', fontsize=24, color='red')
        

    y_ticks = plt.gca().get_yticks()
    #plt.gca().set_yticklabels([hex(int(y)) for y in y_ticks])
    plt.gca().yaxis.set_major_locator(FixedLocator(y_ticks))
    plt.tick_params(axis='both', labelsize=fontsize)

    plt.xlabel('Event Order', fontsize=fontsize)
    plt.ylabel('Address (Adjusted)', fontsize=fontsize)
    #plt.legend()
    # Shrink current axis's height by 10% on the bottom
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=5, fontsize=fontsize)


    #plt.tight_layout()
    print("plotting and saving")
    plt.savefig(output_path, format='png')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    print("This is a utility class; you are probably looking for fault_plot.py")
