import re
import sys

def parse_file(filename):
    times = []  # List to store start values
    pattern = re.compile(r'c,(\d+),0')
    
    with open(filename, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                times.append(int(match.group(1)))
    
    return times # Returning separate lists

def summarize_times(times):
    if not times:
        return None
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    return avg_time, min_time, max_time

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 timings.py <file_path>")
        sys.exit(1)

    filename = sys.argv[1]
    times = parse_file(filename)

    if not times:
        print("No valid timing entries found in file.")
        sys.exit(1)

    avg_time, min_time, max_time = summarize_times(times)
    sum_time = sum(times)

    print(f"Number of samples: {len(times)}")
    print(f"Average time: {avg_time:.2f}")
    print(f"Minimum time: {min_time}")
    print(f"Maximum time: {max_time}")
    print(f"Total time: {sum_time}")
