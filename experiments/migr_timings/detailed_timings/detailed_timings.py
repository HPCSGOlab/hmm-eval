import re
import sys
from collections import defaultdict

def parse_file(filename):
    # Dictionary: letter -> list of times
    times = defaultdict(list)

    # Match lines like: s,408053,0
    pattern = re.compile(r'([a-z]),(\d+),0')

    with open(filename, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                letter = match.group(1)
                value = int(match.group(2))
                times[letter].append(value)

    return times

def summarize_avg(times_by_letter):
    avgs = {}
    for letter, values in times_by_letter.items():
        if values:
            avgs[letter] = sum(values) / len(values)
            print(len(values))
    return avgs

def summarize_letters(times_by_letter):
    avgs = {}
    for letter, values in times_by_letter.items():
        if values:
            avgs[letter] = sum(values)
    return avgs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 timings_by_letter.py <file_path>")
        sys.exit(1)

    filename = sys.argv[1]
    times_by_letter = parse_file(filename)

    if not times_by_letter:
        print("No valid timing entries found in file.")
        sys.exit(1)

    avgs = summarize_letters(times_by_letter)

    print("Total time per tag:")
    for letter in sorted(avgs):
        print(f"  {letter}: {avgs[letter]:.2f}")

