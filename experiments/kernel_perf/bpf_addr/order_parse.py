import re
import sys

def parse_file(filename):
    starts = []  # List to store start values
    ends = []  # List to store end values
    pattern = re.compile(r'start: (\d+), end: (\d+)')
    
    with open(filename, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                start, end = map(int, match.groups())
                starts.append(start)
                ends.append(end)
    
    return starts, ends  # Returning separate lists

# Example usage:
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 parse.py <file_path>")
        sys.exit(1)

    filename = sys.argv[1] 
    starts, ends = parse_file(filename)

    count_pages = 0
    count_full = 0
    size_last = 0

    for start, end in zip(starts, ends):
        if start == 0: #and end ?
            count_full += 1
        else:
            size = (end - start) / 4096 # size of page
            if size_last == size:
                count_pages += 1
            else:
                print(str(count_pages) + " of " + str(size_last))
                count_pages = 1
                size_last = size
