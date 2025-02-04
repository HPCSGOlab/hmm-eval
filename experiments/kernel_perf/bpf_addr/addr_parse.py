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

    count_full = 0
    addr_dict = {}

    for start, end in zip(starts, ends):
        if start == 0: #and end ?
            count_full += 1
        else:
            for i in range(start, end, 4096):
                page_address = i // 4096 # size of page
                if page_address in addr_dict.keys():
                    addr_dict[page_address] += 1
                else:
                    addr_dict[page_address] = 1
           
    count = 0
    sizes_info = {}

    for key in addr_dict.keys():
        if addr_dict[key] > 1:
            count+=1
            curr_size = addr_dict[key]
            if curr_size in sizes_info.keys():
                sizes_info[curr_size] += 1
            else:
                sizes_info[curr_size] = 1

    print(sizes_info)
    print(count)
