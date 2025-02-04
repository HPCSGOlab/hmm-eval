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
    size_dict = {}

    for start, end in zip(starts, ends):
        if start == 0: #and end ?
            count_full += 1
        else:
            size = (end - start) / 4096 # size of page
            count_pages += size
            if size in size_dict.keys():
                size_dict[size] += 1
            else:
                size_dict[size] = 1
           

    #print("Pages flushed w/ native_flush_tlb_multi: " + str(count_pages))
    print("Pages migrated w/ migrate_vma_setup: " + str(count_pages))
    #print("Pages flushed from 0 to MAX: " + str(count_full))
    print()
    #print("native_flush_tlb_multi calls by size: ")
    print("migrate_vma_setup calls by size: ")

    size_dict = dict(sorted(size_dict.items(), key=lambda item:item, reverse=True))

    for key in size_dict.keys():
        #print("A flush of size of " + str(key) + " pages was called this many times: " + str(size_dict[key]))
        print("A migration size of " + str(key) + " pages was called this many times: " + str(size_dict[key]))
