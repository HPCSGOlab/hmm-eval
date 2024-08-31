import struct
import sys

def read_all_integers_from_binary_file(filename):
    with open(filename, 'rb') as f:
        data = f.read()  # Read the entire file into a bytes object
        num_integers = len(data) // 4  # Calculate the number of 4-byte integers

        # Unpack all integers at once
        integers = struct.unpack('<' + 'I'*num_integers, data)
        
    return integers

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <binary_file> <target_value>")
        sys.exit(1)
    
    binary_file = sys.argv[1]
    target_value = int(sys.argv[2])
    
    integers = read_all_integers_from_binary_file(binary_file)
    
    incorrect_count = 0
    first_incorrect_index = None

    for index, integer in enumerate(integers):
        if integer != target_value:
            incorrect_count += 1
            if first_incorrect_index is None:
                first_incorrect_index = index

    if incorrect_count == 0:
        print("Correct: All integers in the file match the target value.")
    else:
        print(f"Incorrect: {incorrect_count} values do not match the target value.")
        print(f"First incorrect value found at index {first_incorrect_index}")

if __name__ == "__main__":
    main()

