#!/usr/bin/env python3
import re
import sys

def parse_uvm_groups(filename):
    prefix_re = re.compile(r"^\[\s*[\d.]+\]\s*")

    groups = []
    current = {
        "c_times": [],
        "faults": [],
        "w_time": None
    }

    last_c_time = None

    with open(filename, "r") as f:
        for line in f:
            line = prefix_re.sub("", line).strip()
            if not line:
                continue

            # c,
            if line.startswith("c,"):
                _, ts_str, flag = line.split(",")
                last_c_time = int(ts_str) / 1e9
                current["c_times"].append(last_c_time)

            # fault,
            elif line.startswith("fault,"):
                _, addr, size, cpu = line.split(",")
                current["faults"].append({
                    "address": int(addr),
                    "size": int(size),
                    "cpu": int(cpu),
                    "c_time": last_c_time
                })
                last_c_time = None

            # w,
            elif line.startswith("w,"):
                _, ts_str, flag = line.split(",")
                current["w_time"] = int(ts_str) / 1e9

                # close the group
                groups.append(current)

                # start next group
                current = {
                    "c_times": [],
                    "faults": [],
                    "w_time": None
                }

    return groups


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 uvm_parse.py <input_file.txt>")
        sys.exit(1)

    filename = sys.argv[1]
    groups = parse_uvm_groups(filename)

    # pretty print output
    from pprint import pprint
    pprint(groups)


if __name__ == "__main__":
    main()

