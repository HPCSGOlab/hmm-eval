import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Directory containing timing files
DATA_DIR = "."   # change if needed

# Regex patterns
TAG_LINE_RE = re.compile(r"^\s*([a-z]):\s*([\d.]+)")
HEADER_RE = re.compile(r"Total time per tag:")

def parse_file(path):
    """
    Parse a timing file and return the adjusted total time.
    If 'w' exists, subtract it from the total.
    """
    tags = {}
    in_block = False

    with open(path, "r", errors="ignore") as f:
        for line in f:
            if HEADER_RE.search(line):
                in_block = True
                continue

            if in_block:
                match = TAG_LINE_RE.match(line)
                if match:
                    tag, val = match.groups()
                    tags[tag] = tags.get(tag, 0.0) + float(val)
                elif line.strip() == "":
                    # end of block
                    in_block = False

    total = sum(tags.values())
    w_time = tags.get("w", 0.0)

    adjusted_total = total - w_time
    return adjusted_total, tags


def main():
    files = [
    f for f in os.listdir(DATA_DIR)
    if (
        "sgemm" in f
        and f.endswith(".txt")
        and os.path.isfile(os.path.join(DATA_DIR, f))
    )
    ]

    prefix_order = ["uvm", "folio", "hmm"]

    def order_key(filename):
        prefix = filename.split("_")[0].lower()
        if prefix in prefix_order:
            return prefix_order.index(prefix)
        return len(prefix_order)  # unknowns go last

    files.sort(key=order_key)

    totals = []
    labels = []

    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        total, tags = parse_file(path)
        totals.append(total)
        labels.append(fname)

        print(f"{fname}")
        print(f"  Raw total: {sum(tags.values()):.2f}")
        if "w" in tags:
            print(f"  Subtracted w: {tags['w']:.2f}")
        print(f"  Adjusted total: {total:.2f}\n")

    # ---- Plot clustered bar chart ----
    cluster_x = 0
    bar_width = 0.2

    # Extract prefix (first word before '_')
    prefixes = [label.split("_")[0] for label in labels]

    # Assign a color per prefix
    unique_prefixes = sorted(set(prefixes))
    color_map = {
        prefix: plt.cm.tab10(i)
        for i, prefix in enumerate(unique_prefixes)
    }

    offsets = np.linspace(
        -bar_width * (len(totals) - 1) / 2,
        bar_width * (len(totals) - 1) / 2,
        len(totals),
    )

    plt.figure(figsize=(6, 5))

    for i, (total, prefix, label) in enumerate(zip(totals, prefixes, labels)):
        plt.bar(
            cluster_x + offsets[i],
            total,
            width=bar_width,
            label=prefix,
            color=color_map[prefix],
            edgecolor="black",
        )

    plt.xticks([cluster_x], ["SGEMM"])
    plt.ylabel("Total Time (adjusted)")
    plt.title("SGEMM Total Time (w subtracted if present)")

    # Deduplicate legend entries
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Configuration")

    output_pdf = "sgemm_total_times.pdf"
    plt.tight_layout()
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {output_pdf}")

if __name__ == "__main__":
    main()

