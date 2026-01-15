import glob
import matplotlib.pyplot as plt


MODES = ["native", "tlb", "overlap"]


def parse_gflops_file(filename):
    """
    Returns (problem_size, avg_gflops)
    """
    gflops = []
    problem_size = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # GPU,32768,16.818781,4183.874512
            if line.startswith("GPU"):
                parts = line.split(",")

                if len(parts) < 4:
                    continue

                if problem_size is None:
                    problem_size = int(parts[1])

                gflops.append(float(parts[-1]))

    if not gflops:
        raise ValueError(f"No GFLOPS data found in {filename}")

    avg_gflops = sum(gflops) / len(gflops)
    return problem_size, avg_gflops


def main():
    plt.figure()

    for mode in MODES:
        files = glob.glob(f"*{mode}.txt")

        if not files:
            print(f"Warning: no files found for mode '{mode}'")
            continue

        sizes = []
        avgs = []

        for fname in files:
            size, avg = parse_gflops_file(fname)
            sizes.append(size)
            avgs.append(avg)

        # Sort by problem size
        sizes, avgs = zip(*sorted(zip(sizes, avgs)))

        if mode == "overlap":
            print("here")
            mode = "overlap \n+ tlb" 

        plt.plot(sizes, avgs, marker='o', label=mode)

    plt.xlabel("Problem Size")
    plt.ylabel("Average GFLOPS")
    plt.title("SGEMM Average GFLOPS Across Optimizations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("gflops_comparison.pdf")


if __name__ == "__main__":
    main()

