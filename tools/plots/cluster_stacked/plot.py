import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("hmm_uvm.csv")

benchmarks = df["benchmark"].unique()
groups = ["UVM", "Native HMM"]
phases = [
    "HMM setup",
    "UVM alloc",
    "HMM pages",
    "UVM finalize/map",
    "HMM finalize"
]

phase_colors = {
    "HMM setup": "tab:blue",
    "UVM alloc": "tab:orange",
    "HMM pages": "tab:green",
    "UVM finalize/map": "tab:red",
    "HMM finalize": "tab:purple"
}

totals = {}
for b in benchmarks:
    for g in groups:
        total = df[
            (df.benchmark == b) &
            (df.group == g)
        ]["value"].sum()
        totals[(b, g)] = total

max_total = {}
for b in benchmarks:
    max_total[b] = max(
        totals[(b, "UVM")],
        totals[(b, "Native HMM")]
    )

x = np.arange(len(benchmarks))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(7, 4))

for i, group in enumerate(groups):
    bottoms = np.zeros(len(benchmarks))

    for phase in phases:
        values = []
        for b in benchmarks:
            row = df[
                (df.benchmark == b) &
                (df.group == group) &
                (df.phase == phase)
            ]
            raw = row.value.iloc[0] if not row.empty else 0
            #values.append(row.value.iloc[0] if not row.empty else 0)
            denom = max_total[b]
            values.append(raw / denom if total > 0 else 0)


        values = np.array(values)

        ax.bar(
            x + i * bar_width,
            values,
            bar_width,
            bottom=bottoms,
            color=phase_colors[phase],
            edgecolor="black",
            linewidth=0.5,
            label=phase if i == 0 else ""
        )

        bottoms += values

# Axis formatting
ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(benchmarks)
ax.set_ylabel("Time Relative to Slowest Case (%)")
ax.set_title("UVM vs Native HMM Breakdown")

# Legend (phases only)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:len(phases)], phases, title="Phase", loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

plt.tight_layout(rect=[0, 0, 0.82, 1])

# Save as PDF (vector, paper-ready)
plt.savefig("uvm_hmm_breakdown.pdf", format="pdf", bbox_inches="tight")
plt.show()

