import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import chain

plt.rcParams['hatch.linewidth'] = 0.4

# Load data
df = pd.read_csv("future_work.csv")

benchmarks = df["benchmark"].unique()
groups = ["UVM","Folio HMM", "Native HMM"]

hatches = {"UVM": "x", "Native HMM": "\\", "Folio HMM": "o"}

phases = [
    "HMM setup",
    "UVM alloc",
    "HMM pages",
    "UVM wait",
    "UVM finalize/map (excluding wait)",
    "HMM finalize"
]

phase_colors = {
    "HMM setup": "tab:blue",
    "UVM alloc": "tab:orange",
    "HMM pages": "tab:green",
    "UVM wait": "tab:brown",
    "UVM finalize/map (excluding wait)": "tab:red",
    "HMM finalize": "tab:purple"
}

group_handles = [
    mpatches.Patch(
        facecolor="white",          # or any neutral color
        edgecolor="black",
        hatch=hatches[g],
        label=g
    )
    for g in groups
]

phase_handles = [
    
    mpatches.Patch(facecolor=phase_colors[p], edgecolor="black", label=p.replace(
        "UVM finalize/map (excluding wait)",
        "UVM finalize/map\n(excluding wait)"
    ))
    for p in phases
]

totals = {}
for b in benchmarks:
    for g in groups:
        total = df[
            (df.benchmark == b) &
            (df.group == g)
        ]["value"].sum()
        totals[(b, g)] = total

print(totals)

max_total = {}
for b in benchmarks:
    max_total[b] = max(
        totals[(b, "Folio HMM")],
        totals[(b, "Native HMM")],
        totals[(b, "UVM")]
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

        print(values)

        values = np.array(values)

        ax.bar(
            x + i * bar_width,
            values,
            bar_width,
            bottom=bottoms,
            color=phase_colors[phase],
            edgecolor="black",
            linewidth=0.8,
            label=phase if i == 0 else "",
            hatch=hatches[group]
        )

        bottoms += values

# Axis formatting
ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(benchmarks)
ax.set_ylabel("Time Relative to Slowest Case (%)")
ax.set_title("Folio HMM vs Native HMM Breakdown")

# Legend (phases only)

legend_phases = [
    p.replace(
        "UVM finalize/map (excluding wait)",
        "UVM finalize/map\n(excluding wait)"
    )
    for p in phases
]

handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles[:len(phases)], legend_phases, title="Phase", loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
#ax.legend(handles=group_handles, title="Group", loc="upper right")
ax.legend(handles=list(chain(phase_handles, group_handles)),
          loc="center left", bbox_to_anchor=(1.02, 0.5),
          title="Phase / Group", handlelength=2.5, handleheight=2.0, handletextpad=0.5, borderaxespad=0.0)

plt.tight_layout(rect=[0, 0, 0.82, 1])

# Save as PDF (vector, paper-ready)
plt.savefig("uvm_hmm_test.pdf", format="pdf", bbox_inches="tight")
plt.show()

