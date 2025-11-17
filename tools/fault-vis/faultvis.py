#!/usr/bin/env python3
import re
import sys
import plotly.graph_objects as go
import pandas as pd

# ------------------------------------------------------------
# Parsing Logic
# ------------------------------------------------------------

def parse_uvm_groups(filename):
    # Capture bracketed timestamp: [ 2209.562393 ]
    ts_prefix = re.compile(r"^\[\s*([\d.]+)\s*\]\s*")

    groups = []
    current = {
        "c_events": [],   # list of {"ts": <timestamp>, "duration": <duration>}
        "faults": [],
        "w_time": None
    }

    last_c_event = None

    with open(filename, "r") as f:
        for line in f:
            m = ts_prefix.match(line)
            if not m:
                continue
            event_ts = float(m.group(1))
            line = ts_prefix.sub("", line).strip()
            if not line:
                continue

            # -------------------------------
            # c-event
            # -------------------------------
            if line.startswith("c,"):
                _, duration_str, flag = line.split(",")
                duration = int(duration_str)
                last_c_event = {
                    "ts": event_ts,  # actual timestamp
                    "duration": duration
                }
                current["c_events"].append(last_c_event)

            # -------------------------------
            # fault-event
            # -------------------------------
            elif line.startswith("fault,"):
                _, addr, size, cpu = line.split(",")
                current["faults"].append({
                    "address": int(addr),
                    "size": int(size),
                    "cpu": int(cpu),
                    "ts": event_ts,         # timestamp of fault
                    "c_event": last_c_event # attach previous c-event
                })
                last_c_event = None

            # -------------------------------
            # w-event (end of group)
            # -------------------------------
            elif line.startswith("w,"):
                _, duration_str, flag = line.split(",")
                current["w_time"] = event_ts  # timestamp of w
                groups.append(current)
                current = {
                    "c_events": [],
                    "faults": [],
                    "w_time": None
                }

    return groups

# ------------------------------------------------------------
# Visualization Logic
# ------------------------------------------------------------

def visualize(groups, output="faults_by_group.html"):
    # Flatten groups into a DataFrame
    rows = []
    for g_index, g in enumerate(groups):
        for f in g["faults"]:
            rows.append({
                "time": f["ts"],                  # use the actual timestamp
                "address": f["address"],
                "size": f["size"],
                "cpu": "CPU" if f["cpu"] == 1 else "GPU",
                "group": g_index,
                "duration": f["c_event"]["duration"] if f["c_event"] else None
            })

    df = pd.DataFrame(rows)

    # ----- Base scatter for first group -----
    first_group = df[df["group"] == 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=first_group["time"],
        y=first_group["size"],
        mode="markers",
        marker=dict(size=10),
        text=[f"addr={a}, cpu={c}, dur={d}" for a, c, d in zip(first_group["address"], first_group["cpu"], first_group["duration"])],
        hoverinfo="text",
    ))

    # ----- Frames: one per group -----
    frames = []
    for g in sorted(df["group"].unique()):
        dfg = df[df["group"] == g]
        frames.append(go.Frame(
            data=[go.Scatter(
                x=dfg["time"],
                y=dfg["size"],
                mode="markers",
                marker=dict(size=10),
                text=[f"addr={a}, cpu={c}, dur={d}" for a, c, d in zip(dfg["address"], dfg["cpu"], dfg["duration"])],
                hoverinfo="text",
            )],
            name=str(g)
        ))
    fig.frames = frames

    # ----- Slider -----
    sliders = [{
        "steps": [
            {
                "args": [[str(g)], {"frame": {"duration": 0}, "mode": "immediate"}],
                "label": f"group {g}",
                "method": "animate",
            } for g in sorted(df["group"].unique())
        ],
        "transition": {"duration": 0},
        "x": 0.1,
        "xanchor": "left",
        "y": -0.1,
        "yanchor": "top",
    }]

    fig.update_layout(
        title="UVM Faults — Scroll Through Groups",
        xaxis_title="Timestamp (s)",
        yaxis_title="Fault Size (bytes)",
        sliders=sliders,
        showlegend=False
    )

    fig.write_html(output)
    print(f"Wrote {output}")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 faultvis.py <input.txt> [output.html]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else "faults_by_group.html"

    groups = parse_uvm_groups(input_file)
    visualize(groups, output_file)

if __name__ == "__main__":
    main()

