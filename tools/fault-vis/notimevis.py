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

def visualize(groups, output="faults_simple_rects.html", rect_width=1):
    # ---------------------------------------------------------
    # Flatten groups into a DataFrame
    # ---------------------------------------------------------
    rows = []
    for g_index, g in enumerate(groups):
        for f in g["faults"]:
            rows.append({
                "time": f["ts"],
                "address": f["address"],
                "size": f["size"],
                "cpu": "CPU" if f["cpu"] == 1 else "GPU",
                "group": g_index,
                "duration": f["c_event"]["duration"] if f["c_event"] else None
            })

    df = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # Normalize time + address
    # ---------------------------------------------------------
    t0 = df["time"].min()
    a0 = df["address"].min()

    df["time_norm"] = df["time"] - t0
    df["address_norm"] = df["address"] - a0

    # Normalize fault size → rectangle height
    max_size = df["size"].max()
    df["height_norm"] = df["size"] / max_size

    # ---------------------------------------------------------
    # Build the figure
    # ---------------------------------------------------------
    fig = go.Figure()

    # ----------------------------------------
    # Add rectangles for every fault
    # ----------------------------------------
    shapes = []
    for _, row in df.iterrows():
        shapes.append({
            "type": "rect",
            "xref": "x",
            "yref": "y",
            "x0": row["time_norm"],
            "x1": row["time_norm"] + rect_width,   # wider rectangle
            "y0": row["address_norm"],
            "y1": row["address_norm"] + row["height_norm"],
            "line": {"width": 0},
            "fillcolor": "#1f77b4" if row["cpu"] == "CPU" else "#d62728",
            "opacity": 0.7,
        })

    fig.update_layout(shapes=shapes)

    # ---------------------------------------------------------
    # Add invisible scatter points for hover
    # ---------------------------------------------------------
    fig.add_trace(go.Scatter(
        x=df["time_norm"] + rect_width / 2,
        y=df["address_norm"] + df["height_norm"] / 2,
        mode="markers",
        marker=dict(size=1, opacity=1),
        text=[
            f"addr={hex(a)}, cpu={c}, size={s}, dur={d}"
            for a, c, s, d in zip(
                df["address"], df["cpu"], df["size"], df["duration"]
            )
        ],
        hoverinfo="text"
    ))

    # ---------------------------------------------------------
    # Layout
    # ---------------------------------------------------------
    fig.update_layout(
        title="UVM Faults — All Faults (Rectangles, No Windows)",
        xaxis_title="Time (normalized)",
        yaxis_title="Address (normalized)",
        showlegend=False
    )

    fig.write_html(output)
    print(f"Wrote {output}")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 faultvis.py <input.txt> [output.html] [window_size]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else "faults_by_ts.html"
    window_size = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.0001

    groups = parse_uvm_groups(input_file)
    visualize(groups, output_file, window_size)

if __name__ == "__main__":
    main()

