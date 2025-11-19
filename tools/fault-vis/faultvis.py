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

def visualize(groups, output="faults_by_ts.html", window_size=0.0001):
    # Flatten groups into a DataFrame
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
    # Normalize timestamp and address
    # ---------------------------------------------------------
    t0 = df["time"].min()
    a0 = df["address"].min()

    df["time_norm"] = df["time"] - t0
    df["address_norm"] = df["address"] - a0

    # ---------------------------------------------------------
    # WINDOWING / BATCHING LOGIC
    # ---------------------------------------------------------
    df["window"] = (df["time_norm"] // window_size).astype(int)
    windows = sorted(df["window"].unique())

    # Base scatter for first window
    first_window = windows[0]
    d0 = df[df["window"] == first_window]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d0["address_norm"],
        y=d0["size"],
        mode="markers",
        marker=dict(size=10),
        text=[
            f"addr={hex(a)}, cpu={c}, dur={d}"
            for a, c, d in zip(d0["address"], d0["cpu"], d0["duration"])
        ],
        hoverinfo="text",
    ))

    # ---------------------------------------------------------
    # Create a frame per window
    # ---------------------------------------------------------
    frames = []
    for w in windows:
        dft = df[df["window"] == w]
        start_t = w * window_size

        frames.append(go.Frame(
            data=[go.Scatter(
                x=dft["address_norm"],
                y=dft["size"],
                mode="markers",
                marker=dict(size=10),
                text=[
                    f"addr={hex(a)}, cpu={c}, dur={d}"
                    for a, c, d in zip(dft["address"], dft["cpu"], dft["duration"])
                ],
                hoverinfo="text",
            )],
            name=str(w),
        ))
    fig.frames = frames

    # ---------------------------------------------------------
    # Slider: each step represents a window
    # ---------------------------------------------------------
    sliders = [{
        "steps": [
            {
                "args": [[str(w)], {"frame": {"duration": 0}, "mode": "immediate"}],
                "label": f"{w * window_size:.6f}s",
                "method": "animate",
            }
            for w in windows
        ],
        "transition": {"duration": 0},
        "x": 0.1,
        "xanchor": "left",
        "y": -0.1,
        "yanchor": "top",
    }]

    fig.update_layout(
        title=f"UVM Faults — Windowed View (window={window_size}s)",
        xaxis_title="Address (normalized)",
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
        print("Usage: python3 faultvis.py <input.txt> [output.html] [window_size]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else "faults_by_ts.html"
    window_size = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.0001

    groups = parse_uvm_groups(input_file)
    visualize(groups, output_file, window_size)

if __name__ == "__main__":
    main()

