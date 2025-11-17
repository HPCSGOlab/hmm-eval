import plotly.express as px
import json

def visualize(groups, output="faults.html"):
    # Flatten groups into a DataFrame-like structure
    times = []
    addrs = []
    sizes = []
    cpus = []
    w_ids = []

    for g_index, g in enumerate(groups):
        for f in g["faults"]:
            times.append(f["c_time"])
            addrs.append(f["address"])
            sizes.append(f["size"])
            cpus.append("CPU" if f["cpu"] == 1 else "GPU")
            w_ids.append(g_index)

    import pandas as pd
    df = pd.DataFrame({
        "time": times,
        "address": addrs,
        "size": sizes,
        "cpu": cpus,
        "group": w_ids,
    })

    fig = px.scatter(
        df,
        x="time",
        y="size",
        color="cpu",
        hover_data=["address", "group"],
        title="UVM Fault Timeline",
    )

    fig.write_html(output)
    print(f"Wrote {output}")
