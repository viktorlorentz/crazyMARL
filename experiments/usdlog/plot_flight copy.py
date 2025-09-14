import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

import cfusdlog


#mapping in file
# 0:  in_data[0]  = payload_pos_err_clamped.x;
# 0:  in_data[1]  = payload_pos_err_clamped.y;
# 1:  in_data[2]  = payload_pos_err_clamped.z;
# 2:  in_data[3]  = payload_vel_world.x;
# 3:  in_data[4]  = payload_vel_world.y;
# 4:  in_data[5]  = payload_vel_world.z;
# 5:  in_data[6]  = rel_pos.x; // rel_pos is 0 for no payload
# 6:  in_data[7]  = rel_pos.y; // rel_vel is
# 7:  in_data[8]  = rel_pos.z; // rel_acc is
# 8:  in_data[9]  = R.m[0][0];
# 9:  in_data[10] = R.m[0][1];
# 10: in_data[11] = R.m[0][2];
# 11: in_data[12] = R.m[1][0];
# 12: in_data[13] = R.m[1][1];
# 13: in_data[14] = R.m[1][2];
# 14: in_data[15] = R.m[2][0];
# 15: in_data[16] = R.m[2][1];
# 16: in_data[17] = R.m[2][2];
# 17: in_data[18] = vel_world.x; // linvels 0.0f for no payload
# 18: in_data[19] = vel_world.y; // linvels 0.0f for no payload
# 19: in_data[20] = vel_world.z; // linvels 0.0f for no payload
# 20: in_data[21] = radians(sensors->gyroNoLpf.x);
# 21: in_data[22] = radians(sensors->gyroNoLpf.y);
# 22: in_data[23] = radians(sensors->gyroNoLpf.z);
# 23: in_data[24] = lastAction[0];
# 24: in_data[25] = lastAction[1];
# 25: in_data[26] = lastAction[2];
# 26: in_data[27] = lastAction[3];
# 27: in_data[28] = ctrl_rl.out[0];
# 28: in_data[29] = ctrl_rl.out[1];
# 29: in_data[30] = ctrl_rl.out[2];
# 30: in_data[31] = ctrl_rl.out[3];
# 31: in_data[32] = ctrl_target_Z.x;
# 32: in_data[33] = ctrl_target_Z.y;
# 33: in_data[34] = ctrl_target_Z.z;


def _time_to_seconds(ts: np.ndarray) -> np.ndarray:
    ts = ts.astype(np.float64)
    ts0 = ts[0]
    span = ts.max() - ts0
    # Heuristics: try to detect unit and convert to seconds
    if span > 1e9:      # nanoseconds
        return (ts - ts0) / 1e9
    elif span > 1e6:    # microseconds
        return (ts - ts0) / 1e6
    elif span > 1e3:    # milliseconds
        return (ts - ts0) / 1e3
    else:               # seconds already
        return ts - ts0


def main():
    parser = argparse.ArgumentParser(description="Plot first three columns of a usdlog file using cfusdlog.")
    parser.add_argument("file_usd", help="Path to usdlog file (e.g., 1-1-takeoff)")
    parser.add_argument("--event", help="Event name to plot (defaults to fixedFrequency or first event)", default=None)
    args = parser.parse_args()

    data = cfusdlog.decode(args.file_usd)

    if not data:
        print("No data decoded.")
        return

    # Pick event: prefer fixedFrequency, else first with >= 2 variables (excluding timestamp)
    event_name = args.event if args.event in data else None
    if event_name is None:
        if "fixedFrequency" in data:
            event_name = "fixedFrequency"
        else:
            # find first with at least 2 variables besides timestamp
            for k, v in data.items():
                vars_excl_ts = [vn for vn in v.keys() if vn != "timestamp"]
                if len(vars_excl_ts) >= 2:
                    event_name = k
                    break
    if event_name is None:
        print("Could not find an event with at least two variables to plot.")
        return

    event = data[event_name]
    vars_excl_ts = [k for k in event.keys() if k != "timestamp"]
    if len(vars_excl_ts) == 0:
        print(f"Event '{event_name}' has no variables.")
        return

    # First up to three "columns"
    to_plot = vars_excl_ts[24:28]

    t = _time_to_seconds(event["timestamp"])

    nrows = len(to_plot)
    fig, axes = plt.subplots(nrows, 1, sharex=True, figsize=(10, 5))
    if nrows == 1:
        axes = [axes]

    for ax, var in zip(axes, to_plot):
        ax.plot(t, event[var], label=var)
        ax.set_ylabel(var)
        ax.grid(True)
        ax.legend()

    title = f"{os.path.basename(args.file_usd)} - {event_name} (first three columns)"
    fig.suptitle(title)
    axes[-1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
