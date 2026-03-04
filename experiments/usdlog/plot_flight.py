import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    import jax
    jax.config.update("jax_platform_name", "cpu")
except ImportError:
    pass

try:
    import mpl_fontkit as fk
    fk.install("Inter")
    fk.set_font("Inter")
except ImportError:
    fk = None
textwidth = 2*3.4127
color_array = sns.color_palette("muted", 10)

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
# 31: in_data[32] = ctrltargetZ.x;
# 32: in_data[33] = ctrltargetZ.y;
# 33: in_data[34] = ctrltargetZ.z;


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


def _log_keys_with_index(data: dict):
    for evt, evdata in data.items():
        vars_excl_ts = [k for k in evdata.keys() if k != "timestamp"]
        print(f"[{evt}] variables ({len(vars_excl_ts)}):")
        for i, name in enumerate(vars_excl_ts):
            print(f"  {i:3d}: {name}")


def main():
    parser = argparse.ArgumentParser(description="Plot first three columns of a usdlog file using cfusdlog.")
    parser.add_argument("file_usd", help="Path to usdlog file (e.g., 1-1-takeoff)")
    parser.add_argument("--event", help="Event name to plot (defaults to fixedFrequency or first event)", default=None)
    parser.add_argument("--xy_zx", action="store_true", help="Also plot payload XY and ZX planes from indata (payload_pos_err + ctrl_target_Z)")
    args = parser.parse_args()

    data = cfusdlog.decode(args.file_usd)
    if not data:
        print("No data decoded.")
        return

    # Log all keys with index on load
    _log_keys_with_index(data)

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

    t = _time_to_seconds(event["timestamp"])
    mask_15s = t <= 15.0
    if not np.any(mask_15s):
        print("No samples at or before 15s; nothing to plot.")
        return
    t = t[mask_15s]

    err_keys = [f"ctrlrl.in{i}" for i in range(3)]
    err_norm = None
    steady_state_rmse = None
    settled_window = (2.5, 11.4)
    if all(k in event for k in err_keys):
        err = np.vstack([event[k] for k in err_keys]).T[mask_15s]
        err_norm = np.linalg.norm(err, axis=1)
        mask = (t >= settled_window[0]) & (t <= settled_window[1])
        if np.any(mask):
            steady_state_rmse = float(np.sqrt(np.mean(err_norm[mask] ** 2)))
        else:
            print("Error norm: no samples between 2.5s and 11s; skipping settled stats.")
    else:
        print("Error norm: missing ctrlrl.in0..2; skipping payload error plot.")

    z_target_key = next((k for k in ("ctrltargetZ.z", "ctrlrl.in33") if k in event), None)
    z_target = (event[z_target_key][mask_15s] / 1000) if z_target_key else None
    if z_target is None:
        print("RMSE: missing target Z key; skipping target plot.")

    rmse_axis_needed = err_norm is not None or z_target is not None
    if not rmse_axis_needed:
        print("No RMSE or Z target data to plot.")
        return

    fig, ax_err = plt.subplots(figsize=(textwidth, textwidth * 0.25), layout="constrained")
    ax_err.axvspan(
        settled_window[0],
        settled_window[1],
        color=color_array[4],
        alpha=0.08,
        label="Steady-state window",
        zorder=0,
    )
    if err_norm is not None:
        ax_err.plot(t, err_norm, label="‖e‖ [m]", color=color_array[0], zorder=2)
    if steady_state_rmse is not None:
        ax_err.hlines(
            steady_state_rmse,
            settled_window[0],
            settled_window[1],
            colors=[color_array[2]],
            label=f"Steady-state RMSE ({steady_state_rmse:.3f} m)",
            linewidth=1.5,
            zorder=3,
        )
    if z_target is not None:
        ax_err.plot(
            t,
            z_target,
            label="Target $z$",
            color=color_array[3],
            linestyle="--",
            linewidth=1.0,
            zorder=10,
        )
    ax_err.set_ylabel("m")
    ax_err.set_xlabel("time [s]")
    ax_err.grid(True)
    ax_err.legend()

    base_name = os.path.splitext(os.path.basename(args.file_usd))[0]
    plt.savefig(
        f"{base_name}_takeoff_land_plot.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.show()


if __name__ == "__main__":
    main()
