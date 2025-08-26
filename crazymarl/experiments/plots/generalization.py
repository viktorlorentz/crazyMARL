from generalizationhelper import run_experiments
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_command(cmd):
    """
    Runs a single shell command.
    Returns a tuple of (command, returncode, stdout, stderr).
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=False,
            capture_output=True,
            text=True
        )
        return (cmd, result.returncode, result.stdout, result.stderr)
    except Exception as e:
        return (cmd, -1, "", f"Exception: {e}")

def main():
    tests1 = {
    "obs_noise": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "payload_mass": [0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001],
    "cable_length": [0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21, 0.2],
    "act_noise": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }

    tests2 = {
        "obs_noise": [ 2.5, 3.0, 3.5, 4.0],
        "cable_length": [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0],
        "payload_mass": [0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03],
    }

    tests3 = {
       
        "cable_length": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
       
    }

    tests = tests3
    commands = []
    for parameter, values in tests.items():
        print(f"Running generalization tests for {parameter} with values: {values}")
        for value in values:
            print(f"Testing {parameter}={value}")
            cmd = f"unset MUJOCO_GL && python crazymarl/experiments/plots/generalizationhelper.py --parameter {parameter} --values {value}"
            print(f"Executing command: {cmd}")
            commands.append(cmd)

    max_workers = 10  # number of parallel workers

    print(f"Starting {len(commands)} commands with up to {max_workers} in parallel.\n")

    # Use ThreadPoolExecutor since subprocess.run frees the GIL
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all commands
        future_to_cmd = {executor.submit(run_command, cmd): cmd for cmd in commands}

        # As each finishes, print its output
        for future in as_completed(future_to_cmd):
            cmd = future_to_cmd[future]
            try:
                cmd, returncode, stdout, stderr = future.result()
                print(f"---\nCommand: {cmd}\nExit Code: {returncode}")
                if stdout:
                    print(f"Stdout:\n{stdout.strip()}")
                if stderr:
                    print(f"Stderr:\n{stderr.strip()}")
            except Exception as exc:
                print(f"Command {cmd!r} generated an exception: {exc}")

    print("\nAll commands completed.")

if __name__ == "__main__":
    main()



