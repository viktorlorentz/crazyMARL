from generalizationhelper import run_experiments

default_env_config = {
  
    "num_quads": 2,              # Number of quads in the environment
    "episode_length": 2500,
    "target_start_ratio": 0.0,
    "obs_noise": 0.0,
    "policy_freq": 250.0,         # Policy frequency in Hz
    "sim_steps_per_action": 1,    # Physics steps between control actions
    "obs_noise": 0.0,             # Observation noise parameter
    "act_noise": 0.0,             # Actuator noise parameter
    "max_thrust_range": 0.15,     # Range for randomizing thrust
    "cable_length": 0.3,          # Cable length connecting payload to quads
    "trajectory": None,           # Default trajectory
    "payload_mass": 0.01,         # Mass of the payload
    "auto_reset": False,          # Auto-reset env on done
    "seed": 0,                   
    
}



def eval_generalization(parameter, values, default_env_config=default_env_config):
    config = default_env_config.copy()
    default_val = config[parameter]  # Store the default value

    for v in values:
        config[parameter] = v
        print(f"Running with {parameter}={v}")
        
        # Run the experiments with the modified config
        successful = run_experiments(
            num_envs=1000,
            env_config=config
        )

        rate = successful / 1000.0*100
        print(f"Success rate for {parameter}={v}: {rate:.2f}")
        # save as csv new line
        with open(f"results_generalization.csv", "a") as f:
            f.write(f"{parameter},{v},{rate:.2f},{default_val}\n")

eval_generalization("seed", [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])