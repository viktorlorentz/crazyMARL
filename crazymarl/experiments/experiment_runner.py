import os
import sys

# add project root to PYTHONPATH so 'crazymarl' can be imported when running script directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

import argparse


def main():
    # scan configs dir for all .yaml/.yml files
    configs_dir = os.path.join(os.getcwd(), "crazymarl", "experiments", "configs")
    config_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(configs_dir)
        if f.endswith((".yaml", ".yml"))
    ]

    parser = argparse.ArgumentParser(
        description="Run train_multiquad_ippo with a custom config"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config-file",
        type=str,
        help="Path to YAML config file to use"
    )
    group.add_argument(
        "--config",
        type=str,
        choices=config_names,
        help="Choose one of: " + ", ".join(config_names)
    )
    args = parser.parse_args()


    if args.config:
        config_file = os.path.join(os.getcwd(),"crazymarl", "experiments", "configs", f"{args.config}.yaml")
    else:
        config_file = args.config_file

    from crazymarl.train.train_multiquad_ippo import main as train_main
    train_main(config_file=config_file)

if __name__ == "__main__":
    main()
