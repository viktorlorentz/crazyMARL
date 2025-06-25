import importlib
import sys

def test_train_cli_runs(monkeypatch):
    """
    Smoke test: ensure that the train CLI entrypoint runs without raising an error
    for the built-in 'test' config.
    """
    # Import the CLI module
    train_module = importlib.import_module('crazymarl.train.train')
    # Monkeypatch the heavy training function in train_multiquad_ippo to a no-op
    ippo_module = importlib.import_module('crazymarl.train.train_multiquad_ippo')
    monkeypatch.setattr(ippo_module, 'main', lambda config_file=None: None)
    # Simulate command-line invocation
    monkeypatch.setattr(sys, 'argv', ['train.py', '--config', 'test'])
    # Call the main function; should complete without error
    train_module.main()
