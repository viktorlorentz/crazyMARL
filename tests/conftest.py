# Ensure the project root is on PYTHONPATH so tests can import the crazymarl package
import os
import sys
test_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(test_dir, os.pardir))
sys.path.insert(0, project_root)
