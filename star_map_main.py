import importlib.util
import os
import sys

# Path setup: project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Helper to load view.py
def import_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load the Streamlit UI from view/view.py
View = import_from_path("view", os.path.join(PROJECT_ROOT, "view", "view.py"))

# Run the Streamlit app
if __name__ == "__main__":
    View.main()
