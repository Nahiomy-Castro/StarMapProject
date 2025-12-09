#     __  __   ___            .--.   _..._         _________   _...._
#    |  |/  `.'   `.          |__| .'     '.       \        |.'      '-. .-.          .-
#    |   .-.  .-.   '         .--..   .-.   .       \        .'```'.    '.\ \        / /
#    |  |  |  |  |  |    __   |  ||  '   '  |        \      |       \     \\ \      / /
#    |  |  |  |  |  | .:--.'. |  ||  |   |  |         |     |        |    | \ \    / /
#    |  |  |  |  |  |/ |   \ ||  ||  |   |  |         |      \      /    .   \ \  / /
#    |  |  |  |  |  |`" __ | ||  ||  |   |  | ,.--.   |     |\`'-.-'   .'     \ `  /
#    |__|  |__|  |__| .'.''| ||__||  |   |  |//    \  |     | '-....-'`        \  /
#                    / /   | |_   |  |   |  |\\    / .'     '.                 / /
#                    \ \._,\ '/   |  |   |  | `'--''-----------'           |`-' /
#                     `--'  `"    '--'   '--'                               '..'
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
StarModel = import_from_path("star_model", os.path.join(PROJECT_ROOT, "model", "star_model.py"))

# Run the Streamlit app
if __name__ == "__main__":
    traditional_model = StarModel.TradModel(nrows=5000)
    trainer = StarModel.ModelTrainer(traditional_model)
    trainer.run_training_pipeline()
    View.view(traditional_model)
