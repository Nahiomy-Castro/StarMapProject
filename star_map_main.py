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

from model.star_model import GraphModel, ClusterModel

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

    # Initializes the graph model
    graph_model = GraphModel(nrows=5000, k_neighbors=5)
    # Initializes the Strategy Pattern ModelTrainer with the graph model
    trainer = StarModel.ModelTrainer(graph_model)
    # Runs the graph model pipeline
    trainer.run_training_pipeline()

    # Initializes the cluster model
    cluster_model = ClusterModel(nrows=5000, n_clusters=10)
    # Changes the strategy to the cluster model
    trainer.set_strategy(cluster_model)
    # Runs the cluster model pipeline
    trainer.run_training_pipeline()

    # Initializes the traditional model
    traditional_model = StarModel.TradModel(nrows=5000)
    # Changes the strategy to the traditional model
    trainer.set_strategy(traditional_model)
    # Runs the traditional model pipeline
    trainer.run_training_pipeline()

    # Calls the view method to start the program.
    # It takes the traditional model as a parameter so the view method can predict the necessary values.
    View.view(traditional_model)
