#controller.py
import os, sys

# Path to the "model" folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "model"))

if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

from stars_factory import StarsFactory

class Controller:
    def __init__(self, star_model, const_model):
        self.star_model = star_model
        self.const_model = const_model

    # Find target star & recenter universe
    def get_target(self, query):
        result = self.star_model.find_star(query)
        if len(result) == 0:
            return None
        return result.iloc[0]

    def prepare_stars(self, target, max_distance_pc):
        cx, cy, cz = target['x'], target['y'], target['z']

        self.star_model.recenter(cx, cy, cz)
        filtered = self.star_model.filter_distance(max_distance_pc)

        # scale
        scale_factor = 20
        filtered['cx_scaled'] = filtered['cx'] * scale_factor
        filtered['cy_scaled'] = filtered['cy'] * scale_factor
        filtered['cz_scaled'] = filtered['cz'] * scale_factor

        return filtered

    def build_plot(self, stars_filtered):
        # Just delegate plot creation to the factory
        return StarsFactory.create_plot(
            "constellations",
            stars_filtered,
            self.const_model.constellations
        )
