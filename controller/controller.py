#            _..._       .-'''-.                                    .-'''-.
#         .-'_..._''.   '   _    \                                 '   _    \  .---..---.
#       .' .'      '.\/   /` '.   \    _..._                     /   /` '.   \ |   ||   |      __.....__                 _________   _...._
#      / .'          .   |     \  '  .'     '.                  .   |     \  ' |   ||   |  .-''         '.               \        |.'      '-. .-.          .-
#     . '            |   '      |  '.   .-.   .     .|  .-,.--. |   '      |  '|   ||   | /     .-''"'-.  `. .-,.--.      \        .'```'.    '.\ \        / /
#     | |            \    \     / / |  '   '  |   .' |_ |  .-. |\    \     / / |   ||   |/     /________\   \|  .-. |      \      |       \     \\ \      / /
#     | |             `.   ` ..' /  |  |   |  | .'     || |  | | `.   ` ..' /  |   ||   ||                  || |  | |       |     |        |    | \ \    / /
#     . '                '-...-'`   |  |   |  |'--.  .-'| |  | |    '-...-'`   |   ||   |\    .-------------'| |  | |       |      \      /    .   \ \  / /
#      \ '.          .              |  |   |  |   |  |  | |  '-                |   ||   | \    '-.____...---.| |  '-,.--.   |     |\`'-.-'   .'     \ `  /
#       '. `._____.-'/              |  |   |  |   |  |  | |                    |   ||   |  `.             .' | |   //    \  |     | '-....-'`        \  /
#         `-.______ /               |  |   |  |   |  '.'| |                    '---''---'    `''-...... -'   | |   \\    / .'     '.                 / /
#                  `                |  |   |  |   |   / |_|                                                  |_|    `'--''-----------'           |`-' /
#                                   '--'   '--'   `'-'                                                                                            '..'
# controller.py
import os, sys # For interacting with the filesystems and to manipulate Python path

#                | Removes the filename and leaves just the folder containing it
#                ↓                                ↓ name of this file (controller.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#     ↑                          ↑ turns that into a full path
#     | now, CURRENT_DIR equals .../controller

MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "model"))
# goes up one folder then into model. MODEL_DIR basically points to .../model

if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)
    # Check /model folder when searching modules so...
    # this import ↓ can work correctly
from model.stars_factory import StarsFactory

class Controller:
    # Constructor
    def __init__(self, star_model, const_model):
        self.star_model = star_model
        self.const_model = const_model

    # Find target star & recenter universe
    # We first get the query ↓ from the user
    def get_target(self, query):
        result = self.star_model.find_star(query)
        #                ↑ Then we search if that star exists
        if len(result) == 0:
            # The find_star(query) can return 0 or more matches. Like a list.
            # If len(result) == 0 -> no matches,
            # If len(result) > 0 -> 1 or more matches. So, what do we do when we have more matches?
            # We give the first match back.
            return None # Otherwise, we return nothing.
        return result.iloc[0] # Return first match.
    # Our view.py has a way to autocomplete the search in the box by selecting
    # the correct proper or b/f. This method is more to have a safe-net
    # because users are unpredictable when typing.

    # This reads the target star's position.
    # First, we get the star   ↰           ↱        and distance the user selected
    def prepare_stars(self, target, max_distance_pc):
        # The database has coordinates x, y and z for each star. We just grab that and
        # save it in variable centered x, centered y, and centered z.
        cx, cy, cz = target['x'], target['y'], target['z']

        # To have the POV of the target star, we need to make their coordinates the origin (0, 0, 0)
        self.star_model.recenter(cx, cy, cz)
        # This line removes stars that are farther than the user's selected distance.
        filtered = self.star_model.filter_distance(max_distance_pc)

        # Next, we create new columns by copying the filtered coordinates.
        # This is because later on Plotly might apply transformations (zoom or rotations)
        # or even scale the coordinates to look better on the graphs. I don't want plotly to be doing
        # all of that in the original filtered coordinates. So, we keep the og as backup.
        filtered['cx_scaled'] = filtered['cx']
        filtered['cy_scaled'] = filtered['cy']
        filtered['cz_scaled'] = filtered['cz']
        # Finally, return the prepared coordinates.
        return filtered

    # Then we hand everything to the plotting system. AKA Factory.
    def build_plot(self, stars_filtered):
        return StarsFactory.create_plot(
            "constellations",
            stars_filtered,
            self.const_model.constellations
        )
    # Basically, it draws a star map showing only the nearby stars and connects them
    # using the JSON file that stores the constellations coordinates.
