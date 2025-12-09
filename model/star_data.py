#                                                     _______
#                                                     \  ___ `'.                                     _________   _...._
#                                                      ' |--.\  \                                    \        |.'      '-. .-.          .-
#                  .|            .-,.--.               | |    \  '               .|                   \        .'```'.    '.\ \        / /
#                .' |_     __    |  .-. |              | |     |  '    __      .' |_     __            \      |       \     \\ \      / /
#           _  .'     | .:--.'.  | |  | |              | |     |  | .:--.'.  .'     | .:--.'.           |     |        |    | \ \    / /
#         .' |'--.  .-'/ |   \ | | |  | |              | |     ' .'/ |   \ |'--.  .-'/ |   \ |          |      \      /    .   \ \  / /
#        .   | / |  |  `" __ | | | |  '-               | |___.' /' `" __ | |   |  |  `" __ | |  ,.--.   |     |\`'-.-'   .'     \ `  /
#      .'.'| |// |  |   .'.''| | | | ________________ /_______.'/   .'.''| |   |  |   .'.''| | //    \  |     | '-....-'`        \  /
#    .'.'.-'  /  |  '.'/ /   | |_| ||________________|\_______|/   / /   | |_  |  '.'/ /   | |_\\    / .'     '.                 / /
#    .'   \_.'   |   / \ \._,\ '/|_|                               \ \._,\ '/  |   / \ \._,\ '/ `'--''-----------'           |`-' /
#                `'-'   `--'  `"                                    `--'  `"   `'-'   `--'  `"                                '..'
# star_data.py
import os # For handling files
import pandas as pd
import numpy as np
import streamlit as st

# This class builds an object that can handle all star data
class StarData:
    def __init__(self): # Constructor
        self.stars = self.load_stars() # Calls the load_stars function to read CSV file

    @st.cache_data # Faster loading if already opened
    # This method just loads the CSV
    def load_stars(_self):
        # Absolute path to CSV
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(project_root, "data", "hyg_v42_updated.csv")
        stars = pd.read_csv(csv_path)
        # We try to clean the data from NaN
        stars['proper'] = stars['proper'].fillna("")
        stars['bf'] = stars['bf'].fillna("")
        return stars

    # Method to search stars by any query
    def find_star(self, query):
        q = query.lower() # lowercase
        matches = self.stars[ # this selects only the rows that match the query
            self.stars['proper'].str.lower().str.contains(q) | # check if the proper name contains the query
            self.stars['bf'].str.lower().str.contains(q) | # same for bf
            self.stars['hip'].astype(str).str.contains(q) # checks HIP number
        ]
        return matches # We return all matching rows

    # Changes the star positions so that the target star is at the origin (0,0,0)
    def recenter(self, cx, cy, cz): # coordinates of the target star
        self.stars['cx'] = self.stars['x'] - cx
        self.stars['cy'] = self.stars['y'] - cy
        self.stars['cz'] = self.stars['z'] - cz
        # Subtracting them from every star's x,y,z moves all stars to be relative to the target star

        # This computes the distance from the target str using the 3D distance formula
        self.stars['dist_from_target'] = np.sqrt(
            self.stars['cx']**2 + self.stars['cy']**2 + self.stars['cz']**2
        )
        # And then we store it in a new column in the df

    # Returns only the stars that are within a user-specified distance from the target star
    def filter_distance(self, dist_pc):
        return self.stars[self.stars['dist_from_target'] <= dist_pc].copy()
    # we make a separate copy to avoid accidentally changing the original data.
