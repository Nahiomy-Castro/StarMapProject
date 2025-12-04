import os
import pandas as pd
import numpy as np
import streamlit as st

class StarData:
    def __init__(self):
        self.stars = self.load_stars()

    @st.cache_data
    def load_stars(_self):
        # Absolute path to CSV
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(project_root, "data", "hyg_v42_updated.csv")
        stars = pd.read_csv(csv_path)
        stars['proper'] = stars['proper'].fillna("")
        stars['bf'] = stars['bf'].fillna("")
        return stars

    # ... rest of your class remains the same


    def find_star(self, query):
        q = query.lower()
        matches = self.stars[
            self.stars['proper'].str.lower().str.contains(q) |
            self.stars['bf'].str.lower().str.contains(q) |
            self.stars['hip'].astype(str).str.contains(q)
        ]
        return matches

    def recenter(self, cx, cy, cz):
        self.stars['cx'] = self.stars['x'] - cx
        self.stars['cy'] = self.stars['y'] - cy
        self.stars['cz'] = self.stars['z'] - cz

        self.stars['dist_from_target'] = np.sqrt(
            self.stars['cx']**2 + self.stars['cy']**2 + self.stars['cz']**2
        )

    def filter_distance(self, dist_pc):
        return self.stars[self.stars['dist_from_target'] <= dist_pc].copy()
