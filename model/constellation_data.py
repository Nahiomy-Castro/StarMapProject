import os
import json
import streamlit as st

class ConstellationData:
    def __init__(self):
        self.constellations = self.load_constellations()

    @st.cache_data
    def load_constellations(_self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_path = os.path.join(project_root, "data", "constellations.json")
        with open(json_path) as f:
            data = json.load(f)
        return data["constellations"]
