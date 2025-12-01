# starmap_app.py
"""
Place hyg_v42.csv and constellations.json in same folder and run:
streamlit run starmap_app.py in the terminal. Clicking Run wont work.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from dataclasses import dataclass, field # Makes writing classes WAY less complicated
from typing import Dict, List, Tuple, Any
# -----------------------------------------------------
#                       SETUP
# -----------------------------------------------------

# Set Streamlit page layout and title along with a little icon on the tab page.
st.set_page_config(layout="wide", page_title="Star Map — Float amongst stars", page_icon="💫")

"""
The HYG database stores distances in parsecs. One parsec corresponds to the distance at which the mean
radius of the earth's orbit subtends an angle of one second of arc.
That is complicated, so I changed it to light years.
1 parsec equal to about 3.26 light years (3.086 × 10 kilometres)
"""

PC_TO_LY = 3.26

""""
The database might have null values or strings instead of floats. We will try to get the x value
and convert it to float. If the data given is a string of characters or null, then instead of crashing,
the program will just return not a number safely.
"""

def null_values_exception(x, default=np.nan): # Input x, default value

    try:                  # Try..
        y = float(x)      # Convert the data into a float and save it into variable y
    except Exception:     # If something happens... (get a text or get a None, Null, NaN...)
        y = default       # Set the y variable to the default value (NaN)
    return y              # Return the value of y

# -----------------------------------------------------
# DATA MODEL — RESPONSIBLE FOR HANDLING STAR DATA ONLY
# -----------------------------------------------------
@dataclass # Decorator that skips writing __init__ and avoids repetitive work
class StarModel:
    """
    The job of this class is to load, clean, and search the data.
    """
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    # DataFrame: create pandas data frame each row is a star and each column is a property
    # (name, distance, magnitude, coordinates)
    # then default_factory=pd.DataFrame ensures that if I create a StarModel() without

    def prepare(self):
        """
        Normalize dataset and create derived columns.
        Called once right after loading the CSV.

        Why:
        - Centralizes transformations (Single Responsibility Principle).
        - Ensures every part of the app sees the same clean, consistent data.
        """
        # Ensure proper + Bayer/Flamsteed names always exist
        self.df['proper'] = self.df.get('proper', "").fillna("")
        self.df['bf'] = self.df.get('bf', "").fillna("")

        # HIP needs to be a number; invalid values become NaN
        self.df['hip'] = pd.to_numeric(self.df.get('hip', pd.Series()), errors='coerce')

        # Convert numerical fields safely
        for col in ['ra', 'dec', 'mag', 'dist']:
            self.df[col] = pd.to_numeric(self.df.get(col, pd.Series()), errors='coerce')

        # Add distance in light-years for user-facing controls
        self.df['dist_ly'] = self.df['dist'] * PC_TO_LY

        # Convert RA to degrees if dataset is in hours (0–24)
        if self.df['ra'].max() <= 24.1:
            self.df['ra_deg'] = self.df['ra'] * 15.0
        else:
            self.df['ra_deg'] = self.df['ra']

        # Guarantee x,y,z exist; needed for 3D plotting and POV shifts
        for axis in ['x', 'y', 'z']:
            if axis not in self.df.columns:
                self.df[axis] = np.nan

    # -------------------------------------------------
    # STAR LOOKUP FUNCTION — SEARCH BY NAME OR ID
    # -------------------------------------------------
    def find_star(self, query: str) -> pd.DataFrame:
        """
        Search star by:
        - proper name
        - Bayer/Flamsteed name
        - HIP ID

        Why:
        - Implements a clean interface for all search operations.
        - Controller/UI don't need to know how filtering works.
        """
        q = (query or "").strip().lower()
        if q == "":
            return pd.DataFrame()

        return self.df[
            self.df['proper'].str.lower().str.contains(q, na=False) |
            self.df['bf'].str.lower().str.contains(q, na=False) |
            self.df['hip'].astype(str).str.contains(q)
        ]


# ------------------------------
# Data Factory (Factory pattern)
# ------------------------------
class DataFactory:
    """Factory to create and return data objects (stars and constellations).
    Why: encapsulates file IO and caching policy in one component.
    """

    @staticmethod
    @st.cache_data  # keeps result in Streamlit cache; acts like a simple Singleton for the loaded data
    def load_stars(path: str = "hyg_v42.csv") -> pd.DataFrame:
        """Load HYG CSV, minimal cleaning."""
        df = pd.read_csv(path)
        # minimal immediate cleanup
        df['proper'] = df['proper'].fillna("")
        df['bf'] = df['bf'].fillna("")
        # Return raw df (StarModel.prepare will further normalize)
        return df

    @staticmethod
    @st.cache_data
    def load_constellations(path: str = "constellations.json") -> List[Dict[str, Any]]:
        """Load the stellarium/IAU JSON that contains constellation lines.

        Why: centralizes format expectation. The returned structure is used by renderer.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw.get("constellations", [])

# ------------------------------
# Renderer (Strategy pattern)
# ------------------------------
class PlotlyRenderer:
    """
    Plotly-based renderer: builds the 3D figure. Implemented as a Strategy so you can
    swap in a Matplotlib renderer later without changing controller logic.
    """

    def __init__(self, dark_bg: bool = True):
        self.dark_bg = dark_bg

    @staticmethod
    def scale_for_visibility(df: pd.DataFrame, factor: float) -> pd.DataFrame:
        """Apply a simple scaling to center-based coordinates so constellations spread out.

        Why: The HYG 3D coordinates near the POV are dense; scaling helps visual separation
        while maintaining relative geometry.
        """
        scaled = df.copy()
        scaled['cx_scaled'] = scaled['cx'] * factor
        scaled['cy_scaled'] = scaled['cy'] * factor
        scaled['cz_scaled'] = scaled['cz'] * factor
        return scaled

    @staticmethod
    def mag_to_marker_size(mag: float) -> float:
        """Convert magnitude to a marker size (pixels). Lower mag -> brighter -> bigger."""
        try:
            m = float(mag)
        except Exception:
            m = 10.0
        # exponential-ish scale, clipped
        return max(2.0, min(50.0, 10.0 * 10**(-0.4 * (m - (-1.0)))))

    def build_figure(self,
                     stars_df: pd.DataFrame,
                     constellations: List[Dict[str, Any]],
                     scale_factor: float = 20.0,
                     show_labels: bool = True) -> go.Figure:
        """Create a Plotly 3D figure showing stars and constellation lines.

        Why: Renderer centralizes all plotting options and trace creation.
        """
        # 1) scale coordinates for visibility
        s = self.scale_for_visibility(stars_df, scale_factor)

        # 2) build HIP -> coordinates map (for constellation lookup)
        hip_map: Dict[int, Tuple[float,float,float]] = {}
        for _, row in s.iterrows():
            hip = row.get('hip')
            if pd.notna(hip):
                hip_map[int(hip)] = (row['cx_scaled'], row['cy_scaled'], row['cz_scaled'])

        # 3) create star scatter trace
        # compute sizes and colors arrays (Plotly accepts arrays per-point)
        sizes = [self.mag_to_marker_size(m) for m in s['mag'].fillna(10).values]
        # We will map constellation membership to color intensity: const member -> brighter color index
        # simple mapping: 1.0 for in-constellation, 0.3 for background
        in_const = s.get('in_constellation', pd.Series([False]*len(s)))
        colors = [1.0 if ic else 0.3 for ic in in_const]

        hover_texts = s.apply(
            lambda r: (
                f"{(r['proper'] or r['bf'] or ('HIP ' + str(int(r['hip'])) if pd.notna(r['hip']) else 'Unnamed'))}<br>"
                f"Mag: {r['mag']:.2f} Dist(pc): {r['dist']:.1f}"
            ),
            axis=1
        )

        star_scatter = go.Scatter3d(
            x=s['cx_scaled'],
            y=s['cy_scaled'],
            z=s['cz_scaled'],
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                colorscale='Viridis',
                opacity=0.85,
                colorbar=dict(title='Const membership'),
                cmin=0,
                cmax=1,
                line=dict(width=0)
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>"
        )

        # 4) constellation line traces and label traces
        line_traces = []
        label_traces = []
        for const in constellations:
            # each const["lines"] is a list of sequences (each sequence is a "polyline" listing HIPs)
            for seq in const.get('lines', []):
                xs, ys, zs = [], [], []
                for hip in seq:
                    if hip in hip_map:
                        x,y,z = hip_map[hip]
                        xs.append(x); ys.append(y); zs.append(z)
                    else:
                        # missing hip -> break the polyline (lift pen)
                        if len(xs) > 1:
                            line_traces.append(go.Scatter3d(
                                x=xs, y=ys, z=zs, mode='lines',
                                line=dict(color='white', width=2),
                                hoverinfo='none'
                            ))
                        xs=[]; ys=[]; zs=[]
                # flush any remaining
                if len(xs) > 1:
                    line_traces.append(go.Scatter3d(
                        x=xs, y=ys, z=zs, mode='lines',
                        line=dict(color='white', width=2),
                        hoverinfo='none'
                    ))

            # constellation label: place roughly at centroid of all available points for the constellation
            all_points = []
            for seq in const.get('lines', []):
                for hip in seq:
                    if hip in hip_map:
                        all_points.append(hip_map[hip])
            if all_points and show_labels:
                xs_all, ys_all, zs_all = zip(*all_points)
                label_traces.append(
                    go.Scatter3d(
                        x=[np.mean(xs_all)],
                        y=[np.mean(ys_all)],
                        z=[np.mean(zs_all)],
                        mode='text',
                        text=[const.get("iau", "") or const.get("id","")],
                        textfont=dict(color="yellow", size=12),
                        hoverinfo='none'
                    )
                )

        # 5) combine traces into a figure
        fig = go.Figure(data=[star_scatter] + line_traces + label_traces)
        # layout tuned for dark sky
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X', showbackground=False, visible=False),
                yaxis=dict(title='Y', showbackground=False, visible=False),
                zaxis=dict(title='Z', showbackground=False, visible=False),
                bgcolor='black'
            ),
            paper_bgcolor='black' if self.dark_bg else 'white',
            font=dict(color='white' if self.dark_bg else 'black'),
            height=800,
            margin=dict(l=0,r=0,t=40,b=0)
        )
        return fig

# ------------------------------
# Controller (application logic)
# ------------------------------
class SkyApp:
    """Controller that wires model, data factory, renderer, and Streamlit UI together."""

    def __init__(self):
        # Create and prepare model using factory-loaded data
        raw_stars = DataFactory.load_stars()
        self.model = StarModel(raw_stars)
        self.model.prepare()

        # load constellations JSON
        self.constellations = DataFactory.load_constellations()

        # instantiate renderer strategy
        self.renderer = PlotlyRenderer(dark_bg=True)

        # small list of interesting stars (for quick POV changes)
        # Why: clicking dots reliably is tricky in Streamlit; a selectbox provides stable UX.
        self.interesting = ["Polaris", "Sirius", "Betelgeuse", "Vega", "Rigel", "Canopus", "Alpha Centauri", "Sol"]

    def run(self):
        """Build UI, handle inputs, and render the figure."""
        st.header("⭐ Interactive Star Atlas — Patterned Edition")

        # Left column for controls
        col_left, col_right = st.columns([1,3])
        with col_left:
            # Slider for distance (keeps application simple; model filtering uses dist_ly)
            distance_ly = st.slider("Max distance (light years)", 10, 5000, 500, 10)
            # helper selectbox for quick POV change (design: improving UX)
            quick = st.selectbox("Quick: choose an interesting star (changes search box)", [""] + self.interesting)
            # search text input (user may type exact name or HIP)
            query = st.text_input("Target star (name or HIP)", value="Polaris")
            # If user picked quick, override query input so they don't have to type
            if quick:
                query = quick

            # visualization controls
            scale_factor = st.slider("Spread scale factor", 1.0, 80.0, 20.0, 1.0)
            show_labels = st.checkbox("Show constellation labels", True)
            show_lines = st.checkbox("Show constellation lines", True)
            # small explanation area
            st.markdown("**Tip:** use the Quick select or type a star name (e.g. Sirius) to change POV.")

        # Choose a star using model
        matches = self.model.find_star(query)
        if matches.empty:
            st.warning("No match found — showing default (top bright stars).")
            # fallback POV: center at origin (0,0,0)
            cx = cy = cz = 0.0
            target = None
        else:
            # take first match as the chosen POV star
            # take first match as the chosen POV star
            target = matches.iloc[0]

            # Build the display name safely without nested f-strings
            proper = target.get("proper", "")
            bf = target.get("bf", "")
            hip = target.get("hip", None)

            if proper:
                display_name = proper
            elif bf:
                display_name = bf
            elif pd.notna(hip):
                display_name = f"HIP {int(hip)}"
            else:
                display_name = "Unnamed"

            # UX: tell the user what we're centered on
            st.success(f"Centered on: **{display_name}**")

            # Recenter POV using original coordinates
            cx = null_values_exception(target.get("x", 0.0), 0.0)
            cy = null_values_exception(target.get("y", 0.0), 0.0)
            cz = null_values_exception(target.get("z", 0.0), 0.0)

        # Recenter stars around chosen POV
        df = self.model.df.copy()
        # If x,y,z are NaN for the catalog, fallback to RA/Dec-based "flat" placement:
        if df[['x','y','z']].isna().all(axis=None):
            # Create a pseudo-3D by converting RA/Dec to a unit sphere coordinates
            # Why: ensures map works even if original 3D coords missing.
            ra_rad = np.deg2rad(df['ra_deg'].fillna(0.0).values)
            dec_rad = np.deg2rad(df['dec'].fillna(0.0).values)
            # unit sphere coordinates
            df['x'] = np.cos(dec_rad) * np.cos(ra_rad)
            df['y'] = np.cos(dec_rad) * np.sin(ra_rad)
            df['z'] = np.sin(dec_rad)

        # Subtract center coordinates to recenter on selected star
        df['cx'] = df['x'] - cx
        df['cy'] = df['y'] - cy
        df['cz'] = df['z'] - cz

        # Filter by chosen distance (light years)
        df = df[df['dist_ly'].fillna(1e9) <= distance_ly].copy()

        # Identify constellation membership (fast set build)
        const_hips = set()
        for c in self.constellations:
            for seq in c.get('lines', []):
                for hip in seq:
                    const_hips.add(hip)
        df['in_constellation'] = df['hip'].apply(lambda x: bool(x in const_hips) if pd.notna(x) else False)

        # Build figure with renderer (lines toggled by show_lines)
        fig = self.renderer.build_figure(df, self.constellations if show_lines else [], scale_factor=scale_factor, show_labels=show_labels)

        # Display the Plotly figure
        st.plotly_chart(fig, use_container_width=True)

        # Show a small detail panel for the chosen star (if any)
        # --- Target Star Details Section (Corrected for Python 3.11) ---
        if target is not None:
            st.subheader("Target Star Details")

            # Name: proper name, Bayer-Flamsteed, or Unnamed
            name = target.get("proper", "")
            if not name:
                name = "Unnamed"
            st.write(f"**Name:** {name}")

            # Bayer/Flamsteed ID
            bf_value = target.get("bf", "")
            st.write(f"**Bayer/Flamsteed:** {bf_value}")

            # HIP ID (must check for NaN)
            hip_value = target.get("hip", None)
            if hip_value is not None and pd.notna(hip_value):
                hip_int = int(hip_value)
                st.write(f"**HIP:** {hip_int}")

            # Distance
            dist_val = target.get("dist", np.nan)
            if pd.notna(dist_val):
                st.write(f"**Distance (pc):** {dist_val:.2f}")

            # Apparent magnitude
            mag_val = target.get("mag", np.nan)
            if pd.notna(mag_val):
                st.write(f"**Magnitude:** {mag_val:.2f}")

            # Spectral type
            spect_val = target.get("spect", "")
            if spect_val:
                st.write(f"**Spectral Type:** {spect_val}")
            else:
                st.write("**Spectral Type:** Unknown")


# ------------------------------
# Run the app
# ------------------------------
if __name__ == "__main__":
    app = SkyApp()
    app.run()
