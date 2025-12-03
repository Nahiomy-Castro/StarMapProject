import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import json

# -------------------------------
# Load star data
# -------------------------------
@st.cache_data
def load_stars():
    stars = pd.read_csv("hyg_v42.csv")
    stars['proper'] = stars['proper'].fillna("")
    stars['bf'] = stars['bf'].fillna("")
    return stars

stars = load_stars()

# -------------------------------
# Load constellation JSON
# -------------------------------
@st.cache_data
def load_constellations():
    with open("constellations.json") as f:
        data = json.load(f)
    return data["constellations"]

constellations = load_constellations()

# -------------------------------
# App Title
# -------------------------------
st.title("⭐ Interactive Star Atlas with Constellations")
st.write("Interesting Stars: Sol, Sirius, Polaris, Betelgeuse, Vega, Rigel, Canopus, Alpha Centauri")

# -------------------------------
# User controls
# -------------------------------
distance_ly = st.slider("Distance from target star (light years)", min_value=10, max_value=500, value=100, step=10)
distance_pc = distance_ly / 3.26  # Convert to parsecs

query = st.text_input("Target star (name, HIP ID, or Bayer/Flamsteed ID)", "Polaris")

# -------------------------------
# Find target star
# -------------------------------
def find_star(q):
    q = q.lower()
    matches = stars[
        stars['proper'].str.lower().str.contains(q) |
        stars['bf'].str.lower().str.contains(q) |
        stars['hip'].astype(str).str.contains(q)
    ]
    return matches

result = find_star(query)

if len(result) == 0:
    st.error("No star found. Try another name or catalog number.")
    cx = cy = cz = 0
else:
    target = result.iloc[0]
    st.success(f"Centered on: **{target['proper'] or target['bf'] or ('HIP ' + str(target['hip']))}**")
    cx, cy, cz = target['x'], target['y'], target['z']

# -------------------------------
# Recenter stars
# -------------------------------
stars['cx'] = stars['x'] - cx
stars['cy'] = stars['y'] - cy
stars['cz'] = stars['z'] - cz

# Filter by distance
stars['dist_from_target'] = np.sqrt(stars['cx']**2 + stars['cy']**2 + stars['cz']**2)
stars_filtered = stars[stars['dist_from_target'] <= distance_pc].copy()

# -------------------------------
# Apply scale to spread stars out
# -------------------------------
scale_factor = 20  # adjust for visibility
stars_filtered['cx_scaled'] = stars_filtered['cx'] * scale_factor
stars_filtered['cy_scaled'] = stars_filtered['cy'] * scale_factor
stars_filtered['cz_scaled'] = stars_filtered['cz'] * scale_factor

# -------------------------------
# Map HIP numbers for constellation lines
# -------------------------------
hip_to_star = {
    int(row['hip']): (row['cx_scaled'], row['cy_scaled'], row['cz_scaled'])
    for _, row in stars_filtered.iterrows() if pd.notna(row['hip'])
}

# -------------------------------
# Star marker sizes and colors
# -------------------------------
def get_marker_size(mag):
    return max(2, 10 - mag)

def get_marker_color(mag, in_constellation):
    return 1.0 - (mag / 10) if in_constellation else 0.3

const_hips = set()
for const in constellations:
    for line in const["lines"]:
        for hip in line:
            const_hips.add(hip)

stars_filtered['in_constellation'] = stars_filtered['hip'].apply(lambda x: x in const_hips if pd.notna(x) else False)
stars_filtered['marker_size'] = stars_filtered['mag'].apply(get_marker_size)
stars_filtered['marker_color'] = stars_filtered.apply(lambda r: get_marker_color(r['mag'], r['in_constellation']), axis=1)

# -------------------------------
# 3D scatter for stars
# -------------------------------
star_scatter = go.Scatter3d(
    x=stars_filtered['cx_scaled'],
    y=stars_filtered['cy_scaled'],
    z=stars_filtered['cz_scaled'],
    mode='markers',
    marker=dict(
        size=stars_filtered['marker_size'],
        color=stars_filtered['marker_color'],
        colorscale='Viridis',
        opacity=0.8,
        colorbar=dict(title='Brightness'),
        cmin=0,
        cmax=1
    ),
    text=stars_filtered.apply(lambda r: r['proper'] or r['bf'] or f"HIP {r['hip']}", axis=1),
    hovertemplate="%{text}<br>Mag: %{marker.size}<extra></extra>"
)

# -------------------------------
# Constellation lines
# -------------------------------
line_traces = []
label_traces = []

for const in constellations:
    # Draw lines
    for line in const["lines"]:
        coords_x, coords_y, coords_z = [], [], []
        for hip in line:
            if hip in hip_to_star:
                x, y, z = hip_to_star[hip]
                coords_x.append(x)
                coords_y.append(y)
                coords_z.append(z)
        if len(coords_x) > 1:
            line_traces.append(
                go.Scatter3d(
                    x=coords_x, y=coords_y, z=coords_z,
                    mode='lines',
                    line=dict(color='white', width=2),
                    hoverinfo='none'
                )
            )
    # Add constellation label at center of first line
    all_coords = [hip_to_star[hip] for line in const["lines"] for hip in line if hip in hip_to_star]
    if all_coords:
        xs, ys, zs = zip(*all_coords)
        label_traces.append(
            go.Scatter3d(
                x=[np.mean(xs)],
                y=[np.mean(ys)],
                z=[np.mean(zs)],
                mode='text',
                text=[const.get("iau", "")],
                textposition="top center",
                textfont=dict(color="yellow", size=12),
                hoverinfo='none'
            )
        )

# -------------------------------
# Combine and plot
# -------------------------------
fig = go.Figure(data=[star_scatter] + line_traces + label_traces)
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X', showbackground=False),
        yaxis=dict(title='Y', showbackground=False),
        zaxis=dict(title='Z', showbackground=False),
        bgcolor='black'
    ),
    paper_bgcolor='black',
    font=dict(color='white'),
    height=800
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Display target star details
# -------------------------------
if len(result) > 0:
    st.subheader("Star Details")
    st.write(f"**Name:** {target['proper'] or 'Unnamed'}")
    st.write(f"**Bayer/Flamsteed:** {target['bf']}")
    st.write(f"**HIP:** {target['hip']}")
    st.write(f"**Distance (pc):** {target['dist']}")
    st.write(f"**Magnitude:** {target['mag']}")
    st.write(f"**Spectral Type:** {target['spect']}")
    st.write(f"**Constellation:** {target['con']}")
